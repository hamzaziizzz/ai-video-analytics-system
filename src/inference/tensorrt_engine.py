import time
from typing import List

import numpy as np

from .base import Detection, InferenceEngine
from .yolo import decode_yolo_output, prepare_input
from ..utils.logging import get_logger


class TensorRTYoloEngine(InferenceEngine):
    def __init__(
        self,
        engine_path: str,
        labels: List[str],
        input_size,
        confidence_threshold,
        nms_iou_threshold,
        has_objectness: bool,
        end_to_end: bool,
        debug_log_raw_output: bool,
        debug_log_raw_interval_seconds: float,
        debug_log_raw_rows: int,
        debug_log_raw_cols: int,
    ) -> None:
        self.engine_path = engine_path
        self.labels = labels
        self.input_size = tuple(input_size)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.has_objectness = has_objectness
        self.end_to_end = end_to_end
        self.debug_log_raw_output = debug_log_raw_output
        self.debug_log_raw_interval_seconds = debug_log_raw_interval_seconds
        self.debug_log_raw_rows = debug_log_raw_rows
        self.debug_log_raw_cols = debug_log_raw_cols
        self.logger = get_logger("inference.tensorrt")
        self.engine = None
        self.context = None
        self.bindings = []
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.stream = None
        self.input_index = None
        self.output_indices = []
        self.dynamic_outputs = False
        self.output_dtypes = []
        self.use_io_tensors = False
        self.input_name = None
        self.input_nbytes = 0
        self.output_names = []
        self.output_shapes = []
        self.cuda_context = None
        self._last_raw_log_ts = 0.0
        self.input_dtype = np.float32

    def load(self) -> None:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError("TensorRT and PyCUDA are required for TensorRT inference") from exc

        cuda.init()
        device_count = cuda.Device.count()
        if device_count < 1:
            raise RuntimeError("No CUDA devices found for TensorRT inference")
        device = cuda.Device(0)
        self.cuda_context = device.make_context()

        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(self.engine_path, "rb") as engine_file, trt.Runtime(logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        except Exception:
            self.close()
            raise

        if self.engine is None:
            self.close()
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()

        if hasattr(self.engine, "num_bindings"):
            self.bindings = [None] * self.engine.num_bindings

            for binding in range(self.engine.num_bindings):
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                shape = self.engine.get_binding_shape(binding)
                if -1 in shape:
                    if self.engine.binding_is_input(binding):
                        shape = (1, 3, self.input_size[0], self.input_size[1])
                    else:
                        self.dynamic_outputs = True
                size = int(np.prod(shape))
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[binding] = int(device_mem)
                if self.engine.binding_is_input(binding):
                    self.input_index = binding
                    self.input_dtype = dtype
                    self.host_inputs.append(host_mem)
                    self.device_inputs.append(device_mem)
                else:
                    self.output_indices.append(binding)
                    if not self.dynamic_outputs:
                        self.host_outputs.append(host_mem)
                        self.device_outputs.append(device_mem)
                        self.output_dtypes.append(dtype)
        else:
            self.use_io_tensors = True
            tensor_count = self.engine.num_io_tensors
            for idx in range(tensor_count):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                if mode == trt.TensorIOMode.INPUT:
                    if self.input_name is None:
                        self.input_name = name
                        self.input_dtype = dtype
                    else:
                        self.logger.warning("Multiple inputs detected; using %s", self.input_name)
                else:
                    self.output_names.append(name)
                    self.output_dtypes.append(dtype)

        try:
            self.stream = cuda.Stream()
            self.logger.info("TensorRT using CUDA device: %s", device.name())
            self.logger.info("Loaded TensorRT engine")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.cuda_context is None:
            return
        try:
            import gc
            import pycuda.driver as cuda
        except Exception:
            self.cuda_context = None
            return

        current = cuda.Context.get_current()
        pushed = False
        if current is None or current != self.cuda_context:
            try:
                self.cuda_context.push()
                pushed = True
            except Exception:
                pass

        try:
            if self.stream is not None:
                try:
                    self.stream.synchronize()
                except Exception:
                    pass
        finally:
            self.stream = None

        self.context = None
        self.engine = None
        self.bindings = []
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.output_indices = []
        self.output_names = []
        self.output_shapes = []

        gc.collect()

        try:
            if pushed or cuda.Context.get_current() == self.cuda_context:
                self.cuda_context.pop()
        except Exception:
            pass
        try:
            self.cuda_context.detach()
        except Exception:
            pass
        self.cuda_context = None

    def infer(self, frame) -> List[Detection]:
        if self.context is None:
            raise RuntimeError("TensorRT engine is not loaded")

        import pycuda.driver as cuda

        input_tensor, meta = prepare_input(frame, self.input_size)
        input_data = np.ascontiguousarray(input_tensor.astype(self.input_dtype, copy=False))

        outputs = []
        if self.use_io_tensors:
            if self.input_name is None:
                raise RuntimeError("TensorRT input tensor not initialized")

            self.context.set_input_shape(self.input_name, input_data.shape)

            if not self.device_inputs or input_data.nbytes > self.input_nbytes:
                self.device_inputs = [cuda.mem_alloc(input_data.nbytes)]
                self.input_nbytes = input_data.nbytes

            self.context.set_tensor_address(self.input_name, int(self.device_inputs[0]))
            cuda.memcpy_htod_async(self.device_inputs[0], input_data, self.stream)

            output_buffers = []
            for idx, name in enumerate(self.output_names):
                out_shape = tuple(self.context.get_tensor_shape(name))
                dtype = self.output_dtypes[idx]
                size = int(np.prod(out_shape))

                if idx >= len(self.output_shapes) or out_shape != self.output_shapes[idx]:
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    if idx >= len(self.output_shapes):
                        self.host_outputs.append(host_mem)
                        self.device_outputs.append(device_mem)
                        self.output_shapes.append(out_shape)
                    else:
                        self.host_outputs[idx] = host_mem
                        self.device_outputs[idx] = device_mem
                        self.output_shapes[idx] = out_shape

                self.context.set_tensor_address(name, int(self.device_outputs[idx]))
                output_buffers.append((self.host_outputs[idx], self.device_outputs[idx], out_shape))

            self.context.execute_async_v3(stream_handle=self.stream.handle)

            for host_mem, device_mem, out_shape in output_buffers:
                cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)
                outputs.append(np.array(host_mem).reshape(out_shape))

            self.stream.synchronize()
        else:
            if self.input_index is None:
                raise RuntimeError("TensorRT input binding not initialized")

            self.context.set_binding_shape(self.input_index, input_data.shape)
            cuda.memcpy_htod_async(self.device_inputs[0], input_data, self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            if self.dynamic_outputs and not self.host_outputs:
                for binding in self.output_indices:
                    out_shape = tuple(self.context.get_binding_shape(binding))
                    dtype = np.float32
                    size = int(np.prod(out_shape))
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.bindings[binding] = int(device_mem)
                    self.host_outputs.append(host_mem)
                    self.device_outputs.append(device_mem)
                    self.output_dtypes.append(dtype)

            for idx, binding in enumerate(self.output_indices):
                host_mem = self.host_outputs[idx]
                device_mem = self.device_outputs[idx]
                out_shape = self.context.get_binding_shape(binding)
                cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)
                outputs.append(np.array(host_mem).reshape(out_shape))

            self.stream.synchronize()

        output = outputs[0] if outputs else np.array([])
        self._maybe_log_raw_output(output)
        return decode_yolo_output(
            output,
            labels=self.labels,
            confidence_threshold=self.confidence_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            has_objectness=self.has_objectness,
            meta=meta,
            end_to_end=self.end_to_end,
        )

    def _maybe_log_raw_output(self, output: np.ndarray) -> None:
        if not self.debug_log_raw_output:
            return

        now = time.time()
        if now - self._last_raw_log_ts < self.debug_log_raw_interval_seconds:
            return
        self._last_raw_log_ts = now

        shape = output.shape
        try:
            stats = {
                "min": float(np.min(output)),
                "max": float(np.max(output)),
                "mean": float(np.mean(output)),
            }
        except Exception:
            stats = {"min": None, "max": None, "mean": None}

        sample = self._sample_rows(output)
        self.logger.info("Raw output shape=%s stats=%s sample=%s", shape, stats, sample)

    def _sample_rows(self, output: np.ndarray):
        if output.ndim == 0:
            return []
        if output.ndim == 1:
            data = output
        else:
            data = output[0] if output.ndim > 1 else output

        if data.ndim == 1:
            return [self._round_list(data[: self.debug_log_raw_cols])]

        rows = min(self.debug_log_raw_rows, data.shape[0])
        cols = min(self.debug_log_raw_cols, data.shape[1])
        return [self._round_list(row[:cols]) for row in data[:rows]]

    def _round_list(self, values):
        return [round(float(v), 4) for v in values]
