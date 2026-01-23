import time
from typing import List

import numpy as np

from .base import Detection, InferenceEngine
from .yolo import decode_yolo_output, prepare_batch
from ..utils.logging import get_logger


class TensorRTYoloEngine(InferenceEngine):
    """TensorRT-based YOLO engine for high-throughput GPU inference."""
    def __init__(
        self,
        engine_path: str,
        labels: List[str],
        input_size,
        confidence_threshold,
        nms_iou_threshold,
        has_objectness: bool,
        end_to_end: bool,
        device: str,
        fp16: bool,
        int8: bool,
        class_id_filter: List[int] | None,
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
        self.device = device
        self.fp16 = fp16
        self.int8 = int8
        self.class_id_filter = class_id_filter
        self.debug_log_raw_output = debug_log_raw_output
        self.debug_log_raw_interval_seconds = debug_log_raw_interval_seconds
        self.debug_log_raw_rows = debug_log_raw_rows
        self.debug_log_raw_cols = debug_log_raw_cols
        self.logger = get_logger("inference.tensorrt")
        self.engine = None
        self.context = None
        self.bindings = []
        self.stream = None
        self.input_index = None
        self.output_indices = []
        self.output_dtypes = []
        self.input_name = None
        self.output_names = []
        self.output_shapes = []
        self.input = None
        self.outputs = []
        self.bindings = []
        self.is_trt10 = False
        self.max_batch_size = 1
        self._last_raw_log_ts = 0.0
        self.input_dtype = np.float32
        self.output_layout = ""
        self.use_cupy_nms = True

    def load(self) -> None:
        try:
            import tensorrt as trt
            import cupy as cp
            import cupyx
        except ImportError as exc:
            raise RuntimeError("TensorRT and CuPy are required for TensorRT inference") from exc

        device_id = 0
        if self.device and "cuda" in str(self.device).lower():
            try:
                device_id = int(str(self.device).split(":")[1])
            except (IndexError, ValueError):
                device_id = 0
        cp.cuda.Device(device_id).use()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as engine_file, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.is_trt10 = not hasattr(self.engine, "num_bindings")

        class HostDeviceMem:
            def __init__(self, size, dtype):
                if size <= 0:
                    raise RuntimeError(f"Invalid TensorRT buffer size {size}; engine profile may be corrupt")
                self.size = size
                self.dtype = dtype
                self.host = cupyx.zeros_pinned(size, dtype)
                self.device = cp.zeros(size, dtype)

            @property
            def devptr(self):
                return self.device.data.ptr

            def copy_dtoh_async(self, stream):
                self.device.data.copy_to_host_async(self.host.ctypes.data, self.host.nbytes, stream)

        if self.is_trt10:
            input_profile_shape = None
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                mode = self.engine.get_tensor_mode(name)
                shape = self._dims_to_tuple(self.engine.get_tensor_shape(name), name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                    profile_shape = shape
                    if -1 in shape:
                        profile_shape = self._dims_to_tuple(self.engine.get_tensor_profile_shape(name, 0)[2], name)
                    self._validate_shape(profile_shape, name)
                    input_profile_shape = profile_shape
                    self.max_batch_size = profile_shape[0]
                    self.input_dtype = dtype
                else:
                    self.output_names.append(name)
                    self.output_dtypes.append(dtype)

            if not self.input_name or input_profile_shape is None:
                raise RuntimeError("TensorRT engine has no valid input tensor")

            self.context.set_input_shape(self.input_name, input_profile_shape)
            self.input = HostDeviceMem(int(np.prod(input_profile_shape)), self.input_dtype)
            self.bindings.append(self.input.devptr)

            for idx, name in enumerate(self.output_names):
                shape = self._dims_to_tuple(self.context.get_tensor_shape(name), name)
                if any(dim <= 0 for dim in shape):
                    try:
                        shape = self._dims_to_tuple(self.engine.get_tensor_profile_shape(name, 0)[2], name)
                    except Exception:
                        pass
                self._validate_shape(shape, name)
                buffer = HostDeviceMem(int(np.prod(shape)), self.output_dtypes[idx])
                self.outputs.append(buffer)
                self.output_shapes.append(shape)
                self.bindings.append(buffer.devptr)
        else:
            self.max_batch_size = self.engine.get_profile_shape(0, 0)[2][0]
            for binding in self.engine:
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                shape = self._dims_to_tuple(self.engine.get_binding_shape(binding), binding)
                if -1 in shape:
                    shape = (self.max_batch_size,) + shape[1:]
                self._validate_shape(shape, binding)
                buffer = HostDeviceMem(int(np.prod(shape)), dtype)
                self.bindings.append(buffer.devptr)
                if self.engine.binding_is_input(binding):
                    self.input_index = int(self.engine.get_binding_index(binding))
                    self.input = buffer
                    self.input_dtype = dtype
                else:
                    self.output_indices.append(int(self.engine.get_binding_index(binding)))
                    self.outputs.append(buffer)
                    self.output_shapes.append(shape)
                    self.output_dtypes.append(dtype)

        self.logger.info("TensorRT using CUDA device: cuda:%d", device_id)
        self.logger.info("Loaded TensorRT engine")
        if self.fp16 or self.int8:
            self.logger.info(
                "TensorRT precision requested fp16=%s int8=%s (engine must be built accordingly)",
                self.fp16,
                self.int8,
            )

        self.output_layout = self._resolve_output_layout()

    def _validate_shape(self, shape, name: str) -> None:
        if any(dim <= 0 for dim in shape):
            raise RuntimeError(
                f"Invalid TensorRT tensor shape for '{name}': {shape}. "
                "Delete the engine file and rebuild with a valid optimization profile."
            )

    def close(self) -> None:
        self.stream = None
        self.context = None
        self.engine = None
        self.bindings = []
        self.output_indices = []
        self.output_names = []
        self.output_shapes = []
        self.outputs = []
        self.input = None

    def _resolve_output_layout(self) -> str:
        try:
            from ..utils.model_registry import get_registry
        except Exception:
            return ""
        registry = get_registry()
        model_name = registry.resolve_model_name(self.engine_path)
        return registry.output_layout(model_name) or ""

    def _dims_to_tuple(self, dims, name: str) -> tuple[int, ...]:
        try:
            count = getattr(dims, "nbDims", None)
            if count is None:
                count = getattr(dims, "nb_dims", None)
            if count is None:
                return tuple(int(v) for v in dims)
            return tuple(int(dims[i]) for i in range(int(count)))
        except Exception as exc:
            raise RuntimeError(f"Failed to read TensorRT dims for '{name}': {dims}") from exc

    def infer(self, frame) -> List[Detection]:
        return self.infer_batch([frame])[0]

    def infer_batch(self, frames) -> List[List[Detection]]:
        if self.context is None or self.input is None:
            raise RuntimeError("TensorRT engine is not loaded")
        if not frames:
            return []

        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("CuPy is required for TensorRT inference") from exc

        if len(frames) > self.max_batch_size:
            raise RuntimeError(f"Batch size {len(frames)} exceeds engine max {self.max_batch_size}")

        batch_imgs, metas = prepare_batch(frames, self.input_size)
        with self.stream:
            g_img = cp.asarray(batch_imgs)
            g_img = g_img[..., ::-1]
            g_img = cp.transpose(g_img, (0, 3, 1, 2))
            g_img = cp.asarray(g_img, dtype=cp.float32) / 255.0
            self.input.device[: g_img.size] = g_img.ravel()
        infer_shape = tuple(g_img.shape)

        if self.is_trt10:
            self.context.set_input_shape(self.input_name, infer_shape)
            for idx, name in enumerate(self.output_names):
                self.context.set_tensor_address(name, self.outputs[idx].devptr)
            self.context.set_tensor_address(self.input_name, self.input.devptr)
            self.context.execute_async_v3(stream_handle=self.stream.ptr)
            for out in self.outputs:
                out.copy_dtoh_async(self.stream)
            self.stream.synchronize()
            output_shapes = [tuple(self.context.get_tensor_shape(name)) for name in self.output_names]
            outputs = []
            for idx, out in enumerate(self.outputs):
                shape = output_shapes[idx]
                size = int(np.prod(shape))
                outputs.append(out.host[:size].reshape(shape))
        else:
            self.context.set_binding_shape(self.input_index, infer_shape)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.ptr)
            for out in self.outputs:
                out.copy_dtoh_async(self.stream)
            self.stream.synchronize()
            output_shapes = [tuple(self.context.get_binding_shape(binding)) for binding in self.output_indices]
            outputs = []
            for idx, out in enumerate(self.outputs):
                shape = output_shapes[idx]
                size = int(np.prod(shape))
                outputs.append(out.host[:size].reshape(shape))

        output = outputs[0] if outputs else np.array([])
        self._maybe_log_raw_output(output)
        batch_dets: List[List[Detection]] = []
        if output.ndim >= 3:
            for idx in range(len(frames)):
                batch_dets.append(
                    decode_yolo_output(
                        output[idx],
                        labels=self.labels,
                        confidence_threshold=self.confidence_threshold,
                        nms_iou_threshold=self.nms_iou_threshold,
                        has_objectness=self.has_objectness,
                        meta=metas[idx],
                        end_to_end=self.end_to_end,
                        output_layout=self.output_layout,
                        class_id_filter=self.class_id_filter,
                        use_cupy_nms=self.use_cupy_nms,
                    )
                )
        else:
            batch_dets.append(
                decode_yolo_output(
                    output,
                    labels=self.labels,
                    confidence_threshold=self.confidence_threshold,
                    nms_iou_threshold=self.nms_iou_threshold,
                    has_objectness=self.has_objectness,
                    meta=metas[0],
                    end_to_end=self.end_to_end,
                    output_layout=self.output_layout,
                    class_id_filter=self.class_id_filter,
                    use_cupy_nms=self.use_cupy_nms,
                )
            )
        return batch_dets

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
