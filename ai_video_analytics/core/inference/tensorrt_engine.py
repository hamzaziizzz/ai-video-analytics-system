import time
from typing import List

import numpy as np

from .base import Detection, InferenceEngine
from .yolo import DecodeWorkspace, PreprocessMeta, decode_yolo_output, prepare_batch, prepare_batch_no_letterbox
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
        use_cupy_nms: bool = False,
        use_gpu_preproc: bool = False,
        use_numba_decode: bool = True,
        no_letterbox: bool = False,
        gpu_timing: bool = True,
        return_keypoints: bool = False,
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
        self.use_cupy_nms = bool(use_cupy_nms)
        self.use_gpu_preproc = bool(use_gpu_preproc)
        self.use_numba_decode = bool(use_numba_decode)
        self.no_letterbox = bool(no_letterbox)
        self.gpu_timing = bool(gpu_timing)
        self.return_keypoints = bool(return_keypoints)
        self._host_batch = None
        self._host_batch_shape = None
        self._gpu_upload = []
        self._gpu_letterbox = []
        self._use_gpu_preproc = False
        self._cv_stream = None
        self._preproc_event = None
        self._decode_workspace = DecodeWorkspace()
        self.last_batch_timings = None
        self._gpu_pre_start = None
        self._gpu_pre_end = None
        self._gpu_infer_start = None
        self._gpu_infer_end = None

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

        self._use_gpu_preproc = False
        self._cv_stream = None
        self._preproc_event = None
        if self.use_gpu_preproc:
            try:
                import cv2

                if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self._use_gpu_preproc = True
                else:
                    self.logger.warning("TRT GPU preproc requested but OpenCV CUDA is unavailable; falling back to CPU.")
            except Exception:
                self.logger.warning("TRT GPU preproc requested but OpenCV is unavailable; falling back to CPU.")

        if self._use_gpu_preproc:
            try:
                import cv2

                self._cv_stream = cv2.cuda.Stream()
            except Exception:
                self._cv_stream = None

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as engine_file, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.is_trt10 = not hasattr(self.engine, "num_bindings")
        if self._use_gpu_preproc and self._cv_stream is not None:
            stream_ptr = None
            for attr in ("cudaPtr", "ptr"):
                if hasattr(self._cv_stream, attr):
                    try:
                        value = getattr(self._cv_stream, attr)
                        stream_ptr = int(value() if callable(value) else value)
                        break
                    except Exception:
                        continue
            if stream_ptr:
                self.stream = cp.cuda.ExternalStream(stream_ptr)
            else:
                self._cv_stream = None
                self._preproc_event = cp.cuda.Event()
        elif self._use_gpu_preproc:
            self._preproc_event = cp.cuda.Event()

        if self.gpu_timing:
            # CUDA events provide GPU-side timings for preproc/infer stages.
            self._gpu_pre_start = cp.cuda.Event()
            self._gpu_pre_end = cp.cuda.Event()
            self._gpu_infer_start = cp.cuda.Event()
            self._gpu_infer_end = cp.cuda.Event()

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
        self.logger.info(f"Loaded TensorRT engine - {self.engine_path}")
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
        self._host_batch = None
        self._host_batch_shape = None
        self._gpu_upload = []
        self._gpu_letterbox = []
        self._cv_stream = None
        self._preproc_event = None
        self._gpu_pre_start = None
        self._gpu_pre_end = None
        self._gpu_infer_start = None
        self._gpu_infer_end = None

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

        t_pre0 = time.perf_counter()
        gpu_pre_ms = None
        if self._use_gpu_preproc:
            try:
                metas, infer_shape = self._prepare_batch_gpu(frames)
            except Exception as exc:
                self.logger.warning("TRT GPU preproc failed (%s); falling back to CPU.", exc)
                self._use_gpu_preproc = False
                self.stream = cp.cuda.Stream(non_blocking=True)
                metas, infer_shape = self._prepare_batch_cpu(frames)
        else:
            metas, infer_shape = self._prepare_batch_cpu(frames)
        t_pre1 = time.perf_counter()
        if self.gpu_timing and self._use_gpu_preproc and self._gpu_pre_start is not None and self._gpu_pre_end is not None:
            try:
                gpu_pre_ms = float(cp.cuda.get_elapsed_time(self._gpu_pre_start, self._gpu_pre_end))
            except Exception:
                gpu_pre_ms = None

        t_infer0 = time.perf_counter()
        if self.gpu_timing and self._gpu_infer_start is not None:
            self._gpu_infer_start.record(self.stream)
        if self.is_trt10:
            self.context.set_input_shape(self.input_name, infer_shape)
            for idx, name in enumerate(self.output_names):
                self.context.set_tensor_address(name, self.outputs[idx].devptr)
            self.context.set_tensor_address(self.input_name, self.input.devptr)
            self.context.execute_async_v3(stream_handle=self.stream.ptr)
            for out in self.outputs:
                out.copy_dtoh_async(self.stream)
            if self.gpu_timing and self._gpu_infer_end is not None:
                self._gpu_infer_end.record(self.stream)
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
            if self.gpu_timing and self._gpu_infer_end is not None:
                self._gpu_infer_end.record(self.stream)
            self.stream.synchronize()
            output_shapes = [tuple(self.context.get_binding_shape(binding)) for binding in self.output_indices]
            outputs = []
            for idx, out in enumerate(self.outputs):
                shape = output_shapes[idx]
                size = int(np.prod(shape))
                outputs.append(out.host[:size].reshape(shape))
        t_infer1 = time.perf_counter()
        gpu_infer_ms = None
        if self.gpu_timing and self._gpu_infer_start is not None and self._gpu_infer_end is not None:
            try:
                gpu_infer_ms = float(cp.cuda.get_elapsed_time(self._gpu_infer_start, self._gpu_infer_end))
            except Exception:
                gpu_infer_ms = None

        output = outputs[0] if outputs else np.array([])
        self._maybe_log_raw_output(output)
        t_decode0 = time.perf_counter()
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
                        use_numba_decode=self.use_numba_decode,
                        workspace=self._decode_workspace,
                        return_keypoints=self.return_keypoints,
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
                    use_numba_decode=self.use_numba_decode,
                    workspace=self._decode_workspace,
                    return_keypoints=self.return_keypoints,
                )
            )
        t_decode1 = time.perf_counter()
        self.last_batch_timings = {
            "pre_ms": (t_pre1 - t_pre0) * 1000.0,
            "infer_ms": (t_infer1 - t_infer0) * 1000.0,
            "decode_ms": (t_decode1 - t_decode0) * 1000.0,
            "gpu_pre_ms": gpu_pre_ms,
            "gpu_infer_ms": gpu_infer_ms,
            "batch": len(frames),
        }
        return batch_dets

    def _ensure_host_batch(self, batch_size: int, height: int, width: int) -> None:
        """Allocate or reuse pinned host batch buffers for CPU preprocessing."""
        if self._host_batch is not None and self._host_batch_shape:
            if (
                self._host_batch_shape[0] >= batch_size
                and self._host_batch_shape[1] == height
                and self._host_batch_shape[2] == width
            ):
                return
        try:
            import cupyx
        except ImportError as exc:
            raise RuntimeError("cupy is required for pinned host batch buffers") from exc
        shape = (self.max_batch_size, height, width, 3)
        self._host_batch = cupyx.zeros_pinned(shape, dtype=np.uint8)
        self._host_batch_shape = shape

    def _prepare_batch_cpu(self, frames) -> tuple[List[PreprocessMeta], tuple[int, int, int, int]]:
        """Prepare CPU batch (letterbox or resize) and upload into TRT input buffer."""
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("CuPy is required for TensorRT preprocessing") from exc
        target_hw = self.input_size
        self._ensure_host_batch(len(frames), target_hw[0], target_hw[1])
        if self.no_letterbox:
            batch_imgs, metas = prepare_batch_no_letterbox(frames, target_hw, out=self._host_batch)
        else:
            batch_imgs, metas = prepare_batch(frames, target_hw, out=self._host_batch)
        with self.stream:
            g_img = cp.asarray(batch_imgs)
            g_img = g_img[..., ::-1]
            g_img = cp.transpose(g_img, (0, 3, 1, 2))
            input_view = self.input.device[: g_img.size].reshape(g_img.shape)
            if self.input_dtype != input_view.dtype:
                input_view = input_view.astype(self.input_dtype, copy=False)
            if input_view.dtype == np.uint8:
                input_view[...] = g_img
            else:
                cp.multiply(g_img, (1.0 / 255.0), out=input_view)
        return metas, tuple(g_img.shape)

    def _ensure_gpu_buffers(self, batch_size: int) -> None:
        if not self._use_gpu_preproc:
            return
        try:
            import cv2
        except Exception:
            return
        while len(self._gpu_upload) < batch_size:
            self._gpu_upload.append(cv2.cuda_GpuMat())
            self._gpu_letterbox.append(None)

    def _prepare_batch_gpu(self, frames) -> tuple[List[PreprocessMeta], tuple[int, int, int, int]]:
        """Prepare GPU batch using OpenCV CUDA, then normalize into TRT input buffer."""
        try:
            import cv2
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("OpenCV CUDA and CuPy are required for TRT GPU preprocessing") from exc

        batch = len(frames)
        self._ensure_gpu_buffers(batch)
        target_hw = self.input_size
        height, width = target_hw
        input_view = self.input.device[: batch * 3 * height * width].reshape((batch, 3, height, width))
        scale = 1.0 / 255.0
        metas: List[PreprocessMeta] = []

        if self.gpu_timing and self._gpu_pre_start is not None:
            self._gpu_pre_start.record(self.stream)

        for idx, frame in enumerate(frames):
            if frame is None:
                raise ValueError("Frame is empty")
            shape = frame.shape[:2]

            gpu_src = self._gpu_upload[idx]
            if self._cv_stream is not None:
                gpu_src.upload(frame, self._cv_stream)
            else:
                gpu_src.upload(frame)

            if self.no_letterbox:
                scale_x = width / max(1, shape[1])
                scale_y = height / max(1, shape[0])
                if shape[0] != height or shape[1] != width:
                    interp = cv2.INTER_AREA if scale_x < 1.0 or scale_y < 1.0 else cv2.INTER_LINEAR
                    if self._cv_stream is not None:
                        gpu_resized = cv2.cuda.resize(
                            gpu_src, (width, height), interpolation=interp, stream=self._cv_stream
                        )
                    else:
                        gpu_resized = cv2.cuda.resize(gpu_src, (width, height), interpolation=interp)
                else:
                    gpu_resized = gpu_src
                gpu_letterbox = gpu_resized
                pad_x = 0.0
                pad_y = 0.0
            else:
                ratio = min(height / shape[0], width / shape[1])
                new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
                dw = width - new_unpad[0]
                dh = height - new_unpad[1]
                pad_x = dw / 2.0
                pad_y = dh / 2.0

                if shape[::-1] != new_unpad:
                    interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
                    if self._cv_stream is not None:
                        gpu_resized = cv2.cuda.resize(
                            gpu_src, new_unpad, interpolation=interp, stream=self._cv_stream
                        )
                    else:
                        gpu_resized = cv2.cuda.resize(gpu_src, new_unpad, interpolation=interp)
                else:
                    gpu_resized = gpu_src

                top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
                left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
                if self._cv_stream is not None:
                    gpu_letterbox = cv2.cuda.copyMakeBorder(
                        gpu_resized,
                        top,
                        bottom,
                        left,
                        right,
                        cv2.BORDER_CONSTANT,
                        value=(114, 114, 114),
                        stream=self._cv_stream,
                    )
                else:
                    gpu_letterbox = cv2.cuda.copyMakeBorder(
                        gpu_resized,
                        top,
                        bottom,
                        left,
                        right,
                        cv2.BORDER_CONSTANT,
                        value=(114, 114, 114),
                    )
                scale_x = ratio
                scale_y = ratio

            self._gpu_letterbox[idx] = gpu_letterbox
            metas.append(
                PreprocessMeta(
                    scale_x=scale_x,
                    scale_y=scale_y,
                    pad_x=pad_x,
                    pad_y=pad_y,
                    orig_shape=shape,
                )
            )

        if self._cv_stream is None and self._preproc_event is not None:
            self._preproc_event.record(cp.cuda.Stream.null)
            self.stream.wait_event(self._preproc_event)

        with self.stream:
            for idx in range(batch):
                img = self._gpumat_to_cupy(self._gpu_letterbox[idx])
                if img.ndim == 3 and img.shape[2] == 3:
                    img = img[..., ::-1]
                chw = cp.transpose(img, (2, 0, 1))
                if input_view.dtype == np.uint8:
                    input_view[idx] = chw
                else:
                    cp.multiply(chw, scale, out=input_view[idx])

        if self.gpu_timing and self._gpu_pre_end is not None and self._gpu_pre_start is not None:
            self._gpu_pre_end.record(self.stream)
            self._gpu_pre_end.synchronize()

        return metas, input_view.shape

    def _gpumat_to_cupy(self, mat):
        import cupy as cp

        try:
            rows = int(mat.rows)
            cols = int(mat.cols)
            if rows <= 0 or cols <= 0:
                return cp.asarray([])
            channels = int(mat.channels())
            elem_size = int(mat.elemSize())
            elem_size1 = int(mat.elemSize1())
            depth = int(mat.depth())
        except Exception:
            return cp.asarray(mat.download())

        try:
            import cv2
        except Exception:
            return cp.asarray(mat.download())

        depth_map = {
            cv2.CV_8U: np.uint8,
            cv2.CV_16U: np.uint16,
            cv2.CV_32F: np.float32,
            cv2.CV_64F: np.float64,
        }
        dtype = depth_map.get(depth)
        if dtype is None:
            return cp.asarray(mat.download())

        try:
            size_bytes = int(mat.step) * rows
            mem = cp.cuda.UnownedMemory(int(mat.cudaPtr()), size_bytes, mat)
            memptr = cp.cuda.MemoryPointer(mem, 0)
            if channels > 1:
                shape = (rows, cols, channels)
                strides = (int(mat.step), elem_size, elem_size1)
            else:
                shape = (rows, cols)
                strides = (int(mat.step), elem_size1)
            return cp.ndarray(shape, dtype=dtype, memptr=memptr, strides=strides)
        except Exception:
            return cp.asarray(mat.download())

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
