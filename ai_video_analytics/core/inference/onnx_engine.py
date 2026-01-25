from typing import List

import time

import numpy as np

from .base import Detection, InferenceEngine
from .yolo import decode_yolo_output, prepare_batch, prepare_input, PreprocessMeta
from ..utils.logging import get_logger

try:
    import cupy as cp
except Exception:
    cp = None

_HAS_CUPY = cp is not None
try:
    import cupyx.scipy.ndimage as _cupyx_ndimage
except Exception:
    _cupyx_ndimage = None

_HAS_CUPYX = _cupyx_ndimage is not None


class OnnxYoloEngine(InferenceEngine):
    """ONNX Runtime-based YOLO engine with optional GPU preprocessing."""
    def __init__(
        self,
        model_path: str,
        labels: List[str],
        input_size,
        confidence_threshold,
        nms_iou_threshold,
        has_objectness: bool,
        end_to_end: bool,
        log_severity_level: int,
        device: str,
        fp16: bool,
        int8: bool,
        class_id_filter: List[int] | None,
        debug_log_raw_output: bool,
        debug_log_raw_interval_seconds: float,
        debug_log_raw_rows: int,
        debug_log_raw_cols: int,
        use_cupy_nms: bool,
        use_numba_decode: bool,
        use_gpu_preproc: bool,
        return_keypoints: bool = False,
    ):
        self.model_path = model_path
        self.labels = labels
        self.input_size = tuple(input_size)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.has_objectness = has_objectness
        self.end_to_end = end_to_end
        self.log_severity_level = log_severity_level
        self.device = device
        self.fp16 = fp16
        self.int8 = int8
        self.class_id_filter = class_id_filter
        self.debug_log_raw_output = debug_log_raw_output
        self.debug_log_raw_interval_seconds = debug_log_raw_interval_seconds
        self.debug_log_raw_rows = debug_log_raw_rows
        self.debug_log_raw_cols = debug_log_raw_cols
        self.return_keypoints = bool(return_keypoints)
        self.session = None
        self.input_name = None
        self.input_dtype = np.float32
        self._last_raw_log_ts = 0.0
        self.logger = get_logger("inference.onnx")
        self.output_layout = ""
        self.use_cupy_nms = bool(use_cupy_nms)
        self.use_numba_decode = bool(use_numba_decode)
        self._use_gpu_preproc_flag = bool(use_gpu_preproc)
        self._device_id = 0
        self._has_cuda_provider = False
        self._use_io_binding = False
        self._use_gpu_preproc = False
        self._output_names: List[str] = []
        self._output_dtypes: List[np.dtype] = []

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNX inference") from exc

        providers = self._resolve_providers(self.device, ort.get_available_providers())
        self._device_id = self._parse_device_id(self.device) or 0
        self._has_cuda_provider = "CUDAExecutionProvider" in providers
        session_options = ort.SessionOptions()
        session_options.log_severity_level = self.log_severity_level
        provider_specs = self._resolve_provider_specs(providers, self._device_id)
        self.session = ort.InferenceSession(self.model_path, sess_options=session_options, providers=provider_specs)
        self.use_cupy_nms = self._has_cuda_provider and _HAS_CUPY and self.use_cupy_nms
        self._use_io_binding = self._has_cuda_provider and _HAS_CUPY
        self._use_gpu_preproc = (
            self._has_cuda_provider and _HAS_CUPY and _HAS_CUPYX and self._use_gpu_preproc_flag
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_dtype = self._resolve_input_dtype(self.session.get_inputs()[0].type)
        self._output_names = [output.name for output in self.session.get_outputs()]
        self._output_dtypes = [self._resolve_input_dtype(output.type) for output in self.session.get_outputs()]
        self.output_layout = self._resolve_output_layout()
        if self.fp16 and self.input_dtype != np.float16:
            self.logger.warning("FP16 requested but model input is %s", self.input_dtype)
        if self.input_dtype not in (np.float16, np.float32):
            self.logger.warning("Model input dtype %s may require custom preprocessing", self.input_dtype)
        if self.int8:
            self.logger.warning("INT8 requested; ensure ONNX model is INT8 calibrated")
        self.logger.info("Loaded ONNX model")

    def infer(self, frame) -> List[Detection]:
        return self.infer_batch([frame])[0]

    def infer_batch(self, frames) -> List[List[Detection]]:
        if self.session is None:
            raise RuntimeError("ONNX engine is not loaded")

        if not frames:
            return []

        if self._has_cuda_provider:
            output, metas = self._infer_cuda(frames)
        else:
            output, metas = self._infer_cpu(frames)

        self._maybe_log_raw_output(output)
        return self._decode_batch(output, metas)

    def _resolve_input_dtype(self, type_name: str):
        if not type_name:
            return np.float32
        if "float16" in type_name:
            return np.float16
        if "float" in type_name:
            return np.float32
        if "uint8" in type_name:
            return np.uint8
        if "int8" in type_name:
            return np.int8
        return np.float32

    def _resolve_output_layout(self) -> str:
        try:
            from ..utils.model_registry import get_registry
        except Exception:
            return ""
        registry = get_registry()
        model_name = registry.resolve_model_name(self.model_path)
        return registry.output_layout(model_name) or ""

    def _parse_device_id(self, device: str) -> int | None:
        if not device:
            return None
        value = device.lower()
        if ":" in value:
            try:
                return int(value.split(":")[-1])
            except ValueError:
                return None
        if value.startswith("cuda") or value.startswith("gpu"):
            return 0
        return None

    def _resolve_providers(self, device: str, available_providers):
        if not device:
            device = "auto"
        if not device:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        value = device.lower()
        if value.startswith("auto"):
            if "CUDAExecutionProvider" in available_providers:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]
        if value.startswith("cpu"):
            return ["CPUExecutionProvider"]
        if value.startswith("cuda") or value.startswith("gpu"):
            if "CUDAExecutionProvider" in available_providers:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.logger.warning("CUDAExecutionProvider unavailable; falling back to CPUExecutionProvider")
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _resolve_provider_specs(self, providers, device_id: int):
        if "CUDAExecutionProvider" not in providers:
            return providers
        specs = []
        for provider in providers:
            if provider == "CUDAExecutionProvider":
                specs.append(("CUDAExecutionProvider", {"device_id": device_id}))
            else:
                specs.append(provider)
        return specs

    def _infer_cpu(self, frames):
        inputs = []
        metas = []
        for frame in frames:
            input_tensor, meta = prepare_input(frame, self.input_size)
            inputs.append(input_tensor)
            metas.append(meta)
        input_tensor = np.concatenate(inputs, axis=0)
        input_tensor = input_tensor.astype(self.input_dtype, copy=False)
        outputs = self.session.run(self._output_names or None, {self.input_name: input_tensor})
        output = outputs[0]
        if isinstance(output, list):
            output = np.array(output)
        return output, metas

    def _infer_cuda(self, frames):
        if not self._use_io_binding:
            return self._infer_cpu(frames)

        if self._use_gpu_preproc:
            gpu_batch, metas = self._prepare_batch_gpu(frames)
        else:
            gpu_batch, metas = self._prepare_batch_cpu_to_gpu(frames)

        gpu_batch = cp.ascontiguousarray(gpu_batch)
        io_binding = self.session.io_binding()
        io_binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=self._device_id,
            element_type=self.input_dtype,
            shape=gpu_batch.shape,
            buffer_ptr=gpu_batch.data.ptr,
        )
        for output_name in self._output_names:
            io_binding.bind_output(output_name, device_type="cuda", device_id=self._device_id)
        self.session.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()
        output = outputs[0]
        if isinstance(output, list):
            output = np.array(output)
        return output, metas

    def _prepare_batch_cpu_to_gpu(self, frames):
        batch, metas = prepare_batch(frames, self.input_size)
        if self.input_dtype == np.uint8:
            cpu_batch = batch[..., ::-1]
            cpu_batch = cpu_batch.transpose(0, 3, 1, 2)
            cpu_batch = cpu_batch.astype(np.uint8, copy=False)
        else:
            cpu_batch = batch[..., ::-1].astype(np.float32, copy=False)
            cpu_batch = cpu_batch.transpose(0, 3, 1, 2)
            cpu_batch = cpu_batch / 255.0
            if self.input_dtype == np.float16:
                cpu_batch = cpu_batch.astype(np.float16, copy=False)
        gpu_batch = cp.asarray(cpu_batch)
        return gpu_batch, metas

    def _prepare_batch_gpu(self, frames):
        gpu_images = []
        metas = []
        for frame in frames:
            gpu_img, ratio, (dw, dh) = self._gpu_letterbox(frame, self.input_size)
            gpu_images.append(gpu_img)
            metas.append(PreprocessMeta(scale_x=ratio, scale_y=ratio, pad_x=dw, pad_y=dh, orig_shape=frame.shape[:2]))

        gpu_batch = cp.stack(gpu_images, axis=0)
        gpu_batch = gpu_batch[..., ::-1]
        gpu_batch = gpu_batch.transpose(0, 3, 1, 2)
        if self.input_dtype == np.uint8:
            gpu_batch = gpu_batch.astype(cp.uint8, copy=False)
        else:
            gpu_batch = gpu_batch.astype(cp.float32, copy=False)
            gpu_batch = gpu_batch / 255.0
            if self.input_dtype == np.float16:
                gpu_batch = gpu_batch.astype(cp.float16)
        return gpu_batch, metas

    def _gpu_letterbox(self, frame, input_size):
        import math

        if not _HAS_CUPY:
            raise RuntimeError("CuPy unavailable")

        shape = frame.shape[:2]
        ratio = min(input_size[0] / shape[0], input_size[1] / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        dw = input_size[1] - new_unpad[0]
        dh = input_size[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        gpu_img = cp.asarray(frame)
        if shape[::-1] != new_unpad:
            gpu_img = self._gpu_resize(gpu_img, (new_unpad[1], new_unpad[0]))

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        if top < 0 or bottom < 0 or left < 0 or right < 0:
            top = max(0, top)
            bottom = max(0, bottom)
            left = max(0, left)
            right = max(0, right)

        gpu_img = cp.pad(
            gpu_img,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=114,
        )

        target_h, target_w = input_size
        if gpu_img.shape[0] != target_h or gpu_img.shape[1] != target_w:
            gpu_img = gpu_img[:target_h, :target_w, :]
            if gpu_img.shape[0] < target_h or gpu_img.shape[1] < target_w:
                pad_h = target_h - gpu_img.shape[0]
                pad_w = target_w - gpu_img.shape[1]
                gpu_img = cp.pad(
                    gpu_img,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=114,
                )

        return gpu_img, ratio, (dw, dh)

    def _gpu_resize(self, gpu_img, new_shape):
        in_h, in_w = gpu_img.shape[:2]
        out_h, out_w = new_shape
        zoom_y = out_h / in_h
        zoom_x = out_w / in_w
        resized = _cupyx_ndimage.zoom(gpu_img, (zoom_y, zoom_x, 1), order=1)
        if resized.shape[0] != out_h or resized.shape[1] != out_w:
            resized = resized[:out_h, :out_w, :]
            if resized.shape[0] < out_h or resized.shape[1] < out_w:
                pad_h = out_h - resized.shape[0]
                pad_w = out_w - resized.shape[1]
                resized = cp.pad(
                    resized,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=114,
                )
        return resized

    def _decode_batch(self, output: np.ndarray, metas):
        detections_batch: List[List[Detection]] = []
        if output.ndim >= 3 and output.shape[0] == len(metas):
            for idx, meta in enumerate(metas):
                detections_batch.append(
                    decode_yolo_output(
                        output[idx : idx + 1],
                        labels=self.labels,
                        confidence_threshold=self.confidence_threshold,
                        nms_iou_threshold=self.nms_iou_threshold,
                        has_objectness=self.has_objectness,
                        meta=meta,
                        end_to_end=self.end_to_end,
                        output_layout=self.output_layout,
                        class_id_filter=self.class_id_filter,
                        use_cupy_nms=self.use_cupy_nms,
                        use_numba_decode=self.use_numba_decode,
                        return_keypoints=self.return_keypoints,
                    )
                )
            return detections_batch

        for meta in metas:
            detections_batch.append(
                decode_yolo_output(
                    output,
                    labels=self.labels,
                    confidence_threshold=self.confidence_threshold,
                    nms_iou_threshold=self.nms_iou_threshold,
                has_objectness=self.has_objectness,
                meta=meta,
                end_to_end=self.end_to_end,
                output_layout=self.output_layout,
                class_id_filter=self.class_id_filter,
                use_cupy_nms=self.use_cupy_nms,
                use_numba_decode=self.use_numba_decode,
                return_keypoints=self.return_keypoints,
            )
        )
        return detections_batch

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
