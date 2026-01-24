from typing import List

import time

import numpy as np

from .base import Detection, InferenceEngine
from .yolo import decode_yolo_output, prepare_input
from ..utils.logging import get_logger


class OpenVINOYoloEngine(InferenceEngine):
    """OpenVINO-based YOLO engine for CPU/GPU inference."""
    def __init__(
        self,
        model_path: str,
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
        return_keypoints: bool = False,
    ) -> None:
        self.model_path = model_path
        self.labels = labels
        self.input_size = tuple(input_size)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.has_objectness = has_objectness
        self.end_to_end = end_to_end
        self.device = device or "CPU"
        self.fp16 = fp16
        self.int8 = int8
        self.class_id_filter = class_id_filter
        self.debug_log_raw_output = debug_log_raw_output
        self.debug_log_raw_interval_seconds = debug_log_raw_interval_seconds
        self.debug_log_raw_rows = debug_log_raw_rows
        self.debug_log_raw_cols = debug_log_raw_cols
        self.return_keypoints = bool(return_keypoints)
        self.logger = get_logger("inference.openvino")
        self.compiled_model = None
        self.output_layer = None
        self.input_layer = None
        self.input_dtype = np.float32
        self._last_raw_log_ts = 0.0
        self.output_layout = ""
        self.use_cupy_nms = False

    def load(self) -> None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("openvino is required for OpenVINO inference") from exc

        core = ov.Core()
        model = core.read_model(self.model_path)
        if model is None:
            raise RuntimeError("Failed to read OpenVINO model")

        input_layer = model.input(0)
        self.input_dtype = self._resolve_input_dtype(input_layer)
        if any(dim.is_dynamic for dim in input_layer.partial_shape):
            model.reshape({input_layer: [1, 3, self.input_size[0], self.input_size[1]]})

        device = self._normalize_device(self.device)
        compile_config = self._build_precision_config()
        try:
            self.compiled_model = core.compile_model(model, device, compile_config) if compile_config else core.compile_model(model, device)
        except Exception:
            if compile_config:
                self.logger.warning("Failed to apply precision hints, retrying without them")
                self.compiled_model = core.compile_model(model, device)
            else:
                raise
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.output_layout = self._resolve_output_layout()
        self.logger.info("Loaded OpenVINO model")

    def infer(self, frame) -> List[Detection]:
        if self.compiled_model is None:
            raise RuntimeError("OpenVINO engine is not loaded")

        input_tensor, meta = prepare_input(frame, self.input_size)
        input_tensor = input_tensor.astype(self.input_dtype, copy=False)
        results = self.compiled_model([input_tensor])

        if isinstance(results, dict):
            if self.output_layer in results:
                output = results[self.output_layer]
            else:
                output = next(iter(results.values()))
        else:
            output = results[0]

        if not isinstance(output, np.ndarray):
            output = np.array(output)

        self._maybe_log_raw_output(output)

        return decode_yolo_output(
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
            return_keypoints=self.return_keypoints,
        )

    def _normalize_device(self, device: str) -> str:
        device_lower = device.lower()
        if device_lower.startswith("cuda") or device_lower.startswith("gpu"):
            return "GPU"
        if device_lower.startswith("cpu"):
            return "CPU"
        if device_lower.startswith("auto"):
            return "AUTO"
        return device

    def _build_precision_config(self) -> dict:
        if self.int8:
            return {"INFERENCE_PRECISION_HINT": "i8"}
        if self.fp16:
            return {"INFERENCE_PRECISION_HINT": "f16"}
        return {}

    def _resolve_output_layout(self) -> str:
        try:
            from ..utils.model_registry import get_registry
        except Exception:
            return ""
        registry = get_registry()
        model_name = registry.resolve_model_name(self.model_path)
        return registry.output_layout(model_name) or ""

    def _resolve_input_dtype(self, input_layer) -> np.dtype:
        try:
            element_type = input_layer.get_element_type()
            if hasattr(element_type, "to_dtype"):
                return element_type.to_dtype()
        except Exception:
            pass
        return np.float32

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
