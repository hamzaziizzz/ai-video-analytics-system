from typing import List

import time

import numpy as np

from .base import Detection, InferenceEngine
from .yolo import decode_yolo_output, prepare_input
from ..utils.logging import get_logger


class OnnxYoloEngine(InferenceEngine):
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
        debug_log_raw_output: bool,
        debug_log_raw_interval_seconds: float,
        debug_log_raw_rows: int,
        debug_log_raw_cols: int,
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
        self.debug_log_raw_output = debug_log_raw_output
        self.debug_log_raw_interval_seconds = debug_log_raw_interval_seconds
        self.debug_log_raw_rows = debug_log_raw_rows
        self.debug_log_raw_cols = debug_log_raw_cols
        self.session = None
        self.input_name = None
        self.input_dtype = np.float32
        self._last_raw_log_ts = 0.0
        self.logger = get_logger("inference.onnx")

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNX inference") from exc

        providers = self._resolve_providers(self.device)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = self.log_severity_level
        self.session = ort.InferenceSession(self.model_path, sess_options=session_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_dtype = self._resolve_input_dtype(self.session.get_inputs()[0].type)
        self.logger.info("Loaded ONNX model")

    def infer(self, frame) -> List[Detection]:
        if self.session is None:
            raise RuntimeError("ONNX engine is not loaded")

        input_tensor, meta = prepare_input(frame, self.input_size)
        input_tensor = input_tensor.astype(self.input_dtype, copy=False)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        output = outputs[0]
        if isinstance(output, list):
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
        )

    def _resolve_input_dtype(self, type_name: str):
        if not type_name:
            return np.float32
        if "float16" in type_name:
            return np.float16
        if "float" in type_name:
            return np.float32
        return np.float32

    def _resolve_providers(self, device: str):
        if not device:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        value = device.lower()
        if value.startswith("cpu"):
            return ["CPUExecutionProvider"]
        if value.startswith("cuda") or value.startswith("gpu"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

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
