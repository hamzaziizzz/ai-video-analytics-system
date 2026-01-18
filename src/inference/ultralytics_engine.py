from typing import List

from .base import Detection, InferenceEngine
from ..utils.logging import get_logger


class UltralyticsYoloEngine(InferenceEngine):
    def __init__(
        self,
        model_path: str,
        labels: List[str],
        input_size,
        confidence_threshold,
        nms_iou_threshold,
        device: str,
    ) -> None:
        self.model_path = model_path
        self.labels = labels
        self.input_size = tuple(input_size)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.model = None
        self.logger = get_logger("inference.ultralytics")

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for PyTorch inference") from exc

        self.model = YOLO(self.model_path)
        if not self.labels:
            self.labels = self.model.names
        self.logger.info("Loaded Ultralytics model")

    def infer(self, frame) -> List[Detection]:
        if self.model is None:
            raise RuntimeError("Ultralytics engine is not loaded")

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        detections: List[Detection] = []
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            score = float(box.conf[0]) if box.conf is not None else 0.0
            class_id = int(box.cls[0]) if box.cls is not None else -1
            class_name = self._class_name(class_id)
            detections.append(
                Detection(
                    bbox=[float(x) for x in xyxy],
                    score=score,
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections

    def _class_name(self, class_id: int) -> str:
        if isinstance(self.labels, dict):
            return str(self.labels.get(class_id, class_id))
        if class_id < 0:
            return "-1"
        if class_id < len(self.labels):
            return self.labels[class_id]
        return str(class_id)
