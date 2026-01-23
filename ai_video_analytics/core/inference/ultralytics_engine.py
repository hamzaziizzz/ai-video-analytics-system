from typing import List

from .base import Detection, InferenceEngine
from ..utils.device import resolve_torch_device
from ..utils.logging import get_logger


class UltralyticsYoloEngine(InferenceEngine):
    """YOLO inference engine backed by the Ultralytics runtime."""
    def __init__(
        self,
        model_path: str,
        labels: List[str],
        input_size,
        confidence_threshold,
        nms_iou_threshold,
        device: str,
        fp16: bool,
        int8: bool,
        class_id_filter: List[int] | None,
        batch_size: int,
    ) -> None:
        self.model_path = model_path
        self.labels = labels
        self.input_size = tuple(input_size)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.fp16 = fp16
        self.int8 = int8
        self.class_id_filter = class_id_filter
        self.batch_size = max(1, int(batch_size) if batch_size else 1)
        self.model = None
        self.logger = get_logger("inference.ultralytics")

    def load(self) -> None:
        try:
            from ultralytics import YOLO
            import torch
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for PyTorch inference") from exc

        self.model = YOLO(self.model_path)
        self.device = resolve_torch_device(self.device)
        if str(self.device).startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        if not self.labels:
            self.labels = self.model.names
        self.logger.info("Loaded Ultralytics model")

    def infer(self, frame) -> List[Detection]:
        if self.model is None:
            raise RuntimeError("Ultralytics engine is not loaded")

        return self._predict_detections([frame])[0]

    def infer_batch(self, frames) -> List[List[Detection]]:
        if self.model is None:
            raise RuntimeError("Ultralytics engine is not loaded")
        if not frames:
            return []
        return self._predict_detections(frames)

    def _predict_detections(self, frames) -> List[List[Detection]]:
        batch = len(frames)
        pad_to = max(self.batch_size, batch)
        if batch < pad_to:
            frames = list(frames) + [frames[-1]] * (pad_to - batch)

        results = self.model.predict(
            source=frames,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            half=self.fp16,
            batch=len(frames),
            classes=self.class_id_filter,
            verbose=False,
        )

        if not results:
            return [[] for _ in range(batch)]

        detections: List[List[Detection]] = []
        for result in results[:batch]:
            boxes = result.boxes
            if boxes is None:
                detections.append([])
                continue

            frame_detections: List[Detection] = []
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                score = float(box.conf[0]) if box.conf is not None else 0.0
                class_id = int(box.cls[0]) if box.cls is not None else -1
                class_name = self._class_name(class_id)
                frame_detections.append(
                    Detection(
                        bbox=[float(x) for x in xyxy],
                        score=score,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )
            detections.append(frame_detections)
        return detections

    def _class_name(self, class_id: int) -> str:
        if isinstance(self.labels, dict):
            return str(self.labels.get(class_id, class_id))
        if class_id < 0:
            return "-1"
        if class_id < len(self.labels):
            return self.labels[class_id]
        return str(class_id)
