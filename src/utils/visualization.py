from typing import Iterable, Optional, Tuple

from ..inference.base import Detection


def draw_detections(frame, detections: Iterable[Detection], zone: Optional[Iterable[Tuple[float, float]]] = None) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for drawing") from exc

    if zone:
        pts = [(int(x), int(y)) for x, y in zone]
        if len(pts) >= 3:
            import numpy as np

            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 255), thickness=2)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.score:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
