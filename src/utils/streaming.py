import threading
from typing import Dict, Optional, Tuple


class FrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: Dict[str, Tuple[bytes, int]] = {}
        self._last_update: Dict[str, int] = {}

    def update(self, camera_id: str, jpeg_bytes: bytes, timestamp_ms: int, fps_limit: float) -> None:
        if fps_limit > 0:
            last_ts = self._last_update.get(camera_id)
            min_interval = int(1000 / fps_limit)
            if last_ts is not None and (timestamp_ms - last_ts) < min_interval:
                return

        with self._lock:
            self._frames[camera_id] = (jpeg_bytes, timestamp_ms)
            self._last_update[camera_id] = timestamp_ms

    def get(self, camera_id: str) -> Optional[Tuple[bytes, int]]:
        with self._lock:
            return self._frames.get(camera_id)


def encode_jpeg(frame, quality: int) -> bytes:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for JPEG encoding") from exc

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode JPEG frame")
    return buffer.tobytes()
