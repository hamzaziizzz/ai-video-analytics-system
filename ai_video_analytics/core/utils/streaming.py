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


_nvjpeg = None
_turbojpeg = None


def encode_jpeg(frame, quality: int, use_nvjpeg: bool = False) -> bytes:
    global _nvjpeg, _turbojpeg

    if use_nvjpeg:
        if _nvjpeg is None:
            try:
                from nvjpeg import NvJpeg

                _nvjpeg = NvJpeg()
            except Exception:
                _nvjpeg = False
        if _nvjpeg:
            try:
                if hasattr(_nvjpeg, "encode"):
                    return _nvjpeg.encode(frame, quality=int(quality))
            except Exception:
                pass

    if _turbojpeg is None:
        try:
            from turbojpeg import TurboJPEG

            _turbojpeg = TurboJPEG()
        except Exception:
            _turbojpeg = False

    if _turbojpeg:
        try:
            return _turbojpeg.encode(frame, quality=int(quality))
        except Exception:
            pass

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for JPEG encoding") from exc

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode JPEG frame")
    return buffer.tobytes()
