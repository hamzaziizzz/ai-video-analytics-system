import threading
import time
from typing import Optional

import queue

from ..utils.logging import get_logger
from ..utils.queue import put_drop_oldest


class CameraState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.connected = False
        self.last_frame_ts = 0.0
        self.last_error: Optional[str] = None
        self.fps = 0.0

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "connected": self.connected,
                "last_frame_ts": self.last_frame_ts,
                "last_error": self.last_error,
                "fps": self.fps,
            }


class RTSPCameraWorker:
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        state: CameraState,
        fps_limit: float,
        min_backoff: float,
        max_backoff: float,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.state = state
        self.fps_limit = fps_limit
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff
        self.logger = get_logger(f"ingestion.{camera_id}")

    def run(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for RTSP ingestion") from exc

        backoff = self.min_backoff
        last_frame_time = 0.0
        fps_window_start = time.time()
        frame_counter = 0

        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                self.state.update(connected=False, last_error="open_failed")
                self.logger.warning("Failed to open RTSP stream")
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
                continue

            self.state.update(connected=True, last_error=None)
            backoff = self.min_backoff

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                now = time.time()
                if not ok:
                    self.state.update(connected=False, last_error="read_failed")
                    self.logger.warning("RTSP read failed")
                    break

                if self.fps_limit > 0:
                    min_interval = 1.0 / self.fps_limit
                    if now - last_frame_time < min_interval:
                        continue

                last_frame_time = now
                frame_counter += 1
                if now - fps_window_start >= 1.0:
                    fps = frame_counter / (now - fps_window_start)
                    fps_window_start = now
                    frame_counter = 0
                    self.state.update(fps=fps)

                self.state.update(connected=True, last_frame_ts=now)
                put_drop_oldest(self.frame_queue, (now, frame))

            cap.release()
            time.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff)
