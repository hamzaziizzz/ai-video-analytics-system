import queue
import threading
import time
from typing import Dict, List, Optional

from .alerts.webhook import WebhookAlertDispatcher
from .events.people_count import PeopleCountEngine
from .events.store import EventStore
from .inference.factory import create_engine
from .ingestion.rtsp import CameraState, RTSPCameraWorker
from .utils.config import AppConfig
from .utils.geometry import bbox_center, point_in_polygon
from .utils.logging import get_logger
from .utils.opencv import configure_opencv_logging
from .utils.streaming import FrameStore, encode_jpeg
from .utils.visualization import draw_detections


class CameraRuntime:
    def __init__(self, camera_id: str, state: CameraState, frame_queue: queue.Queue) -> None:
        self.camera_id = camera_id
        self.state = state
        self.frame_queue = frame_queue
        self.ingest_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None


class VideoAnalyticsService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.event_store = EventStore()
        self.logger = get_logger("service")
        self._stop_event = threading.Event()
        self._cameras: Dict[str, CameraRuntime] = {}
        self._frame_store = FrameStore() if config.streaming.enabled else None
        self._debug_last_log_ts: Dict[str, float] = {}
        self._alerts = WebhookAlertDispatcher(
            webhooks=config.alerts.webhooks,
            timeout_seconds=config.alerts.timeout_seconds,
        )

    def start(self) -> None:
        self.logger.info("Starting video analytics service")
        configure_opencv_logging(self.config.app.suppress_cv_warnings)
        self._alerts.start()
        for camera in self.config.cameras:
            if not camera.enabled:
                continue

            frame_queue: queue.Queue = queue.Queue(maxsize=camera.queue_size)
            state = CameraState()
            runtime = CameraRuntime(camera.camera_id, state, frame_queue)
            self._cameras[camera.camera_id] = runtime

            ingest_worker = RTSPCameraWorker(
                camera_id=camera.camera_id,
                rtsp_url=camera.rtsp_url,
                frame_queue=frame_queue,
                stop_event=self._stop_event,
                state=state,
                fps_limit=camera.fps_limit,
                min_backoff=camera.reconnect_min_backoff,
                max_backoff=camera.reconnect_max_backoff,
            )
            runtime.ingest_thread = threading.Thread(target=ingest_worker.run, daemon=True)
            runtime.ingest_thread.start()

            runtime.process_thread = threading.Thread(
                target=self._process_camera,
                args=(camera.camera_id, frame_queue),
                daemon=True,
            )
            runtime.process_thread.start()

    def stop(self) -> None:
        self.logger.info("Stopping video analytics service")
        self._stop_event.set()
        for runtime in self._cameras.values():
            if runtime.ingest_thread:
                runtime.ingest_thread.join(timeout=2)
            if runtime.process_thread:
                runtime.process_thread.join(timeout=2)
        self._alerts.stop()

    def _process_camera(self, camera_id: str, frame_queue: queue.Queue) -> None:
        logger = get_logger(f"processor.{camera_id}")
        engine = create_engine(self.config.inference)
        try:
            engine.load()
        except Exception as exc:
            logger.error("Failed to load inference engine: %s", exc)
            return

        zone_name = self.config.people_count.zone_name
        zone_polygon = None
        if zone_name:
            zone_polygon = self.config.people_count.zones.get(zone_name)
            if zone_polygon:
                zone_polygon = [(pt[0], pt[1]) for pt in zone_polygon]

        people_engine = PeopleCountEngine(
            camera_id=camera_id,
            min_count=self.config.people_count.min_count,
            stable_frames=self.config.people_count.stable_frames,
            cooldown_seconds=self.config.people_count.cooldown_seconds,
            report_interval_seconds=self.config.people_count.report_interval_seconds,
            zone=zone_name,
        )

        person_label = self.config.people_count.class_name
        person_class_id = self.config.people_count.class_id
        stream_cfg = self.config.streaming
        inference_cfg = self.config.inference

        try:
            while not self._stop_event.is_set():
                try:
                    _, frame = frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                timestamp_ms = int(time.time() * 1000)
                detections = engine.infer(frame)
                self._maybe_log_detections(logger, camera_id, detections, inference_cfg)

                filtered = []
                for det in detections:
                    if person_class_id is not None:
                        if det.class_id != person_class_id:
                            continue
                    elif det.class_name != person_label:
                        continue
                    if zone_polygon:
                        center = bbox_center(det.bbox)
                        if not point_in_polygon(center, zone_polygon):
                            continue
                    filtered.append(det)

                count = len(filtered)
                event = people_engine.update(count, timestamp_ms)
                if event:
                    self.event_store.add(event)
                    self._alerts.enqueue(event)

                if self._frame_store:
                    stream_detections = detections if stream_cfg.draw_all_detections else filtered
                    self._update_stream_frame(
                        camera_id=camera_id,
                        frame=frame,
                        detections=stream_detections,
                        zone_polygon=zone_polygon,
                        timestamp_ms=timestamp_ms,
                        stream_cfg=stream_cfg,
                    )
        finally:
            if hasattr(engine, "close"):
                engine.close()

    def get_events(self, limit: int = 100) -> List[dict]:
        return self.event_store.list(limit=limit)

    def get_camera_status(self) -> Dict[str, dict]:
        return {camera_id: runtime.state.snapshot() for camera_id, runtime in self._cameras.items()}

    def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        if not self._frame_store:
            return None
        entry = self._frame_store.get(camera_id)
        if not entry:
            return None
        return entry[0]

    def streaming_enabled(self) -> bool:
        return self._frame_store is not None

    def stream_frames(self, camera_id: str):
        if not self._frame_store:
            return

        last_ts = 0
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        try:
            while not self._stop_event.is_set():
                entry = self._frame_store.get(camera_id)
                if not entry:
                    time.sleep(0.1)
                    continue
                jpeg_bytes, timestamp_ms = entry
                if timestamp_ms == last_ts:
                    time.sleep(0.05)
                    continue
                last_ts = timestamp_ms
                yield boundary + jpeg_bytes + b"\r\n"
        except GeneratorExit:
            return

    def _update_stream_frame(self, camera_id: str, frame, detections, zone_polygon, timestamp_ms: int, stream_cfg) -> None:
        try:
            if stream_cfg.draw_boxes:
                draw_detections(frame, detections, zone_polygon)
            jpeg_bytes = encode_jpeg(frame, stream_cfg.jpeg_quality)
            self._frame_store.update(camera_id, jpeg_bytes, timestamp_ms, stream_cfg.fps_limit)
        except Exception as exc:
            self.logger.warning("Failed to update stream frame: %s", exc)

    def _maybe_log_detections(self, logger, camera_id: str, detections, inference_cfg) -> None:
        if not inference_cfg.debug_log_detections:
            return

        now = time.time()
        last_ts = self._debug_last_log_ts.get(camera_id, 0.0)
        if now - last_ts < inference_cfg.debug_log_interval_seconds:
            return
        self._debug_last_log_ts[camera_id] = now

        sample = []
        for det in detections[: inference_cfg.debug_log_max_detections]:
            sample.append(
                {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "score": round(det.score, 3),
                    "bbox": [round(v, 1) for v in det.bbox],
                }
            )
        logger.info("Detections: count=%d sample=%s", len(detections), sample)
