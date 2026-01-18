import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class AppSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    suppress_cv_warnings: bool = False


@dataclass
class CameraConfig:
    camera_id: str
    rtsp_url: str
    name: str = ""
    enabled: bool = True
    fps_limit: float = 10.0
    queue_size: int = 30
    reconnect_min_backoff: float = 1.0
    reconnect_max_backoff: float = 30.0


@dataclass
class InferenceConfig:
    engine: str
    model_path: str
    labels: List[str]
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.4
    nms_iou_threshold: float = 0.5
    device: str = "cuda:0"
    has_objectness: bool = False
    end_to_end: bool = False
    onnx_log_severity_level: int = 2
    debug_log_detections: bool = False
    debug_log_interval_seconds: float = 2.0
    debug_log_max_detections: int = 5
    debug_log_raw_output: bool = False
    debug_log_raw_interval_seconds: float = 2.0
    debug_log_raw_rows: int = 3
    debug_log_raw_cols: int = 6


@dataclass
class PeopleCountConfig:
    min_count: int = 1
    stable_frames: int = 3
    cooldown_seconds: float = 10.0
    report_interval_seconds: float = 60.0
    class_name: str = "person"
    class_id: Optional[int] = None
    zones: Dict[str, List[List[float]]] = field(default_factory=dict)
    zone_name: Optional[str] = None


@dataclass
class AlertConfig:
    webhooks: List[str] = field(default_factory=list)
    timeout_seconds: float = 5.0


@dataclass
class StreamingConfig:
    enabled: bool = False
    draw_boxes: bool = True
    draw_all_detections: bool = False
    jpeg_quality: int = 80
    fps_limit: float = 5.0


@dataclass
class AppConfig:
    app: AppSettings
    cameras: List[CameraConfig]
    inference: InferenceConfig
    people_count: PeopleCountConfig
    alerts: AlertConfig
    streaming: StreamingConfig


def _load_json(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    data = _load_json(config_path)

    app_data = data.get("app", {})
    app = AppSettings(
        host=app_data.get("host", AppSettings.host),
        port=int(app_data.get("port", AppSettings.port)),
        log_level=app_data.get("log_level", AppSettings.log_level),
        suppress_cv_warnings=bool(app_data.get("suppress_cv_warnings", False)),
    )

    cameras_raw = data.get("cameras")
    if not cameras_raw:
        raise ValueError("Config must include at least one camera")
    cameras = [
        CameraConfig(
            camera_id=cam["camera_id"],
            rtsp_url=cam["rtsp_url"],
            name=cam.get("name", ""),
            enabled=cam.get("enabled", True),
            fps_limit=float(cam.get("fps_limit", 10.0)),
            queue_size=int(cam.get("queue_size", 30)),
            reconnect_min_backoff=float(cam.get("reconnect_min_backoff", 1.0)),
            reconnect_max_backoff=float(cam.get("reconnect_max_backoff", 30.0)),
        )
        for cam in cameras_raw
    ]

    inference_raw = data.get("inference")
    if not inference_raw:
        raise ValueError("Config must include inference settings")
    inference = InferenceConfig(
        engine=inference_raw["engine"],
        model_path=inference_raw["model_path"],
        labels=inference_raw.get("labels", []),
        input_size=tuple(inference_raw.get("input_size", [640, 640])),
        confidence_threshold=float(inference_raw.get("confidence_threshold", 0.4)),
        nms_iou_threshold=float(inference_raw.get("nms_iou_threshold", 0.5)),
        device=inference_raw.get("device", "cuda:0"),
        has_objectness=bool(inference_raw.get("has_objectness", False)),
        end_to_end=bool(inference_raw.get("end_to_end", False)),
        onnx_log_severity_level=int(inference_raw.get("onnx_log_severity_level", 2)),
        debug_log_detections=bool(inference_raw.get("debug_log_detections", False)),
        debug_log_interval_seconds=float(inference_raw.get("debug_log_interval_seconds", 2.0)),
        debug_log_max_detections=int(inference_raw.get("debug_log_max_detections", 5)),
        debug_log_raw_output=bool(inference_raw.get("debug_log_raw_output", False)),
        debug_log_raw_interval_seconds=float(inference_raw.get("debug_log_raw_interval_seconds", 2.0)),
        debug_log_raw_rows=int(inference_raw.get("debug_log_raw_rows", 3)),
        debug_log_raw_cols=int(inference_raw.get("debug_log_raw_cols", 6)),
    )

    people_raw = data.get("people_count", {})
    people_count = PeopleCountConfig(
        min_count=int(people_raw.get("min_count", 1)),
        stable_frames=int(people_raw.get("stable_frames", 3)),
        cooldown_seconds=float(people_raw.get("cooldown_seconds", 10.0)),
        report_interval_seconds=float(people_raw.get("report_interval_seconds", 60.0)),
        class_name=people_raw.get("class_name", "person"),
        class_id=people_raw.get("class_id"),
        zones=people_raw.get("zones", {}),
        zone_name=people_raw.get("zone_name"),
    )

    alerts_raw = data.get("alerts", {})
    alerts = AlertConfig(
        webhooks=alerts_raw.get("webhooks", []),
        timeout_seconds=float(alerts_raw.get("timeout_seconds", 5.0)),
    )

    streaming_raw = data.get("streaming", {})
    streaming = StreamingConfig(
        enabled=bool(streaming_raw.get("enabled", False)),
        draw_boxes=bool(streaming_raw.get("draw_boxes", True)),
        draw_all_detections=bool(streaming_raw.get("draw_all_detections", False)),
        jpeg_quality=int(streaming_raw.get("jpeg_quality", 80)),
        fps_limit=float(streaming_raw.get("fps_limit", 5.0)),
    )

    return AppConfig(
        app=app,
        cameras=cameras,
        inference=inference,
        people_count=people_count,
        alerts=alerts,
        streaming=streaming,
    )
