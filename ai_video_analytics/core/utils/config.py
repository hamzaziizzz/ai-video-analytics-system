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
    algorithm: str = "YOLO"
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.4
    nms_iou_threshold: float = 0.5
    device: str = "cuda:0"
    has_objectness: bool = False
    end_to_end: bool = False
    onnx_log_severity_level: int = 2
    workers: int = 0
    queue_size: int = 0
    request_timeout_seconds: float = 1.0
    batch_size: int = 1
    max_batch_wait_ms: float = 0.0
    class_id_filter: Optional[List[int]] = None
    debug_log_detections: bool = False
    debug_log_interval_seconds: float = 2.0
    debug_log_max_detections: int = 5
    debug_log_raw_output: bool = False
    debug_log_raw_interval_seconds: float = 2.0
    debug_log_raw_rows: int = 3
    debug_log_raw_cols: int = 6
    fp16: bool = False
    int8: bool = False
    int8_calib_data: Optional[str] = None
    int8_calib_images: int = 300
    int8_calib_cache: Optional[str] = None
    use_nvjpeg: bool = False
    auto_fps_enabled: bool = False
    target_fps_per_stream: float = 20.0
    max_fps_per_stream: float = 30.0
    min_fps_per_stream: float = 15.0
    fps_adjust_interval_seconds: float = 2.0
    fps_headroom: float = 0.9
    warmup_iters: int = 1
    warmup_batch_size: int = 1


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
class DatabaseConfig:
    enabled: bool = False
    host: str = "localhost"
    port: int = 5432
    name: str = "avas"
    user: str = "avas_user"
    password: str = "avas_password"
    sslmode: str = "prefer"
    connect_timeout_seconds: float = 5.0
    queue_size: int = 1000
    flush_interval_seconds: float = 0.5


@dataclass
class StreamingConfig:
    enabled: bool = False
    draw_boxes: bool = True
    draw_all_detections: bool = False
    jpeg_quality: int = 80
    fps_limit: float = 5.0


@dataclass
class FeatureConfig:
    tracking: bool = False
    fall_detection: bool = False
    counting: bool = True


@dataclass
class ModelConfig:
    algorithm: str = "YOLO"
    detection_model: Optional[str] = None
    classification_model: Optional[str] = None
    segmentation_model: Optional[str] = None
    obb_model: Optional[str] = None


@dataclass
class AppConfig:
    app: AppSettings
    cameras: List[CameraConfig]
    inference: InferenceConfig
    people_count: PeopleCountConfig
    alerts: AlertConfig
    database: DatabaseConfig
    streaming: StreamingConfig
    features: FeatureConfig
    models: ModelConfig


def validate_config(config: AppConfig) -> None:
    errors: List[str] = []

    camera_ids = [camera.camera_id for camera in config.cameras]
    duplicates = {cid for cid in camera_ids if camera_ids.count(cid) > 1}
    if duplicates:
        errors.append(f"Duplicate camera_id values: {', '.join(sorted(duplicates))}")

    enabled_cameras = [camera for camera in config.cameras if camera.enabled]
    if not enabled_cameras:
        errors.append("At least one camera must be enabled")

    zone_name = config.people_count.zone_name
    if zone_name and zone_name not in config.people_count.zones:
        errors.append(f"zone_name '{zone_name}' not found in people_count.zones")

    engine_name = config.inference.engine.lower()
    needs_labels = engine_name not in {"ultralytics", "pt", "pytorch"}
    if (
        needs_labels
        and not config.inference.labels
        and config.people_count.class_id is None
        and not config.inference.class_id_filter
    ):
        errors.append("inference.labels is required when class_id is not set for non-Ultralytics engines")

    if not config.models.algorithm:
        errors.append("models.algorithm must be set (e.g. YOLO)")

    if config.database.enabled:
        if not config.database.host or not config.database.name:
            errors.append("database.host and database.name are required when database.enabled is true")
        if not config.database.user or not config.database.password:
            errors.append("database.user and database.password are required when database.enabled is true")

    if errors:
        raise ValueError("Config validation failed: " + "; ".join(errors))


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
        algorithm=inference_raw.get("algorithm", "YOLO"),
        input_size=tuple(inference_raw.get("input_size", [640, 640])),
        confidence_threshold=float(inference_raw.get("confidence_threshold", 0.4)),
        nms_iou_threshold=float(inference_raw.get("nms_iou_threshold", 0.5)),
        device=inference_raw.get("device", "cuda:0"),
        has_objectness=bool(inference_raw.get("has_objectness", False)),
        end_to_end=bool(inference_raw.get("end_to_end", False)),
        onnx_log_severity_level=int(inference_raw.get("onnx_log_severity_level", 2)),
        workers=int(inference_raw.get("workers", 0)),
        queue_size=int(inference_raw.get("queue_size", 0)),
        request_timeout_seconds=float(inference_raw.get("request_timeout_seconds", 1.0)),
        batch_size=int(inference_raw.get("batch_size", 1)),
        max_batch_wait_ms=float(inference_raw.get("max_batch_wait_ms", 0.0)),
        class_id_filter=[int(v) for v in inference_raw.get("class_id_filter", [])] or None,
        debug_log_detections=bool(inference_raw.get("debug_log_detections", False)),
        debug_log_interval_seconds=float(inference_raw.get("debug_log_interval_seconds", 2.0)),
        debug_log_max_detections=int(inference_raw.get("debug_log_max_detections", 5)),
        debug_log_raw_output=bool(inference_raw.get("debug_log_raw_output", False)),
        debug_log_raw_interval_seconds=float(inference_raw.get("debug_log_raw_interval_seconds", 2.0)),
        debug_log_raw_rows=int(inference_raw.get("debug_log_raw_rows", 3)),
        debug_log_raw_cols=int(inference_raw.get("debug_log_raw_cols", 6)),
        fp16=bool(inference_raw.get("fp16", False)),
        int8=bool(inference_raw.get("int8", False)),
        int8_calib_data=inference_raw.get("int8_calib_data"),
        int8_calib_images=int(inference_raw.get("int8_calib_images", 300)),
        int8_calib_cache=inference_raw.get("int8_calib_cache"),
        use_nvjpeg=bool(inference_raw.get("use_nvjpeg", False)),
        auto_fps_enabled=bool(inference_raw.get("auto_fps_enabled", False)),
        target_fps_per_stream=float(inference_raw.get("target_fps_per_stream", 20.0)),
        max_fps_per_stream=float(inference_raw.get("max_fps_per_stream", 30.0)),
        min_fps_per_stream=float(inference_raw.get("min_fps_per_stream", 15.0)),
        fps_adjust_interval_seconds=float(inference_raw.get("fps_adjust_interval_seconds", 2.0)),
        fps_headroom=float(inference_raw.get("fps_headroom", 0.9)),
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

    database_raw = data.get("database", {})
    database = DatabaseConfig(
        enabled=bool(database_raw.get("enabled", False)),
        host=database_raw.get("host", DatabaseConfig.host),
        port=int(database_raw.get("port", DatabaseConfig.port)),
        name=database_raw.get("name", DatabaseConfig.name),
        user=database_raw.get("user", DatabaseConfig.user),
        password=database_raw.get("password", DatabaseConfig.password),
        sslmode=database_raw.get("sslmode", DatabaseConfig.sslmode),
        connect_timeout_seconds=float(database_raw.get("connect_timeout_seconds", DatabaseConfig.connect_timeout_seconds)),
        queue_size=int(database_raw.get("queue_size", DatabaseConfig.queue_size)),
        flush_interval_seconds=float(database_raw.get("flush_interval_seconds", DatabaseConfig.flush_interval_seconds)),
    )

    streaming_raw = data.get("streaming", {})
    streaming = StreamingConfig(
        enabled=bool(streaming_raw.get("enabled", False)),
        draw_boxes=bool(streaming_raw.get("draw_boxes", True)),
        draw_all_detections=bool(streaming_raw.get("draw_all_detections", False)),
        jpeg_quality=int(streaming_raw.get("jpeg_quality", 80)),
        fps_limit=float(streaming_raw.get("fps_limit", 5.0)),
    )

    features_raw = data.get("features", {})
    features = FeatureConfig(
        tracking=bool(features_raw.get("tracking", False)),
        fall_detection=bool(features_raw.get("fall_detection", False)),
        counting=bool(features_raw.get("counting", True)),
    )

    models_raw = data.get("models", {})
    models = ModelConfig(
        algorithm=models_raw.get("algorithm", "YOLO"),
        detection_model=models_raw.get("detection_model"),
        classification_model=models_raw.get("classification_model"),
        segmentation_model=models_raw.get("segmentation_model"),
        obb_model=models_raw.get("obb_model"),
    )

    return AppConfig(
        app=app,
        cameras=cameras,
        inference=inference,
        people_count=people_count,
        alerts=alerts,
        database=database,
        streaming=streaming,
        features=features,
        models=models,
    )
