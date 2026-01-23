from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from pydantic import BaseSettings
except ImportError:  # pragma: no cover - fastapi installs pydantic
    BaseSettings = object  # type: ignore

from .utils.config import AppConfig


class Settings(BaseSettings):
    app_host: Optional[str] = None
    app_port: Optional[int] = None
    app_log_level: Optional[str] = None
    inference_device: Optional[str] = None
    models_dir: str = "models"
    algorithm: Optional[str] = None
    detection_model: Optional[str] = None
    classification_model: Optional[str] = None
    segmentation_model: Optional[str] = None
    obb_model: Optional[str] = None
    tracking: Optional[bool] = None
    fall_detection: Optional[bool] = None
    count: Optional[bool] = None
    inference_backend: Optional[str] = None
    num_workers: Optional[int] = None
    gpu_id: Optional[int] = None
    batch_size: Optional[int] = None
    max_batch_wait_ms: Optional[float] = None
    fp16: Optional[bool] = None
    int8: Optional[bool] = None
    int8_calib_data: Optional[str] = None
    int8_calib_images: Optional[int] = None
    int8_calib_cache: Optional[str] = None
    use_nvjpeg: Optional[bool] = None
    camera_urls: Optional[str] = None
    camera_ids: Optional[str] = None
    camera_fps_limit: Optional[float] = None
    camera_queue_size: Optional[int] = None
    reconnect_min_backoff: Optional[float] = None
    reconnect_max_backoff: Optional[float] = None
    db_enabled: Optional[bool] = None
    db_host: Optional[str] = None
    db_port: Optional[int] = None
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_sslmode: Optional[str] = None
    db_connect_timeout_seconds: Optional[float] = None
    db_queue_size: Optional[int] = None
    db_flush_interval_seconds: Optional[float] = None
    detection_threshold: Optional[float] = None
    nms_iou_threshold: Optional[float] = None
    frame_width: Optional[int] = None
    auto_fps_enabled: Optional[bool] = None
    target_fps_per_stream: Optional[float] = None
    max_fps_per_stream: Optional[float] = None
    min_fps_per_stream: Optional[float] = None
    fps_adjust_interval_seconds: Optional[float] = None
    fps_headroom: Optional[float] = None
    warmup_iters: Optional[int] = None
    warmup_batch_size: Optional[int] = None

    class Config:
        env_prefix = "AVAS_"
        case_sensitive = False


def build_default_config(settings: Settings) -> AppConfig:
    from .utils.config import (
        AlertConfig,
        AppConfig,
        AppSettings,
        CameraConfig,
        DatabaseConfig,
        FeatureConfig,
        InferenceConfig,
        ModelConfig,
        PeopleCountConfig,
        StreamingConfig,
    )

    app = AppSettings()
    cameras = [
        CameraConfig(
            camera_id="cam_entrance",
            rtsp_url=0,
            name="Main Entrance",
            enabled=True,
            fps_limit=30.0,
            queue_size=30,
            reconnect_min_backoff=1.0,
            reconnect_max_backoff=30.0,
        )
    ]
    inference = InferenceConfig(
        engine="ultralytics",
        model_path="models/yolo26x.pt",
        labels=[],
        algorithm="YOLO",
        input_size=(640, 640),
        confidence_threshold=0.4,
        nms_iou_threshold=0.5,
        device="auto",
        has_objectness=False,
        end_to_end=False,
    )
    people_count = PeopleCountConfig(
        min_count=1,
        stable_frames=3,
        cooldown_seconds=10.0,
        report_interval_seconds=60.0,
        class_name="person",
        class_id=0,
        zones={"entrance_zone": [[0, 0], [640, 0], [640, 360], [0, 360]]},
        zone_name=None,
    )
    if people_count.class_id is not None:
        inference.class_id_filter = [people_count.class_id]
    alerts = AlertConfig()
    database = DatabaseConfig()
    streaming = StreamingConfig(enabled=True, draw_boxes=True, draw_all_detections=False, jpeg_quality=80, fps_limit=30.0)
    features = FeatureConfig(tracking=False, fall_detection=False, counting=True)
    models = ModelConfig(
        algorithm=settings.algorithm or "YOLO",
        detection_model=settings.detection_model or "yolo26x",
        classification_model=settings.classification_model,
        segmentation_model=settings.segmentation_model,
        obb_model=settings.obb_model,
    )

    config = AppConfig(
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
    config.inference.algorithm = models.algorithm
    apply_settings_overrides(config, settings)
    return config


def apply_settings_overrides(config: AppConfig, settings: Settings) -> None:
    app_host = _env_str(settings.app_host, "APP_HOST")
    if app_host:
        config.app.host = app_host
    app_port = _env_int(settings.app_port, "APP_PORT")
    if app_port is None:
        app_port = _env_int(None, "PORT")
    if app_port is not None:
        config.app.port = app_port
    app_log_level = _env_str(settings.app_log_level, "APP_LOG_LEVEL")
    if not app_log_level:
        app_log_level = _env_str(None, "LOG_LEVEL")
    if app_log_level:
        config.app.log_level = app_log_level

    inference_device = _env_str(settings.inference_device, "INFERENCE_DEVICE")
    if inference_device:
        config.inference.device = inference_device
    gpu_id = _env_int(settings.gpu_id, "GPU_ID")
    if gpu_id is not None:
        if not inference_device or inference_device.strip().lower() == "auto":
            config.inference.device = f"cuda:{gpu_id}"
    inference_backend = _env_str(settings.inference_backend, "INFERENCE_BACKEND")
    if inference_backend:
        config.inference.engine = _normalize_backend(inference_backend)
    num_workers = _env_int(settings.num_workers, "NUM_WORKERS")
    if num_workers is not None:
        config.inference.workers = num_workers
    batch_size = _env_int(settings.batch_size, "BATCH_SIZE")
    if batch_size is None:
        batch_size = _env_int(None, "DET_BATCH_SIZE")
    if batch_size is not None:
        config.inference.batch_size = batch_size
    max_batch_wait_ms = _env_float(settings.max_batch_wait_ms, "MAX_BATCH_WAIT_MS")
    if max_batch_wait_ms is not None:
        config.inference.max_batch_wait_ms = max_batch_wait_ms
    detection_threshold = _env_float(settings.detection_threshold, "DETECTION_THRESHOLD")
    if detection_threshold is None:
        detection_threshold = _env_float(None, "DEF_DET_THRESH")
    if detection_threshold is not None:
        config.inference.confidence_threshold = detection_threshold
    nms_iou_threshold = _env_float(settings.nms_iou_threshold, "NMS_IOU_THRESHOLD")
    if nms_iou_threshold is not None:
        config.inference.nms_iou_threshold = nms_iou_threshold
    frame_width = _env_int(settings.frame_width, "FRAME_WIDTH")
    if frame_width is not None and frame_width > 0:
        base_h, base_w = config.inference.input_size
        ratio = base_h / base_w if base_w else 1.0
        frame_height = max(1, int(round(frame_width * ratio)))
        config.inference.input_size = (frame_height, frame_width)
    max_size = _env_str(None, "MAX_SIZE")
    if max_size:
        size = _parse_size_list(max_size)
        if size:
            width, height = size
            config.inference.input_size = (height, width)
    auto_fps_enabled = _env_bool(settings.auto_fps_enabled, "AUTO_FPS")
    if auto_fps_enabled is not None:
        config.inference.auto_fps_enabled = auto_fps_enabled
    target_fps = _env_float(settings.target_fps_per_stream, "TARGET_FPS_PER_STREAM")
    if target_fps is not None:
        config.inference.target_fps_per_stream = target_fps
    max_fps = _env_float(settings.max_fps_per_stream, "MAX_FPS_PER_STREAM")
    if max_fps is not None:
        config.inference.max_fps_per_stream = max_fps
    min_fps = _env_float(settings.min_fps_per_stream, "MIN_FPS_PER_STREAM")
    if min_fps is not None:
        config.inference.min_fps_per_stream = min_fps
    fps_adjust_interval = _env_float(settings.fps_adjust_interval_seconds, "FPS_ADJUST_INTERVAL_SECONDS")
    if fps_adjust_interval is not None:
        config.inference.fps_adjust_interval_seconds = fps_adjust_interval
    fps_headroom = _env_float(settings.fps_headroom, "FPS_HEADROOM")
    if fps_headroom is not None:
        config.inference.fps_headroom = fps_headroom
    warmup_iters = _env_int(settings.warmup_iters, "WARMUP_ITERS")
    if warmup_iters is not None:
        config.inference.warmup_iters = max(0, warmup_iters)
    warmup_batch_size = _env_int(settings.warmup_batch_size, "WARMUP_BATCH_SIZE")
    if warmup_batch_size is not None:
        config.inference.warmup_batch_size = max(1, warmup_batch_size)
    fp16 = _env_bool(settings.fp16, "FP16")
    if fp16 is None:
        fp16 = _env_bool(None, "FORCE_FP16")
    if fp16 is not None:
        config.inference.fp16 = fp16
    int8 = _env_bool(settings.int8, "INT8")
    if int8 is not None:
        config.inference.int8 = int8
    int8_calib_data = _env_str(settings.int8_calib_data, "INT8_CALIB_DATA")
    if int8_calib_data:
        config.inference.int8_calib_data = int8_calib_data
    int8_calib_images = _env_int(settings.int8_calib_images, "INT8_CALIB_IMAGES")
    if int8_calib_images is not None:
        config.inference.int8_calib_images = int8_calib_images
    int8_calib_cache = _env_str(settings.int8_calib_cache, "INT8_CALIB_CACHE")
    if int8_calib_cache:
        config.inference.int8_calib_cache = int8_calib_cache
    use_nvjpeg = _env_bool(settings.use_nvjpeg, "USE_NVJPEG")
    if use_nvjpeg is not None:
        config.inference.use_nvjpeg = use_nvjpeg
    if config.inference.int8:
        config.inference.fp16 = False

    algorithm = _env_str(settings.algorithm, "ALGORITHM")
    if algorithm:
        config.models.algorithm = algorithm
        config.inference.algorithm = algorithm
    detection_model = _env_str(settings.detection_model, "DETECTION_MODEL")
    if not detection_model:
        detection_model = _env_str(None, "DET_NAME")
    if detection_model:
        config.models.detection_model = detection_model
    classification_model = _env_str(settings.classification_model, "CLASSIFICATION_MODEL")
    if classification_model:
        config.models.classification_model = classification_model
    segmentation_model = _env_str(settings.segmentation_model, "SEGMENTATION_MODEL")
    if segmentation_model:
        config.models.segmentation_model = segmentation_model
    obb_model = _env_str(settings.obb_model, "OBB_MODEL")
    if obb_model:
        config.models.obb_model = obb_model

    tracking = _env_bool(settings.tracking, "TRACKING")
    if tracking is not None:
        config.features.tracking = tracking
    fall_detection = _env_bool(settings.fall_detection, "FALL_DETECTION")
    if fall_detection is not None:
        config.features.fall_detection = fall_detection
    count = _env_bool(settings.count, "COUNT")
    if count is not None:
        config.features.counting = count

    if detection_model:
        config.inference.model_path = _resolve_model_path(
            detection_model,
            config.inference.engine,
        )

    db_enabled = _env_bool(settings.db_enabled, "DB_ENABLED")
    if db_enabled is not None:
        config.database.enabled = db_enabled
    db_host = _env_str(settings.db_host, "DB_HOST")
    if db_host:
        config.database.host = db_host
    db_port = _env_int(settings.db_port, "DB_PORT")
    if db_port is not None:
        config.database.port = db_port
    db_name = _env_str(settings.db_name, "DB_NAME")
    if db_name:
        config.database.name = db_name
    db_user = _env_str(settings.db_user, "DB_USER")
    if db_user:
        config.database.user = db_user
    db_password = _env_str(settings.db_password, "DB_PASSWORD")
    if db_password:
        config.database.password = db_password
    db_sslmode = _env_str(settings.db_sslmode, "DB_SSLMODE")
    if db_sslmode:
        config.database.sslmode = db_sslmode
    db_connect_timeout = _env_float(settings.db_connect_timeout_seconds, "DB_CONNECT_TIMEOUT_SECONDS")
    if db_connect_timeout is not None:
        config.database.connect_timeout_seconds = db_connect_timeout
    db_queue_size = _env_int(settings.db_queue_size, "DB_QUEUE_SIZE")
    if db_queue_size is not None:
        config.database.queue_size = db_queue_size
    db_flush_interval = _env_float(settings.db_flush_interval_seconds, "DB_FLUSH_INTERVAL_SECONDS")
    if db_flush_interval is not None:
        config.database.flush_interval_seconds = db_flush_interval

    camera_config_path = Path("config/camera_config.json")
    camera_defaults = _camera_defaults(settings)
    camera_config = _load_camera_config(camera_config_path, camera_defaults)
    if camera_config:
        from .utils.config import CameraConfig

        config.cameras = [CameraConfig(**cam) for cam in camera_config[0]]
        if camera_config[1]:
            config.people_count.zones = camera_config[1]
    else:
        camera_urls = _env_str(settings.camera_urls, "CAMERA_URLS")
        if camera_urls:
            urls = _split_csv(camera_urls)
            camera_ids = _env_str(settings.camera_ids, "CAMERA_IDS")
            ids = _split_csv(camera_ids) if camera_ids else []
            cameras = []
            for idx, url in enumerate(urls):
                camera_id = ids[idx] if idx < len(ids) else f"cam_{idx + 1}"
                cameras.append(
                    {
                        "camera_id": camera_id,
                        "rtsp_url": _coerce_rtsp(url),
                        "name": camera_id,
                        "enabled": True,
                        "fps_limit": camera_defaults[0],
                        "queue_size": camera_defaults[1],
                        "reconnect_min_backoff": camera_defaults[2],
                        "reconnect_max_backoff": camera_defaults[3],
                    }
                )
            from .utils.config import CameraConfig

            config.cameras = [CameraConfig(**cam) for cam in cameras]


def _normalize_backend(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"tensorrt", "trt"}:
        return "tensorrt"
    if normalized in {"onnx", "onnxruntime"}:
        return "onnx"
    if normalized in {"openvino", "vino"}:
        return "openvino"
    if normalized in {"ultralytics", "pytorch", "pt"}:
        return "ultralytics"
    return normalized


def _resolve_model_path(model_name: str, engine: str) -> str:
    from .utils.model_registry import get_registry

    if "/" in model_name or model_name.endswith((".pt", ".onnx", ".engine", ".xml")):
        return model_name

    registry = get_registry()
    key_map = {
        "tensorrt": "engine",
        "trt": "engine",
        "onnx": "onnx",
        "openvino": "openvino_xml",
        "ultralytics": "pt",
        "pt": "pt",
        "pytorch": "pt",
    }
    file_key = key_map.get(engine.lower(), "pt")
    entry = registry.file(model_name, file_key)
    if entry:
        return str(entry.path)
    return _fallback_model_path(model_name, engine)


def _fallback_model_path(model_name: str, engine: str) -> str:
    if "/" in model_name or "." in model_name:
        return model_name
    normalized = engine.lower()
    if normalized in {"tensorrt", "trt"}:
        primary = f"models/trt-engines/{model_name}.engine"
        legacy = f"models/trt-engine/{model_name}.engine"
        if os.path.exists(legacy) and not os.path.exists(primary):
            return legacy
        return primary
    if normalized == "onnx":
        return f"models/onnx/{model_name}.onnx"
    if normalized == "openvino":
        return f"models/openvino/{model_name}_openvino_model"
    return f"models/{model_name}.pt"


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_size_list(value: str) -> Optional[Tuple[int, int]]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _coerce_rtsp(value: str):
    if value.isdigit():
        return int(value)
    return value


def _env_str(current: Optional[str], env_name: str) -> Optional[str]:
    if current is not None and current != "":
        return current
    return os.environ.get(env_name)


def _env_int(current: Optional[int], env_name: str) -> Optional[int]:
    if current is not None:
        return current
    value = os.environ.get(env_name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(current: Optional[float], env_name: str) -> Optional[float]:
    if current is not None:
        return current
    value = os.environ.get(env_name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _env_bool(current: Optional[bool], env_name: str) -> Optional[bool]:
    if current is not None:
        return current
    value = os.environ.get(env_name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _camera_defaults(settings: Settings) -> Tuple[float, int, float, float]:
    fps_limit = _env_float(settings.camera_fps_limit, "CAMERA_FPS_LIMIT")
    if fps_limit is None:
        fps_limit = 30.0
    queue_size = _env_int(settings.camera_queue_size, "CAMERA_QUEUE_SIZE")
    if queue_size is None:
        queue_size = 30
    min_backoff = _env_float(settings.reconnect_min_backoff, "RECONNECT_MIN_BACKOFF")
    if min_backoff is None:
        min_backoff = 1.0
    max_backoff = _env_float(settings.reconnect_max_backoff, "RECONNECT_MAX_BACKOFF")
    if max_backoff is None:
        max_backoff = 30.0
    return fps_limit, queue_size, min_backoff, max_backoff


def _load_camera_config(path: Path, defaults: Tuple[float, int, float, float]) -> Optional[Tuple[List[Dict], Dict[str, List[List[float]]]]]:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    cameras_raw = raw.get("cameras") if isinstance(raw, dict) else raw
    if not cameras_raw:
        return None
    fps_limit, queue_size, min_backoff, max_backoff = defaults
    cameras: List[Dict] = []
    zones: Dict[str, List[List[float]]] = {}
    for idx, cam in enumerate(cameras_raw):
        camera_id = cam.get("camera_id") or cam.get("id") or f"cam_{idx + 1}"
        if "rtsp_url" in cam:
            rtsp_url = cam.get("rtsp_url")
        else:
            rtsp_url = cam.get("url")
        if rtsp_url is None:
            continue
        cameras.append(
            {
                "camera_id": camera_id,
                "rtsp_url": _coerce_rtsp(str(rtsp_url)),
                "name": cam.get("name", camera_id),
                "enabled": bool(cam.get("enabled", True)),
                "fps_limit": float(cam.get("fps_limit", fps_limit)),
                "queue_size": int(cam.get("queue_size", queue_size)),
                "reconnect_min_backoff": float(cam.get("reconnect_min_backoff", min_backoff)),
                "reconnect_max_backoff": float(cam.get("reconnect_max_backoff", max_backoff)),
            }
        )
        masking = cam.get("masking_coordinates")
        if masking:
            zones[camera_id] = masking
    if not cameras:
        return None
    return cameras, zones
