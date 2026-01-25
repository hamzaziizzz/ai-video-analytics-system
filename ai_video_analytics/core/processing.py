import asyncio
import base64
import copy
import io
import time
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import cv2
import numpy as np
from fastapi import Depends

from ai_video_analytics.schemas import Images
from ai_video_analytics.settings import Settings

from ai_video_analytics.core.inference.factory import create_engine, create_pose_engine
from ai_video_analytics.core.inference.warmup import warmup_engine
from ai_video_analytics.core.model_zoo.yolo import prepare_yolo_inference_assets
from ai_video_analytics.core.model_zoo.reid import prepare_reid_assets
from ai_video_analytics.core.settings import Settings as CoreSettings, build_default_config
from ai_video_analytics.core.utils.image_provider import get_images
from ai_video_analytics.core.utils.logging import get_logger


_COCO_SKELETON = (
    (0, 1),
    (0, 2),
    (0, 5),
    (0, 6),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
_KPT_CONF_THRES = 0.25
_PALETTE_HEX = (
    "042AFF",
    "0BDBEB",
    "F3F3F3",
    "00DFB7",
    "111F68",
    "FF6FDD",
    "FF444F",
    "CCED00",
    "00F344",
    "BD00FF",
    "00B4FF",
    "DD00BA",
    "00FFFF",
    "26C000",
    "01FFB3",
    "7D24FF",
    "7B0068",
    "FF1B6C",
    "FC6D2F",
    "A2FF0B",
)
_POSE_PALETTE = (
    (0, 128, 255),
    (51, 153, 255),
    (102, 178, 255),
    (0, 230, 230),
    (255, 153, 255),
    (255, 204, 153),
    (255, 102, 255),
    (255, 51, 255),
    (255, 178, 102),
    (255, 153, 51),
    (153, 153, 255),
    (102, 102, 255),
    (51, 51, 255),
    (153, 255, 153),
    (102, 255, 102),
    (51, 255, 51),
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 255),
)
_KPT_COLOR_IDX = (16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9)
_LIMB_COLOR_IDX = (9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16)


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


_PALETTE = tuple(_hex_to_bgr(value) for value in _PALETTE_HEX)


def _line_width(image: np.ndarray) -> int:
    return max(int(round((image.shape[0] + image.shape[1]) / 2 * 0.003)), 2)


def _font_scale(line_width: int) -> float:
    return max(line_width / 3, 0.5)


def _font_thickness(line_width: int) -> int:
    return max(line_width - 1, 1)


def _text_color(bg_color: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 160 else (255, 255, 255)


class Processing:
    """Core request handler for person and pose detection."""
    def __init__(
        self,
        det_name: str = "yolo26x",
        max_size: Optional[List[int]] = None,
        backend_name: str = "trt",
        max_det_batch_size: int = 1,
        force_fp16: bool = False,
        triton_uri=None,
        root_dir: str = "/models",
        pose_enabled: bool = True,
        pose_name: Optional[str] = None,
        **_kwargs,
    ) -> None:
        if max_size is None:
            max_size = [640, 640]

        self.max_det_batch_size = max(1, int(max_det_batch_size))
        self.det_name = det_name
        self.max_size = max_size
        self.backend_name = backend_name
        self.force_fp16 = force_fp16
        self.triton_uri = triton_uri
        self.root_dir = root_dir
        self.dl_client = None
        self.engine = None
        self.config = None
        self.pose_enabled = pose_enabled
        self.pose_name = pose_name
        self.pose_batch_size = self.max_det_batch_size
        self.pose_max_size = list(self.max_size)
        self.pose_engine = None
        self.pose_config = None
        self.logger = get_logger("processing")
        self.tracking_enabled = False
        self.det_tracker = None
        self.pose_tracker = None

    async def start(self, dl_client=None):
        """Initialize inference engine and shared runtime dependencies."""
        self.dl_client = dl_client
        core_settings = CoreSettings()
        config = build_default_config(core_settings)
        runtime_defaults = Settings()
        config.models.detection_model = self.det_name
        config.inference.engine = _normalize_backend(self.backend_name)
        config.inference.batch_size = self.max_det_batch_size
        config.inference.fp16 = self.force_fp16
        config.inference.confidence_threshold = runtime_defaults.defaults.det_thresh
        self.tracking_enabled = bool(config.features.tracking)
        if (
            config.inference.engine != "ultralytics"
            and not config.inference.labels
            and config.people_count.class_id is not None
            and config.people_count.class_name
        ):
            labels = [str(i) for i in range(config.people_count.class_id + 1)]
            labels[config.people_count.class_id] = config.people_count.class_name
            config.inference.labels = labels
        if self.max_size:
            config.inference.input_size = (self.max_size[1], self.max_size[0])
        self.config = config
        prepare_yolo_inference_assets(config)
        self.engine = create_engine(config.inference)
        self.engine.load()
        engine_name = type(self.engine).__name__ if self.engine else "unknown"
        self.logger.info("Inference engine: %s", engine_name)
        if config.inference.engine in {"tensorrt", "trt"} and self.engine:
            self.logger.info(
                "TRT runtime: impl=%s gpu_preproc=%s (active=%s) numba_decode=%s cupy_nms=%s "
                "no_letterbox=%s gpu_timing=%s",
                config.inference.trt_implementation,
                getattr(self.engine, "use_gpu_preproc", None),
                getattr(self.engine, "_use_gpu_preproc", None),
                getattr(self.engine, "use_numba_decode", None),
                getattr(self.engine, "use_cupy_nms", None),
                getattr(self.engine, "no_letterbox", None),
                getattr(self.engine, "gpu_timing", None),
            )
        warmup_batch = min(
            config.inference.batch_size,
            getattr(self.engine, "max_batch_size", config.inference.batch_size),
        )
        warmup_engine(
            self.engine,
            config.inference.input_size,
            warmup_batch,
            logger_name="processing.warmup",
        )

        if self.tracking_enabled:
            prepare_reid_assets(config)
            from ai_video_analytics.core.tracking import TrackerConfig, create_tracker

            tracker_type = getattr(config.features, "tracker_type", "bytetrack")
            tracker_config = TrackerConfig(
                use_numba=getattr(config.features, "tracker_numba", True),
                reid_model=getattr(config.features, "tracker_reid_model", None),
                reid_batch_size=getattr(config.features, "tracker_reid_batch_size", 32),
                reid_device=getattr(config.features, "tracker_reid_device", None),
                matching=getattr(config.features, "tracker_matching", "greedy"),
            )
            self.det_tracker = create_tracker(tracker_type, tracker_config)
            try:
                from ai_video_analytics.core.inference.warmup import warmup_reid

                warmup_reid(
                    getattr(self.det_tracker, "reid_encoder", None),
                    getattr(config.features, "tracker_reid_batch_size", 1),
                    logger_name="processing.reid_warmup",
                )
            except Exception:
                self.logger.warning("ReID warmup skipped: failed to initialize warmup")

        if self._pose_enabled():
            pose_config = copy.deepcopy(config)
            pose_config.models.detection_model = self.pose_name
            if self.pose_name:
                pose_path = Path(str(self.pose_name))
                if pose_path.suffix or str(self.pose_name).strip().startswith(("/", ".")):
                    pose_config.inference.model_path = str(pose_path)
            pose_batch_size = getattr(config.features, "pose_batch_size", None)
            if pose_batch_size:
                pose_config.inference.batch_size = pose_batch_size
            self.pose_config = pose_config
            prepare_yolo_inference_assets(pose_config)
            self.pose_engine = create_pose_engine(pose_config.inference)
            self.pose_engine.load()
            pose_engine_name = type(self.pose_engine).__name__ if self.pose_engine else "unknown"
            self.logger.info("Pose inference engine: %s", pose_engine_name)
            self.pose_batch_size = pose_config.inference.batch_size
            self.pose_max_size = [pose_config.inference.input_size[1], pose_config.inference.input_size[0]]
            warmup_batch = min(
                pose_config.inference.batch_size,
                getattr(self.pose_engine, "max_batch_size", pose_config.inference.batch_size),
            )
            warmup_engine(
                self.pose_engine,
                pose_config.inference.input_size,
                warmup_batch,
                logger_name="processing.pose_warmup",
            )
            if self.tracking_enabled:
                from ai_video_analytics.core.tracking import TrackerConfig, create_tracker

                tracker_type = getattr(pose_config.features, "tracker_type", "bytetrack")
                tracker_config = TrackerConfig(
                    use_numba=getattr(pose_config.features, "tracker_numba", True),
                    reid_model=getattr(pose_config.features, "tracker_reid_model", None),
                    reid_batch_size=getattr(pose_config.features, "tracker_reid_batch_size", 32),
                    reid_device=getattr(pose_config.features, "tracker_reid_device", None),
                    matching=getattr(pose_config.features, "tracker_matching", "greedy"),
                )
                self.pose_tracker = create_tracker(tracker_type, tracker_config)
                try:
                    from ai_video_analytics.core.inference.warmup import warmup_reid

                    warmup_reid(
                        getattr(self.pose_tracker, "reid_encoder", None),
                        getattr(pose_config.features, "tracker_reid_batch_size", 1),
                        logger_name="processing.pose_reid_warmup",
                    )
                except Exception:
                    self.logger.warning("Pose ReID warmup skipped: failed to initialize warmup")

    async def extract(
        self,
        images: Images,
        max_size: Optional[List[int]] = None,
        threshold: float = 0.6,
        limit_people: int = 0,
        min_person_size: int = 0,
        return_person_data: bool = False,
        reset_tracking: bool = False,
        verbose_timings: bool = True,
        b64_decode: bool = True,
        img_req_headers: Optional[dict] = None,
        **_kwargs,
    ) -> Dict:
        """Run person detection on one or more images and return structured results."""
        if img_req_headers is None:
            img_req_headers = {}

        if not max_size:
            max_size = self.max_size

        if self.det_tracker and reset_tracking:
            self.det_tracker.reset()

        t0 = time.time()
        output = dict(took={}, data=[])

        tl0 = time.time()
        image_entries = await get_images(
            images,
            decode=True,
            session=self.dl_client,
            b64_decode=b64_decode,
            headers=img_req_headers,
        )
        tl1 = time.time()
        took_loading = tl1 - tl0

        valid_images = []
        index_map = []
        for idx, entry in enumerate(image_entries):
            if entry.get("traceback") is None and entry.get("data") is not None:
                valid_images.append(entry["data"])
                index_map.append(idx)

        detections_per_image: Dict[int, List[dict]] = {}
        track_history_per_image: Dict[int, List[dict]] = {}
        took_ms_per_image: Dict[int, float] = {}
        took_breakdown_per_image: Dict[int, Dict[str, float]] = {}
        pre_total_ms = 0.0
        infer_total_ms = 0.0
        decode_total_ms = 0.0
        post_total_ms = 0.0
        gpu_pre_total_ms = 0.0
        gpu_infer_total_ms = 0.0
        te0 = time.time()
        if valid_images and self.engine:
            for start in range(0, len(valid_images), self.max_det_batch_size):
                batch = valid_images[start : start + self.max_det_batch_size]
                tb0 = time.perf_counter()
                det_batches = self.engine.infer_batch(batch)
                batch_ms = (time.perf_counter() - tb0) * 1000
                per_image_ms = batch_ms / max(1, len(batch))
                batch_timings = getattr(self.engine, "last_batch_timings", None)
                per_pre_ms = None
                per_infer_ms = None
                per_decode_ms = None
                per_gpu_pre_ms = None
                per_gpu_infer_ms = None
                if (
                    isinstance(batch_timings, dict)
                    and batch_timings.get("batch") == len(batch)
                    and all(key in batch_timings for key in ("pre_ms", "infer_ms", "decode_ms"))
                ):
                    per_pre_ms = float(batch_timings["pre_ms"]) / max(1, len(batch))
                    per_infer_ms = float(batch_timings["infer_ms"]) / max(1, len(batch))
                    per_decode_ms = float(batch_timings["decode_ms"]) / max(1, len(batch))
                    if batch_timings.get("gpu_pre_ms") is not None:
                        per_gpu_pre_ms = float(batch_timings["gpu_pre_ms"]) / max(1, len(batch))
                    if batch_timings.get("gpu_infer_ms") is not None:
                        per_gpu_infer_ms = float(batch_timings["gpu_infer_ms"]) / max(1, len(batch))
                for offset, dets in enumerate(det_batches):
                    img_index = index_map[start + offset]
                    tp0 = time.perf_counter()
                    track_ids = {}
                    if self.det_tracker:
                        track_ids = self.det_tracker.update(dets, batch[offset])
                        track_history_per_image[img_index] = self._serialize_tracks(self.det_tracker.history())
                    detections_per_image[img_index] = self._serialize_detections(
                        dets,
                        threshold=threshold,
                        limit_people=limit_people,
                        min_person_size=min_person_size,
                        image=batch[offset],
                        return_person_data=return_person_data,
                        track_ids=track_ids,
                    )
                    post_ms = (time.perf_counter() - tp0) * 1000
                    post_total_ms += post_ms
                    if per_pre_ms is not None and per_infer_ms is not None and per_decode_ms is not None:
                        total_ms = per_pre_ms + per_infer_ms + per_decode_ms + post_ms
                        took_ms_per_image[img_index] = total_ms
                        took_breakdown_per_image[img_index] = {
                            "preproc_ms": per_pre_ms,
                            "infer_ms": per_infer_ms,
                            "decode_ms": per_decode_ms,
                            "post_ms": post_ms,
                            "total_ms": total_ms,
                        }
                        if per_gpu_pre_ms is not None:
                            took_breakdown_per_image[img_index]["gpu_preproc_ms"] = per_gpu_pre_ms
                            gpu_pre_total_ms += per_gpu_pre_ms
                        if per_gpu_infer_ms is not None:
                            took_breakdown_per_image[img_index]["gpu_infer_ms"] = per_gpu_infer_ms
                            gpu_infer_total_ms += per_gpu_infer_ms
                        pre_total_ms += per_pre_ms
                        infer_total_ms += per_infer_ms
                        decode_total_ms += per_decode_ms
                    else:
                        took_ms_per_image[img_index] = per_image_ms + post_ms
                    await asyncio.sleep(0)
        took_detect = time.time() - te0
        took = time.time() - t0
        output["took"]["total_ms"] = took * 1000
        if verbose_timings:
            output["took"]["read_imgs_ms"] = took_loading * 1000
            output["took"]["pose_all_ms"] = took_detect * 1000
            if pre_total_ms or infer_total_ms or decode_total_ms or post_total_ms:
                output["took"]["pose_pre_ms"] = pre_total_ms
                output["took"]["pose_infer_ms"] = infer_total_ms
                output["took"]["pose_decode_ms"] = decode_total_ms
                output["took"]["pose_post_ms"] = post_total_ms
            if gpu_pre_total_ms or gpu_infer_total_ms:
                output["took"]["pose_gpu_pre_ms"] = gpu_pre_total_ms
                output["took"]["pose_gpu_infer_ms"] = gpu_infer_total_ms

        for idx, entry in enumerate(image_entries):
            item = {"status": "failed", "took_ms": 0.0, "people": []}
            if entry.get("traceback") is not None:
                item["traceback"] = entry.get("traceback")
            else:
                item["status"] = "ok"
                item["people"] = detections_per_image.get(idx, [])
                if self.det_tracker:
                    item["tracks"] = track_history_per_image.get(idx, [])
                if idx in took_ms_per_image:
                    item["took_ms"] = took_ms_per_image[idx]
                if idx in took_breakdown_per_image:
                    item["took_breakdown_ms"] = took_breakdown_per_image[idx]
            output["data"].append(item)

        return output

    async def extract_pose(
        self,
        images: Images,
        max_size: Optional[List[int]] = None,
        threshold: float = 0.6,
        limit_people: int = 0,
        min_person_size: int = 0,
        reset_tracking: bool = False,
        verbose_timings: bool = True,
        b64_decode: bool = True,
        img_req_headers: Optional[dict] = None,
        **_kwargs,
    ) -> Dict:
        """Run pose detection on one or more images and return structured results."""
        if img_req_headers is None:
            img_req_headers = {}
        if not max_size:
            max_size = self.pose_max_size
        if not self.pose_engine:
            raise RuntimeError("Pose engine is not loaded")

        if self.pose_tracker and reset_tracking:
            self.pose_tracker.reset()

        t0 = time.time()
        output = dict(took={}, data=[])

        tl0 = time.time()
        image_entries = await get_images(
            images,
            decode=True,
            session=self.dl_client,
            b64_decode=b64_decode,
            headers=img_req_headers,
        )
        tl1 = time.time()
        took_loading = tl1 - tl0

        valid_images = []
        index_map = []
        for idx, entry in enumerate(image_entries):
            if entry.get("traceback") is None and entry.get("data") is not None:
                valid_images.append(entry["data"])
                index_map.append(idx)

        poses_per_image: Dict[int, List[dict]] = {}
        track_history_per_image: Dict[int, List[dict]] = {}
        took_ms_per_image: Dict[int, float] = {}
        took_breakdown_per_image: Dict[int, Dict[str, float]] = {}
        pre_total_ms = 0.0
        infer_total_ms = 0.0
        decode_total_ms = 0.0
        post_total_ms = 0.0
        gpu_pre_total_ms = 0.0
        gpu_infer_total_ms = 0.0
        te0 = time.time()
        if valid_images and self.pose_engine:
            for start in range(0, len(valid_images), self.pose_batch_size):
                batch = valid_images[start : start + self.pose_batch_size]
                tb0 = time.perf_counter()
                pose_batches = self.pose_engine.infer_batch(batch)
                batch_ms = (time.perf_counter() - tb0) * 1000
                per_image_ms = batch_ms / max(1, len(batch))
                batch_timings = getattr(self.pose_engine, "last_batch_timings", None)
                per_pre_ms = None
                per_infer_ms = None
                per_decode_ms = None
                per_gpu_pre_ms = None
                per_gpu_infer_ms = None
                if (
                    isinstance(batch_timings, dict)
                    and batch_timings.get("batch") == len(batch)
                    and all(key in batch_timings for key in ("pre_ms", "infer_ms", "decode_ms"))
                ):
                    per_pre_ms = float(batch_timings["pre_ms"]) / max(1, len(batch))
                    per_infer_ms = float(batch_timings["infer_ms"]) / max(1, len(batch))
                    per_decode_ms = float(batch_timings["decode_ms"]) / max(1, len(batch))
                    if batch_timings.get("gpu_pre_ms") is not None:
                        per_gpu_pre_ms = float(batch_timings["gpu_pre_ms"]) / max(1, len(batch))
                    if batch_timings.get("gpu_infer_ms") is not None:
                        per_gpu_infer_ms = float(batch_timings["gpu_infer_ms"]) / max(1, len(batch))
                for offset, dets in enumerate(pose_batches):
                    img_index = index_map[start + offset]
                    tp0 = time.perf_counter()
                    track_ids = {}
                    if self.pose_tracker:
                        track_ids = self.pose_tracker.update(dets, batch[offset])
                        track_history_per_image[img_index] = self._serialize_tracks(self.pose_tracker.history())
                    poses_per_image[img_index] = self._serialize_pose_detections(
                        dets,
                        threshold=threshold,
                        limit_people=limit_people,
                        min_person_size=min_person_size,
                        track_ids=track_ids,
                    )
                    post_ms = (time.perf_counter() - tp0) * 1000
                    post_total_ms += post_ms
                    if per_pre_ms is not None and per_infer_ms is not None and per_decode_ms is not None:
                        total_ms = per_pre_ms + per_infer_ms + per_decode_ms + post_ms
                        took_ms_per_image[img_index] = total_ms
                        took_breakdown_per_image[img_index] = {
                            "preproc_ms": per_pre_ms,
                            "infer_ms": per_infer_ms,
                            "decode_ms": per_decode_ms,
                            "post_ms": post_ms,
                            "total_ms": total_ms,
                        }
                        if per_gpu_pre_ms is not None:
                            took_breakdown_per_image[img_index]["gpu_preproc_ms"] = per_gpu_pre_ms
                            gpu_pre_total_ms += per_gpu_pre_ms
                        if per_gpu_infer_ms is not None:
                            took_breakdown_per_image[img_index]["gpu_infer_ms"] = per_gpu_infer_ms
                            gpu_infer_total_ms += per_gpu_infer_ms
                        pre_total_ms += per_pre_ms
                        infer_total_ms += per_infer_ms
                        decode_total_ms += per_decode_ms
                    else:
                        took_ms_per_image[img_index] = per_image_ms + post_ms
                    await asyncio.sleep(0)
        took_detect = time.time() - te0
        took = time.time() - t0
        output["took"]["total_ms"] = took * 1000
        if verbose_timings:
            output["took"]["read_imgs_ms"] = took_loading * 1000
            output["took"]["detect_all_ms"] = took_detect * 1000
            if pre_total_ms or infer_total_ms or decode_total_ms or post_total_ms:
                output["took"]["detect_pre_ms"] = pre_total_ms
                output["took"]["detect_infer_ms"] = infer_total_ms
                output["took"]["detect_decode_ms"] = decode_total_ms
                output["took"]["detect_post_ms"] = post_total_ms
            if gpu_pre_total_ms or gpu_infer_total_ms:
                output["took"]["detect_gpu_pre_ms"] = gpu_pre_total_ms
                output["took"]["detect_gpu_infer_ms"] = gpu_infer_total_ms

        for idx, entry in enumerate(image_entries):
            item = {"status": "failed", "took_ms": 0.0, "poses": []}
            if entry.get("traceback") is not None:
                item["traceback"] = entry.get("traceback")
            else:
                item["status"] = "ok"
                item["poses"] = poses_per_image.get(idx, [])
                if self.pose_tracker:
                    item["tracks"] = track_history_per_image.get(idx, [])
                if idx in took_ms_per_image:
                    item["took_ms"] = took_ms_per_image[idx]
                if idx in took_breakdown_per_image:
                    item["took_breakdown_ms"] = took_breakdown_per_image[idx]
            output["data"].append(item)

        return output

    async def draw(
        self,
        images: Images | bytes,
        threshold: float = 0.6,
        draw_scores: bool = True,
        draw_sizes: bool = True,
        limit_people: int = 0,
        min_person_size: int = 0,
        multipart: bool = False,
        **_kwargs,
    ) -> io.BytesIO:
        """Return an annotated image with detected people drawn."""
        if not multipart:
            image_entries = await get_images(images, session=self.dl_client)
            image = image_entries[0].get("data") if image_entries else None
        else:
            data = np.frombuffer(images, np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError("Failed to decode image")

        dets = []
        if self.engine:
            dets = self.engine.infer_batch([image])[0]
        dets = self._filter_detections(dets, threshold, limit_people, min_person_size)
        if self.det_tracker:
            self.det_tracker.update(dets, image)
            track_history = self._serialize_tracks(self.det_tracker.history())
            if track_history:
                image = self._draw_track_trails(image, track_history)
        image = self._draw_detections(image, dets, draw_scores=draw_scores, draw_sizes=draw_sizes)

        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            raise RuntimeError("Failed to encode result image")
        return io.BytesIO(buffer)

    async def draw_pose(
        self,
        images: Images | bytes,
        threshold: float = 0.6,
        draw_scores: bool = True,
        draw_sizes: bool = True,
        limit_people: int = 0,
        min_person_size: int = 0,
        multipart: bool = False,
        **_kwargs,
    ) -> io.BytesIO:
        """Return an annotated image with detected poses drawn."""
        if not self.pose_engine:
            raise RuntimeError("Pose detection engine is not initialized")

        if not multipart:
            image_entries = await get_images(images, session=self.dl_client)
            image = image_entries[0].get("data") if image_entries else None
        else:
            data = np.frombuffer(images, np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError("Failed to decode image")

        dets = self.pose_engine.infer_batch([image])[0]
        dets = self._filter_detections(dets, threshold, limit_people, min_person_size)
        if self.pose_tracker:
            self.pose_tracker.update(dets, image)
            track_history = self._serialize_tracks(self.pose_tracker.history())
            if track_history:
                image = self._draw_track_trails(image, track_history)
        image = self._draw_pose_detections(image, dets, draw_scores=draw_scores, draw_sizes=draw_sizes)

        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            raise RuntimeError("Failed to encode result image")
        return io.BytesIO(buffer)

    def _serialize_detections(
        self,
        dets,
        threshold: float,
        limit_people: int,
        min_person_size: int,
        image: np.ndarray,
        return_person_data: bool,
        track_ids: Optional[Dict[int, int]] = None,
    ) -> List[dict]:
        """Convert detections to the response-friendly dict structure."""
        filtered: List[tuple[int, object]] = []
        for idx, det in enumerate(dets):
            if det.score < threshold:
                continue
            x1, y1, x2, y2 = det.bbox
            if min_person_size > 0 and (x2 - x1) < min_person_size:
                continue
            filtered.append((idx, det))
        if limit_people and limit_people > 0:
            filtered = filtered[:limit_people]
        results: List[dict] = []
        for out_idx, (det_idx, det) in enumerate(filtered):
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            item = {
                "bbox": [x1, y1, x2, y2],
                "prob": float(det.score),
                "class_id": int(det.class_id),
                "class_name": det.class_name,
                "num_det": out_idx,
            }
            if track_ids and det_idx in track_ids:
                item["track_id"] = int(track_ids[det_idx])
            if return_person_data:
                crop = image[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
                item["persondata"] = _encode_crop(crop) if crop.size else None
            results.append(item)
        return results

    def _serialize_pose_detections(
        self,
        dets,
        threshold: float,
        limit_people: int,
        min_person_size: int,
        track_ids: Optional[Dict[int, int]] = None,
    ) -> List[dict]:
        """Convert pose detections to the response-friendly dict structure."""
        filtered: List[tuple[int, object]] = []
        for idx, det in enumerate(dets):
            if det.score < threshold:
                continue
            x1, y1, x2, y2 = det.bbox
            if min_person_size > 0 and (x2 - x1) < min_person_size:
                continue
            filtered.append((idx, det))
        if limit_people and limit_people > 0:
            filtered = filtered[:limit_people]
        results: List[dict] = []
        for out_idx, (det_idx, det) in enumerate(filtered):
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            keypoints = det.keypoints
            if isinstance(keypoints, np.ndarray):
                keypoints = keypoints.tolist()
            if keypoints is None:
                keypoints = []
            item = {
                "bbox": [x1, y1, x2, y2],
                "prob": float(det.score),
                "class_id": int(det.class_id),
                "class_name": det.class_name,
                "keypoints": keypoints,
                "num_det": out_idx,
            }
            if track_ids and det_idx in track_ids:
                item["track_id"] = int(track_ids[det_idx])
            results.append(item)
        return results

    def _serialize_tracks(self, history: Dict[int, List[tuple[int, int]]]) -> List[dict]:
        if not history:
            return []
        tracks: List[dict] = []
        for track_id, points in history.items():
            if not points:
                continue
            tracks.append(
                {
                    "track_id": int(track_id),
                    "points": [[int(x), int(y)] for x, y in points],
                }
            )
        tracks.sort(key=lambda item: item["track_id"])
        return tracks

    def _filter_detections(self, dets, threshold, limit_people, min_person_size):
        """Apply score/size filtering and optional top-k limiting."""
        filtered = []
        for det in dets:
            if det.score < threshold:
                continue
            x1, y1, x2, y2 = det.bbox
            if min_person_size > 0 and (x2 - x1) < min_person_size:
                continue
            filtered.append(det)
        if limit_people and limit_people > 0:
            filtered = filtered[:limit_people]
        return filtered

    def _draw_detections(self, image, detections, draw_scores: bool, draw_sizes: bool):
        """Draw bounding boxes and optional labels on an image."""
        line_width = _line_width(image)
        font_scale = _font_scale(line_width)
        font_thickness = _font_thickness(line_width)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color_idx = det.class_id if det.class_id is not None else 0
            color = _PALETTE[color_idx % len(_PALETTE)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width, lineType=cv2.LINE_AA)
            if draw_scores:
                label = det.class_name
                if getattr(det, "track_id", None) is not None:
                    label = f"{label} #{det.track_id}"
                label = f"{label} {det.score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                y0 = max(y1 - th - 6, 0)
                y1_label = y0 + th + 6
                cv2.rectangle(image, (x1, y0), (x1 + tw + 6, y1_label), color, -1)
                cv2.putText(
                    image,
                    label,
                    (x1 + 3, y1_label - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    _text_color(color),
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )
            if draw_sizes:
                size = int(x2 - x1)
                label = f"w:{size}"
                cv2.putText(
                    image,
                    label,
                    (x1, min(y2 + 10, image.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.9,
                    color,
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )
        return image

    def _draw_pose_detections(self, image, detections, draw_scores: bool, draw_sizes: bool):
        """Draw bounding boxes and keypoints on an image."""
        height, width = image.shape[:2]
        line_width = _line_width(image)
        font_scale = _font_scale(line_width)
        font_thickness = _font_thickness(line_width)
        kpt_radius = max(line_width, 2)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color_idx = det.class_id if det.class_id is not None else 0
            color = _PALETTE[color_idx % len(_PALETTE)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width, lineType=cv2.LINE_AA)
            if draw_scores:
                label = det.class_name
                if getattr(det, "track_id", None) is not None:
                    label = f"{label} #{det.track_id}"
                label = f"{label} {det.score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                y0 = max(y1 - th - 6, 0)
                y1_label = y0 + th + 6
                cv2.rectangle(image, (x1, y0), (x1 + tw + 6, y1_label), color, -1)
                cv2.putText(
                    image,
                    label,
                    (x1 + 3, y1_label - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    _text_color(color),
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )
            if draw_sizes:
                size = int(x2 - x1)
                label = f"w:{size}"
                cv2.putText(
                    image,
                    label,
                    (x1, min(y2 + 10, image.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.9,
                    color,
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )
            keypoints = det.keypoints
            if isinstance(keypoints, np.ndarray):
                keypoints = keypoints.tolist()
            if not keypoints:
                continue
            for kp_idx, kp in enumerate(keypoints):
                if not kp or len(kp) < 2:
                    continue
                x_val, y_val = kp[0], kp[1]
                if x_val <= 0 or y_val <= 0 or x_val >= width or y_val >= height:
                    continue
                if len(kp) >= 3 and kp[2] < _KPT_CONF_THRES:
                    continue
                color_k = _POSE_PALETTE[_KPT_COLOR_IDX[kp_idx % len(_KPT_COLOR_IDX)]]
                cv2.circle(
                    image,
                    (int(x_val), int(y_val)),
                    kpt_radius,
                    color_k,
                    -1,
                    lineType=cv2.LINE_AA,
                )
            if len(keypoints) == 17:
                for edge_idx, (a, b) in enumerate(_COCO_SKELETON):
                    kp_a = keypoints[a]
                    kp_b = keypoints[b]
                    if len(kp_a) < 2 or len(kp_b) < 2:
                        continue
                    ax, ay = kp_a[0], kp_a[1]
                    bx, by = kp_b[0], kp_b[1]
                    if ax <= 0 or ay <= 0 or ax >= width or ay >= height:
                        continue
                    if bx <= 0 or by <= 0 or bx >= width or by >= height:
                        continue
                    if len(kp_a) >= 3 and kp_a[2] < _KPT_CONF_THRES:
                        continue
                    if len(kp_b) >= 3 and kp_b[2] < _KPT_CONF_THRES:
                        continue
                    limb_color = _POSE_PALETTE[_LIMB_COLOR_IDX[edge_idx % len(_LIMB_COLOR_IDX)]]
                    cv2.line(
                        image,
                        (int(ax), int(ay)),
                        (int(bx), int(by)),
                        limb_color,
                        max(int(round(line_width / 2)), 1),
                        lineType=cv2.LINE_AA,
                    )
        return image

    def _draw_track_trails(self, image: np.ndarray, tracks: List[dict]) -> np.ndarray:
        if not tracks:
            return image
        line_width = max(int(round(_line_width(image) * 1.2)), 2)
        head_radius = max(line_width * 2, 2)
        trail_color = (0, 0, 255)
        for track in tracks:
            points = track.get("points") or []
            if not points:
                continue
            if len(points) >= 2:
                max_thickness = line_width
                min_thickness = max(1, int(round(line_width * 0.4)))
                last = points[0]
                for idx in range(1, len(points)):
                    frac = idx / (len(points) - 1)
                    thickness = int(round(min_thickness + (max_thickness - min_thickness) * frac))
                    cv2.line(
                        image,
                        (int(last[0]), int(last[1])),
                        (int(points[idx][0]), int(points[idx][1])),
                        trail_color,
                        max(thickness, 1),
                        lineType=cv2.LINE_AA,
                    )
                    last = points[idx]
            head = points[-1]
            cv2.circle(
                image,
                (int(head[0]), int(head[1])),
                head_radius,
                trail_color,
                -1,
                lineType=cv2.LINE_AA,
            )
        return image

    def _pose_enabled(self) -> bool:
        if not self.pose_enabled:
            return False
        if not self.pose_name:
            return False
        value = str(self.pose_name).strip().lower()
        return value not in {"none", "off", "false", "0"}


def _encode_crop(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _normalize_backend(name: str) -> str:
    normalized = (name or "").strip().lower()
    if normalized in {"trt", "tensorrt"}:
        return "tensorrt"
    if normalized in {"onnx", "onnxruntime"}:
        return "onnx"
    if normalized in {"pt", "pytorch", "ultralytics"}:
        return "ultralytics"
    if normalized in {"openvino", "vino"}:
        return "openvino"
    return normalized


processing: Processing | None = None


async def get_processing() -> Processing:
    global processing
    if not processing:
        settings = Settings()
        processing = Processing(
            det_name=settings.models.det_name,
            max_size=settings.models.max_size,
            max_det_batch_size=settings.models.det_batch_size,
            backend_name=settings.models.inference_backend,
            force_fp16=settings.models.force_fp16,
            triton_uri=settings.models.triton_uri,
            root_dir="/models",
            pose_enabled=settings.models.pose_detection,
            pose_name=settings.models.pose_name,
        )
    return processing


ProcessingDep = Annotated[Processing, Depends(get_processing)]
