import asyncio
import base64
import io
import time
from typing import Annotated, Dict, List, Optional

import cv2
import numpy as np
from fastapi import Depends

from ai_video_analytics.schemas import Images
from ai_video_analytics.settings import Settings

from ai_video_analytics.core.inference.factory import create_engine
from ai_video_analytics.core.inference.warmup import warmup_engine
from ai_video_analytics.core.model_zoo.yolo import prepare_yolo_inference_assets
from ai_video_analytics.core.settings import Settings as CoreSettings, build_default_config
from ai_video_analytics.core.utils.image_provider import get_images
from ai_video_analytics.core.utils.logging import get_logger


class Processing:
    """Core request handler for person detection and visualization."""
    def __init__(
        self,
        det_name: str = "yolo26x",
        max_size: Optional[List[int]] = None,
        backend_name: str = "trt",
        max_det_batch_size: int = 1,
        force_fp16: bool = False,
        triton_uri=None,
        root_dir: str = "/models",
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
        self.logger = get_logger("processing")

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
        warmup_batch = min(config.inference.warmup_batch_size, config.inference.batch_size)
        max_batch = getattr(self.engine, "max_batch_size", warmup_batch)
        warmup_batch = min(warmup_batch, max_batch)
        warmup_engine(
            self.engine,
            config.inference.input_size,
            warmup_batch,
            config.inference.warmup_iters,
            logger_name="processing.warmup",
        )

    async def extract(
        self,
        images: Images,
        max_size: Optional[List[int]] = None,
        threshold: float = 0.6,
        limit_people: int = 0,
        min_person_size: int = 0,
        return_person_data: bool = False,
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
        took_ms_per_image: Dict[int, float] = {}
        te0 = time.time()
        if valid_images and self.engine:
            for start in range(0, len(valid_images), self.max_det_batch_size):
                batch = valid_images[start : start + self.max_det_batch_size]
                tb0 = time.perf_counter()
                det_batches = self.engine.infer_batch(batch)
                batch_ms = (time.perf_counter() - tb0) * 1000
                per_image_ms = batch_ms / max(1, len(batch))
                for offset, dets in enumerate(det_batches):
                    img_index = index_map[start + offset]
                    tp0 = time.perf_counter()
                    detections_per_image[img_index] = self._serialize_detections(
                        dets,
                        threshold=threshold,
                        limit_people=limit_people,
                        min_person_size=min_person_size,
                        image=batch[offset],
                        return_person_data=return_person_data,
                    )
                    post_ms = (time.perf_counter() - tp0) * 1000
                    took_ms_per_image[img_index] = per_image_ms + post_ms
                    await asyncio.sleep(0)
        took_detect = time.time() - te0
        took = time.time() - t0
        output["took"]["total_ms"] = took * 1000
        if verbose_timings:
            output["took"]["read_imgs_ms"] = took_loading * 1000
            output["took"]["detect_all_ms"] = took_detect * 1000

        for idx, entry in enumerate(image_entries):
            item = {"status": "failed", "took_ms": 0.0, "people": []}
            if entry.get("traceback") is not None:
                item["traceback"] = entry.get("traceback")
            else:
                item["status"] = "ok"
                item["people"] = detections_per_image.get(idx, [])
                if idx in took_ms_per_image:
                    item["took_ms"] = took_ms_per_image[idx]
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
        image = self._draw_detections(image, dets, draw_scores=draw_scores, draw_sizes=draw_sizes)

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
    ) -> List[dict]:
        """Convert detections to the response-friendly dict structure."""
        filtered = self._filter_detections(dets, threshold, limit_people, min_person_size)
        total = len(filtered)
        results: List[dict] = []
        for det in filtered:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            item = {
                "bbox": [x1, y1, x2, y2],
                "prob": float(det.score),
                "class_id": int(det.class_id),
                "class_name": det.class_name,
                "num_det": total,
            }
            if return_person_data:
                crop = image[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
                item["persondata"] = _encode_crop(crop) if crop.size else None
            results.append(item)
        return results

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
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            if draw_scores:
                label = f"{det.class_name} {det.score:.2f}"
                cv2.putText(
                    image,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            if draw_sizes:
                size = int(x2 - x1)
                label = f"w:{size}"
                cv2.putText(
                    image,
                    label,
                    (x1, min(y2 + 15, image.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
        return image


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
        )
    return processing


ProcessingDep = Annotated[Processing, Depends(get_processing)]
