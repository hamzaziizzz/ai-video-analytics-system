import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..utils.logging import get_logger


def warmup_engine(
    engine,
    input_size: Tuple[int, int],
    batch_size: int,
    logger_name: str = "inference.warmup",
    image_path: Optional[str] = None,
) -> None:
    """Warm up the inference engine using a real sample image."""
    if batch_size <= 0:
        return
    iters = 5
    logger = get_logger(logger_name)
    image = _load_warmup_image(input_size, image_path=image_path, logger=logger)
    if image is None:
        logger.warning("Warmup skipped: test image not available")
        return

    batch = [image for _ in range(batch_size)]
    height, width = image.shape[:2]
    logger.info("Warming up inference engine (batch=%d iters=%d input=%dx%d)", batch_size, iters, width, height)
    try:
        for _ in range(iters):
            engine.infer_batch(batch)
    except Exception as exc:
        logger.warning("Warmup failed: %s", exc)
        return
    logger.info("Warmup complete")


def warmup_reid(
    encoder,
    batch_size: int,
    logger_name: str = "tracking.reid_warmup",
    image_path: Optional[str] = None,
) -> None:
    """Warm up ReID encoder using a repeated full-frame crop."""
    if encoder is None or batch_size <= 0:
        return
    iters = 5
    logger = get_logger(logger_name)
    image = _load_warmup_image(None, image_path=image_path, logger=logger)
    if image is None:
        logger.warning("ReID warmup skipped: test image not available")
        return
    height, width = image.shape[:2]
    batch_size = min(batch_size, _reid_max_batch(encoder))
    if batch_size <= 0:
        return
    boxes = np.tile(np.array([[0, 0, width, height]], dtype=np.float32), (batch_size, 1))
    logger.info("Warming up ReID encoder (batch=%d iters=%d input=%dx%d)", batch_size, iters, width, height)
    try:
        for _ in range(iters):
            encoder.extract(image, boxes)
    except Exception as exc:
        logger.warning("ReID warmup failed: %s", exc)
        return
    logger.info("ReID warmup complete")


def _reid_max_batch(encoder) -> int:
    trt = getattr(encoder, "_trt", None)
    max_batch = getattr(trt, "max_batch_size", None) if trt is not None else None
    if max_batch is None:
        return getattr(encoder, "batch_size", 1)
    return max(1, int(max_batch))


def _load_warmup_image(
    input_size: Optional[Tuple[int, int]],
    image_path: Optional[str],
    logger,
) -> Optional[np.ndarray]:
    path = _resolve_warmup_path(image_path)
    if path is None:
        return None
    try:
        import cv2
    except Exception:
        logger.warning("cv2 is required for warmup image loading")
        return None
    image = cv2.imread(str(path))
    if image is None:
        return None
    if input_size:
        height, width = input_size
        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _resolve_warmup_path(image_path: Optional[str]) -> Optional[Path]:
    if image_path:
        path = Path(image_path)
        if path.exists():
            return path
    candidates = []
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / "misc" / "test_images" / "person.jpg")
    cwd_path = Path.cwd() / "misc" / "test_images" / "person.jpg"
    candidates.append(cwd_path)
    root_images = os.environ.get("ROOT_IMAGES_DIR", "/images")
    candidates.append(Path(root_images) / "test_images" / "person.jpg")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
