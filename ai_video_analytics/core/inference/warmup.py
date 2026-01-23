from typing import Tuple

import numpy as np

from ..utils.logging import get_logger


def warmup_engine(
    engine,
    input_size: Tuple[int, int],
    batch_size: int,
    iters: int,
    logger_name: str = "inference.warmup",
) -> None:
    """Run a short dummy pass to warm up the inference engine."""
    if iters <= 0 or batch_size <= 0:
        return

    height, width = input_size
    batch = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(batch_size)]
    logger = get_logger(logger_name)
    logger.info(
        "Warming up inference engine (batch=%d iters=%d input=%dx%d)",
        batch_size,
        iters,
        width,
        height,
    )
    try:
        for _ in range(iters):
            engine.infer_batch(batch)
    except Exception as exc:
        logger.warning("Warmup failed: %s", exc)
        return
    logger.info("Warmup complete")
