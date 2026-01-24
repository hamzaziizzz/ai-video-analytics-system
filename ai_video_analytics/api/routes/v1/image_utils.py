import math
from typing import List

import cv2
import numpy as np


def tile_images(images: List[np.ndarray]) -> np.ndarray:
    """Tile images into a square-ish grid for preview."""
    if not images:
        raise RuntimeError("No images to tile")
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    channels = images[0].shape[2] if images[0].ndim == 3 else 1
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    tiled = np.zeros((rows * max_h, cols * max_w, channels), dtype=images[0].dtype)
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        height, width = img.shape[:2]
        if height != max_h or width != max_w:
            scale = min(max_w / width, max_h / height)
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((max_h, max_w, channels), dtype=images[0].dtype)
            y0 = (max_h - new_h) // 2
            x0 = (max_w - new_w) // 2
            canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
            img = canvas
        y0 = row * max_h
        x0 = col * max_w
        tiled[y0 : y0 + max_h, x0 : x0 + max_w] = img
    return tiled
