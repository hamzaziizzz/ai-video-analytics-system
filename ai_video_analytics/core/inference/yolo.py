from typing import List, Optional, Tuple

import numpy as np

from .base import Detection

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_NUMBA = False

    def njit(*_args, **_kwargs):  # type: ignore
        def wrapper(func):
            return func

        return wrapper


try:
    import cupy as cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_CUPY = False
    cp = None  # type: ignore


class PreprocessMeta:
    def __init__(self, scale: float, pad_x: float, pad_y: float, orig_shape: Tuple[int, int]):
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.orig_shape = orig_shape


def letterbox(image, new_shape: Tuple[int, int], color: Tuple[int, int, int] = (114, 114, 114)):
    import cv2

    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, ratio, (dw, dh)


def prepare_input(frame, input_size) -> Tuple[np.ndarray, PreprocessMeta]:
    import cv2

    img, ratio, (dw, dh) = letterbox(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    meta = PreprocessMeta(scale=ratio, pad_x=dw, pad_y=dh, orig_shape=frame.shape[:2])
    return img, meta


def prepare_batch(frames, input_size: Tuple[int, int]):
    imgs = []
    metas = []
    for frame in frames:
        img, ratio, (dw, dh) = letterbox(frame, input_size)
        imgs.append(img)
        metas.append(PreprocessMeta(scale=ratio, pad_x=dw, pad_y=dh, orig_shape=frame.shape[:2]))
    batch = np.stack(imgs, axis=0)
    return batch, metas


def rescale_boxes(boxes: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    boxes[:, [0, 2]] -= meta.pad_x
    boxes[:, [1, 3]] -= meta.pad_y
    boxes[:, :4] /= meta.scale

    h, w = meta.orig_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
    return boxes


@njit(cache=True)
def _nms_numba(dets: np.ndarray, thresh: float) -> List[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def _nms_cupy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if not _HAS_CUPY:
        return _nms_numpy(boxes, scores, iou_threshold)
    if boxes.size == 0:
        return []

    boxes_cp = cp.asarray(boxes, dtype=cp.float32)
    scores_cp = cp.asarray(scores, dtype=cp.float32)

    order = scores_cp.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = cp.maximum(boxes_cp[i, 0], boxes_cp[rest, 0])
        yy1 = cp.maximum(boxes_cp[i, 1], boxes_cp[rest, 1])
        xx2 = cp.minimum(boxes_cp[i, 2], boxes_cp[rest, 2])
        yy2 = cp.minimum(boxes_cp[i, 3], boxes_cp[rest, 3])

        w = cp.maximum(0.0, xx2 - xx1 + 1.0)
        h = cp.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        area_i = (boxes_cp[i, 2] - boxes_cp[i, 0] + 1.0) * (boxes_cp[i, 3] - boxes_cp[i, 1] + 1.0)
        area_rest = (
            (boxes_cp[rest, 2] - boxes_cp[rest, 0] + 1.0)
            * (boxes_cp[rest, 3] - boxes_cp[rest, 1] + 1.0)
        )
        iou = inter / (area_i + area_rest - inter + 1e-6)
        order = rest[iou <= iou_threshold]

    return keep


def decode_yolo_output(
    output: np.ndarray,
    labels: List[str],
    confidence_threshold: float,
    nms_iou_threshold: float,
    has_objectness: bool,
    meta: PreprocessMeta,
    end_to_end: bool = False,
    output_layout: str = "",
    class_id_filter: Optional[List[int]] = None,
    use_cupy_nms: bool = False,
) -> List[Detection]:
    preds = output
    while preds.ndim > 2:
        preds = preds[0]

    if preds.ndim != 2:
        return []

    layout = (output_layout or "").lower()
    if layout in {"channel_first", "channels_first", "chw"}:
        preds = preds.T
    elif layout in {"end_to_end", "end2end"}:
        end_to_end = True
        if preds.shape[0] == 6 and preds.shape[1] != 6:
            preds = preds.T
    elif not layout and preds.shape[0] < preds.shape[1] and preds.shape[0] >= 6:
        preds = preds.T

    if preds.shape[1] < 6:
        return []

    if end_to_end and preds.shape[1] != 6:
        end_to_end = False
    elif not end_to_end and preds.shape[1] == 6:
        end_to_end = True

    class_filter = np.array(class_id_filter, dtype=int) if class_id_filter else None

    if end_to_end:
        boxes = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5].astype(int)

        mask = scores >= confidence_threshold
        if class_filter is not None:
            mask &= np.isin(class_ids, class_filter)
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if boxes.size == 0:
            return []

        boxes = rescale_boxes(boxes, meta)

        detections: List[Detection] = []
        for idx in range(len(boxes)):
            class_id = int(class_ids[idx])
            class_name = labels[class_id] if class_id < len(labels) else str(class_id)
            detections.append(
                Detection(
                    bbox=boxes[idx].tolist(),
                    score=float(scores[idx]),
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections

    num_classes = len(labels)
    has_obj = has_objectness
    if not has_objectness:
        expected = 5 + num_classes
        if preds.shape[1] == expected:
            has_obj = True

    if has_obj:
        boxes = preds[:, :4]
        obj_conf = preds[:, 4]
        class_scores = preds[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(class_scores)), class_ids]
        scores = obj_conf * class_conf
    else:
        boxes = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]

    mask = scores >= confidence_threshold
    if class_filter is not None:
        mask &= np.isin(class_ids, class_filter)
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if boxes.size == 0:
        return []

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    boxes_xyxy = rescale_boxes(boxes_xyxy, meta)

    if use_cupy_nms and _HAS_CUPY:
        keep = _nms_cupy(boxes_xyxy, scores, nms_iou_threshold)
    elif _HAS_NUMBA:
        dets = np.concatenate([boxes_xyxy, scores[:, None]], axis=1)
        dets = np.ascontiguousarray(dets)
        keep = list(_nms_numba(dets, nms_iou_threshold))
    else:
        keep = _nms_numpy(boxes_xyxy, scores, nms_iou_threshold)

    detections: List[Detection] = []
    for idx in keep:
        class_id = int(class_ids[idx])
        class_name = labels[class_id] if class_id < len(labels) else str(class_id)
        detections.append(
            Detection(
                bbox=boxes_xyxy[idx].tolist(),
                score=float(scores[idx]),
                class_id=class_id,
                class_name=class_name,
            )
        )
    return detections
