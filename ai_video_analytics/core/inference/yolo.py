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

_CUPY_NMS_MIN_BOXES = 200


class PreprocessMeta:
    """Per-image preproc metadata used to rescale detections back to original size."""
    def __init__(
        self,
        scale_x: float,
        scale_y: float,
        pad_x: float,
        pad_y: float,
        orig_shape: Tuple[int, int],
    ):
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.orig_shape = orig_shape


class DecodeWorkspace:
    """Reusable buffers to avoid per-call allocations during YOLO decode."""
    def __init__(self):
        self.capacity = 0
        self.dtype: Optional[np.dtype] = None
        self.boxes: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None
        self.class_ids: Optional[np.ndarray] = None
        self.dets: Optional[np.ndarray] = None
        self.row_index: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None
        self.kpts: Optional[np.ndarray] = None
        self.kpts_shape: Optional[tuple[int, int, int]] = None
        self.kpts_dtype: Optional[np.dtype] = None

    def ensure(self, capacity: int, dtype: np.dtype) -> None:
        if self.capacity >= capacity and self.dtype == dtype:
            return
        self.capacity = max(1, capacity)
        self.dtype = dtype
        self.boxes = np.zeros((self.capacity, 4), dtype=dtype)
        self.scores = np.zeros((self.capacity,), dtype=dtype)
        self.class_ids = np.zeros((self.capacity,), dtype=np.int32)
        self.dets = np.zeros((self.capacity, 5), dtype=dtype)
        self.row_index = np.arange(self.capacity, dtype=np.int32)
        self.indices = np.zeros((self.capacity,), dtype=np.int32)

    def ensure_kpts(self, count: int, num_kpts: int, kpt_dim: int, dtype: np.dtype) -> None:
        shape = (max(1, count), max(1, num_kpts), max(1, kpt_dim))
        if self.kpts_shape == shape and self.kpts_dtype == dtype and self.kpts is not None:
            return
        self.kpts_shape = shape
        self.kpts_dtype = dtype
        self.kpts = np.zeros(shape, dtype=dtype)


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
        interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, new_unpad, interpolation=interp)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, ratio, (dw, dh)


def resize_no_letterbox(image, new_shape: Tuple[int, int]):
    """Resize to target shape without padding (aspect ratio may change)."""
    import cv2

    orig_h, orig_w = image.shape[:2]
    new_h, new_w = new_shape
    if (orig_h, orig_w) != (new_h, new_w):
        scale_x = new_w / max(1, orig_w)
        scale_y = new_h / max(1, orig_h)
        interp = cv2.INTER_AREA if scale_x < 1.0 or scale_y < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
    else:
        scale_x = 1.0
        scale_y = 1.0
    return image, scale_x, scale_y


def prepare_input(frame, input_size) -> Tuple[np.ndarray, PreprocessMeta]:
    import cv2

    img, ratio, (dw, dh) = letterbox(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    meta = PreprocessMeta(scale_x=ratio, scale_y=ratio, pad_x=dw, pad_y=dh, orig_shape=frame.shape[:2])
    return img, meta


def prepare_batch(frames, input_size: Tuple[int, int], out: Optional[np.ndarray] = None):
    metas = []
    if out is None:
        imgs = []
        for frame in frames:
            img, ratio, (dw, dh) = letterbox(frame, input_size)
            imgs.append(img)
            metas.append(PreprocessMeta(scale_x=ratio, scale_y=ratio, pad_x=dw, pad_y=dh, orig_shape=frame.shape[:2]))
        batch = np.stack(imgs, axis=0)
        return batch, metas

    if out.shape[0] < len(frames):
        raise ValueError("Output batch buffer is smaller than the requested batch size")
    if out.shape[1] != input_size[0] or out.shape[2] != input_size[1] or out.shape[3] != 3:
        raise ValueError("Output batch buffer shape does not match input_size")

    for idx, frame in enumerate(frames):
        img, ratio, (dw, dh) = letterbox(frame, input_size)
        out[idx] = img
        metas.append(PreprocessMeta(scale_x=ratio, scale_y=ratio, pad_x=dw, pad_y=dh, orig_shape=frame.shape[:2]))
    return out[: len(frames)], metas


def prepare_batch_no_letterbox(frames, input_size: Tuple[int, int], out: Optional[np.ndarray] = None):
    """Batch resize without letterbox padding, tracking per-axis scales."""
    metas = []
    if out is None:
        imgs = []
        for frame in frames:
            img, scale_x, scale_y = resize_no_letterbox(frame, input_size)
            imgs.append(img)
            metas.append(
                PreprocessMeta(
                    scale_x=scale_x,
                    scale_y=scale_y,
                    pad_x=0.0,
                    pad_y=0.0,
                    orig_shape=frame.shape[:2],
                )
            )
        batch = np.stack(imgs, axis=0)
        return batch, metas

    if out.shape[0] < len(frames):
        raise ValueError("Output batch buffer is smaller than the requested batch size")
    if out.shape[1] != input_size[0] or out.shape[2] != input_size[1] or out.shape[3] != 3:
        raise ValueError("Output batch buffer shape does not match input_size")

    for idx, frame in enumerate(frames):
        img, scale_x, scale_y = resize_no_letterbox(frame, input_size)
        out[idx] = img
        metas.append(
            PreprocessMeta(
                scale_x=scale_x,
                scale_y=scale_y,
                pad_x=0.0,
                pad_y=0.0,
                orig_shape=frame.shape[:2],
            )
        )
    return out[: len(frames)], metas


def rescale_boxes(boxes: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    # Reverse letterbox/resize scaling and padding to map back to original image.
    boxes[:, [0, 2]] -= meta.pad_x
    boxes[:, [1, 3]] -= meta.pad_y
    boxes[:, [0, 2]] /= meta.scale_x
    boxes[:, [1, 3]] /= meta.scale_y

    h, w = meta.orig_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
    return boxes


def rescale_keypoints(keypoints: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    """Reverse letterbox/resize scaling for keypoints."""
    if keypoints.size == 0:
        return keypoints

    keypoints[..., 0] -= meta.pad_x
    keypoints[..., 1] -= meta.pad_y
    keypoints[..., 0] /= meta.scale_x
    keypoints[..., 1] /= meta.scale_y

    h, w = meta.orig_shape
    keypoints[..., 0] = np.clip(keypoints[..., 0], 0, w)
    keypoints[..., 1] = np.clip(keypoints[..., 1], 0, h)
    return keypoints


def _infer_kpt_dim(extra: int) -> Optional[int]:
    if extra <= 0:
        return None
    if extra % 3 == 0:
        return 3
    if extra % 2 == 0:
        return 2
    return None


def _choose_pose_layout(
    pred_dim: int,
    num_classes: int,
    has_objectness: bool,
) -> tuple[int, bool, Optional[int]]:
    """Infer pose layout (classes/objectness/kpt_dim) from prediction width."""
    candidates = []
    class_candidates = [num_classes]
    if num_classes != 1:
        class_candidates.append(1)
    for nc in class_candidates:
        for obj_flag in (has_objectness, not has_objectness):
            base = (5 + nc) if obj_flag else (4 + nc)
            extra = pred_dim - base
            kpt_dim = _infer_kpt_dim(extra) if extra >= 0 else None
            if kpt_dim:
                score = 0
                if kpt_dim == 3:
                    score += 2
                if nc == num_classes:
                    score += 1
                if obj_flag == has_objectness:
                    score += 1
                candidates.append((score, nc, obj_flag, kpt_dim))
    if not candidates:
        return num_classes, has_objectness, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, nc, obj_flag, kpt_dim = candidates[0]
    return nc, obj_flag, kpt_dim


def _reshape_keypoints(flat: np.ndarray, kpt_dim: int) -> np.ndarray:
    if kpt_dim <= 0:
        return np.empty((flat.shape[0], 0, 0), dtype=flat.dtype)
    return flat.reshape(flat.shape[0], -1, kpt_dim)


@njit(cache=True)
def _reshape_keypoints_numba(flat: np.ndarray, kpt_dim: int, out: np.ndarray) -> None:
    n = flat.shape[0]
    k = flat.shape[1] // kpt_dim
    for i in range(n):
        base = 0
        for j in range(k):
            for d in range(kpt_dim):
                out[i, j, d] = flat[i, base + d]
            base += kpt_dim


@njit(cache=True)
def _rescale_keypoints_numba(
    keypoints: np.ndarray,
    scale_x: float,
    scale_y: float,
    pad_x: float,
    pad_y: float,
    orig_h: int,
    orig_w: int,
) -> None:
    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            x = (keypoints[i, j, 0] - pad_x) / scale_x
            y = (keypoints[i, j, 1] - pad_y) / scale_y
            if x < 0:
                x = 0.0
            elif x > orig_w:
                x = float(orig_w)
            if y < 0:
                y = 0.0
            elif y > orig_h:
                y = float(orig_h)
            keypoints[i, j, 0] = x
            keypoints[i, j, 1] = y


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


@njit(cache=True)
def _decode_candidates(
    preds: np.ndarray,
    num_classes: int,
    has_obj: bool,
    conf_thresh: float,
    class_filter: np.ndarray,
    out_boxes: np.ndarray,
    out_scores: np.ndarray,
    out_class_ids: np.ndarray,
    out_indices: np.ndarray,
) -> int:
    count = 0
    filter_len = class_filter.shape[0]
    for i in range(preds.shape[0]):
        if has_obj:
            obj = preds[i, 4]
            max_score = -1.0
            max_idx = -1
            for c in range(num_classes):
                score = preds[i, 5 + c]
                if score > max_score:
                    max_score = score
                    max_idx = c
            score = obj * max_score
        else:
            max_score = -1.0
            max_idx = -1
            for c in range(num_classes):
                score = preds[i, 4 + c]
                if score > max_score:
                    max_score = score
                    max_idx = c
            score = max_score

        if score < conf_thresh:
            continue
        if filter_len > 0:
            ok = False
            for j in range(filter_len):
                if max_idx == class_filter[j]:
                    ok = True
                    break
            if not ok:
                continue

        x = preds[i, 0]
        y = preds[i, 1]
        w = preds[i, 2]
        h = preds[i, 3]
        out_boxes[count, 0] = x - w / 2.0
        out_boxes[count, 1] = y - h / 2.0
        out_boxes[count, 2] = x + w / 2.0
        out_boxes[count, 3] = y + h / 2.0
        out_scores[count] = score
        out_class_ids[count] = max_idx
        out_indices[count] = i
        count += 1
    return count


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
    use_numba_decode: bool = False,
    workspace: Optional[DecodeWorkspace] = None,
    return_keypoints: bool = False,
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

    if end_to_end and preds.shape[1] < 6:
        end_to_end = False
    elif not end_to_end and preds.shape[1] == 6:
        end_to_end = True

    if not end_to_end and return_keypoints and preds.shape[1] >= 6:
        # NMS-exported pose models return (N, 6 + K*D); raw outputs have many more rows.
        if preds.shape[0] <= preds.shape[1] * 10:
            end_to_end = True

    class_filter = np.array(class_id_filter, dtype=int) if class_id_filter else None

    if end_to_end:
        extra = preds.shape[1] - 6
        kpt_dim = _infer_kpt_dim(extra) if return_keypoints else None
        boxes = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5].astype(int)
        kpts = None
        if return_keypoints and kpt_dim:
            kpts_flat = preds[:, 6 : 6 + extra]
            if use_numba_decode and _HAS_NUMBA and workspace is not None:
                num_kpts = kpts_flat.shape[1] // kpt_dim
                workspace.ensure_kpts(kpts_flat.shape[0], num_kpts, kpt_dim, kpts_flat.dtype)
                _reshape_keypoints_numba(
                    kpts_flat,
                    kpt_dim,
                    workspace.kpts[: kpts_flat.shape[0], :num_kpts, :kpt_dim],
                )
                kpts = workspace.kpts[: kpts_flat.shape[0], :num_kpts, :kpt_dim]
            else:
                kpts = _reshape_keypoints(kpts_flat, kpt_dim)

        mask = scores >= confidence_threshold
        if class_filter is not None:
            mask &= np.isin(class_ids, class_filter)
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        if kpts is not None:
            kpts = kpts[mask]

        if boxes.size == 0:
            return []

        boxes = rescale_boxes(boxes, meta)
        if kpts is not None:
            if use_numba_decode and _HAS_NUMBA:
                _rescale_keypoints_numba(
                    kpts,
                    meta.scale_x,
                    meta.scale_y,
                    meta.pad_x,
                    meta.pad_y,
                    meta.orig_shape[0],
                    meta.orig_shape[1],
                )
            else:
                kpts = rescale_keypoints(kpts, meta)

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
                    keypoints=kpts[idx].tolist() if kpts is not None else None,
                )
            )
        return detections

    num_classes = len(labels)
    if return_keypoints and num_classes <= 0:
        num_classes = 1
    if num_classes <= 0:
        return []
    has_obj = has_objectness
    kpt_dim = None
    kpts_raw = None
    if return_keypoints:
        num_classes, has_obj, kpt_dim = _choose_pose_layout(preds.shape[1], num_classes, has_obj)
        base = (5 + num_classes) if has_obj else (4 + num_classes)
        extra = preds.shape[1] - base
        if kpt_dim and extra >= kpt_dim:
            kpts_raw = preds[:, base : base + extra]
    elif not has_objectness:
        expected = 5 + num_classes
        if preds.shape[1] == expected:
            has_obj = True

    if use_numba_decode and _HAS_NUMBA and workspace is not None:
        preds_work = preds.astype(np.float32, copy=False)
        workspace.ensure(preds_work.shape[0], preds_work.dtype)
        class_filter_arr = (
            np.array(class_id_filter, dtype=np.int32) if class_id_filter else np.empty(0, dtype=np.int32)
        )
        count = _decode_candidates(
            preds_work,
            num_classes,
            has_obj,
            confidence_threshold,
            class_filter_arr,
            workspace.boxes,
            workspace.scores,
            workspace.class_ids,
            workspace.indices,
        )
        if count == 0:
            return []

        boxes_xyxy = workspace.boxes[:count]
        scores = workspace.scores[:count]
        class_ids = workspace.class_ids[:count]
        boxes_xyxy = rescale_boxes(boxes_xyxy, meta)
        kpts = None
        if kpts_raw is not None and kpt_dim and workspace.indices is not None:
            selected = kpts_raw[workspace.indices[:count]]
            num_kpts = selected.shape[1] // kpt_dim
            workspace.ensure_kpts(selected.shape[0], num_kpts, kpt_dim, selected.dtype)
            if use_numba_decode and _HAS_NUMBA:
                _reshape_keypoints_numba(
                    selected,
                    kpt_dim,
                    workspace.kpts[: selected.shape[0], :num_kpts, :kpt_dim],
                )
                kpts = workspace.kpts[: selected.shape[0], :num_kpts, :kpt_dim]
            else:
                kpts = _reshape_keypoints(selected, kpt_dim)
        if kpts is not None:
            if use_numba_decode and _HAS_NUMBA:
                _rescale_keypoints_numba(
                    kpts,
                    meta.scale_x,
                    meta.scale_y,
                    meta.pad_x,
                    meta.pad_y,
                    meta.orig_shape[0],
                    meta.orig_shape[1],
                )
            else:
                kpts = rescale_keypoints(kpts, meta)

        use_gpu_nms = use_cupy_nms and _HAS_CUPY and boxes_xyxy.shape[0] >= _CUPY_NMS_MIN_BOXES
        if use_gpu_nms:
            keep = _nms_cupy(boxes_xyxy, scores, nms_iou_threshold)
        elif _HAS_NUMBA and workspace.dets is not None:
            workspace.dets[:count, :4] = boxes_xyxy
            workspace.dets[:count, 4] = scores
            keep = list(_nms_numba(workspace.dets[:count], nms_iou_threshold))
        else:
            keep = _nms_numpy(boxes_xyxy, scores, nms_iou_threshold)

        if kpts is not None:
            kpts = kpts[keep]

        detections: List[Detection] = []
        for keep_idx, det_idx in enumerate(keep):
            class_id = int(class_ids[det_idx])
            class_name = labels[class_id] if class_id < len(labels) else str(class_id)
            detections.append(
                Detection(
                    bbox=boxes_xyxy[det_idx].tolist(),
                    score=float(scores[det_idx]),
                    class_id=class_id,
                    class_name=class_name,
                    keypoints=kpts[keep_idx].tolist() if kpts is not None else None,
                )
            )
        return detections

    if has_obj:
        boxes = preds[:, :4]
        obj_conf = preds[:, 4]
        class_scores = preds[:, 5 : 5 + num_classes]
        class_ids = np.argmax(class_scores, axis=1)
        if workspace is not None and workspace.row_index is not None:
            row_index = workspace.row_index[: len(class_scores)]
        else:
            row_index = np.arange(len(class_scores))
        class_conf = class_scores[row_index, class_ids]
        scores = obj_conf * class_conf
    else:
        boxes = preds[:, :4]
        class_scores = preds[:, 4 : 4 + num_classes]
        class_ids = np.argmax(class_scores, axis=1)
        if workspace is not None and workspace.row_index is not None:
            row_index = workspace.row_index[: len(class_scores)]
        else:
            row_index = np.arange(len(class_scores))
        scores = class_scores[row_index, class_ids]

    mask = scores >= confidence_threshold
    if class_filter is not None:
        mask &= np.isin(class_ids, class_filter)
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    kpts = None
    if kpts_raw is not None and kpt_dim:
        selected = kpts_raw[mask]
        num_kpts = selected.shape[1] // kpt_dim
        if use_numba_decode and _HAS_NUMBA and workspace is not None:
            workspace.ensure_kpts(selected.shape[0], num_kpts, kpt_dim, selected.dtype)
            _reshape_keypoints_numba(
                selected,
                kpt_dim,
                workspace.kpts[: selected.shape[0], :num_kpts, :kpt_dim],
            )
            kpts = workspace.kpts[: selected.shape[0], :num_kpts, :kpt_dim]
        else:
            kpts = _reshape_keypoints(selected, kpt_dim)

    if boxes.size == 0:
        return []

    if workspace is not None:
        workspace.ensure(boxes.shape[0], boxes.dtype)
        boxes_xyxy = workspace.boxes[: boxes.shape[0]]
    else:
        boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    boxes_xyxy = rescale_boxes(boxes_xyxy, meta)
    if kpts is not None:
        if use_numba_decode and _HAS_NUMBA:
            _rescale_keypoints_numba(
                kpts,
                meta.scale_x,
                meta.scale_y,
                meta.pad_x,
                meta.pad_y,
                meta.orig_shape[0],
                meta.orig_shape[1],
            )
        else:
            kpts = rescale_keypoints(kpts, meta)

    use_gpu_nms = use_cupy_nms and _HAS_CUPY and boxes_xyxy.shape[0] >= _CUPY_NMS_MIN_BOXES
    if use_gpu_nms:
        keep = _nms_cupy(boxes_xyxy, scores, nms_iou_threshold)
    elif _HAS_NUMBA:
        dets = np.concatenate([boxes_xyxy, scores[:, None]], axis=1)
        dets = np.ascontiguousarray(dets)
        keep = list(_nms_numba(dets, nms_iou_threshold))
    else:
        keep = _nms_numpy(boxes_xyxy, scores, nms_iou_threshold)

    if kpts is not None:
        kpts = kpts[keep]

    detections: List[Detection] = []
    for keep_idx, det_idx in enumerate(keep):
        class_id = int(class_ids[det_idx])
        class_name = labels[class_id] if class_id < len(labels) else str(class_id)
        detections.append(
            Detection(
                bbox=boxes_xyxy[det_idx].tolist(),
                score=float(scores[det_idx]),
                class_id=class_id,
                class_name=class_name,
                keypoints=kpts[keep_idx].tolist() if kpts is not None else None,
            )
        )
    return detections
