from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import numba as nb
except Exception:  # pragma: no cover - optional dependency
    nb = None  # type: ignore[assignment]

from ai_video_analytics.core.inference.base import Detection
from ai_video_analytics.core.utils.logging import get_logger


@dataclass
class TrackerConfig:
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    max_age: int = 30
    min_hits: int = 3
    iou_thresh: float = 0.3
    use_numba: bool = True
    reid_model: Optional[str] = None
    reid_batch_size: int = 32
    reid_device: Optional[str] = None
    matching: str = "greedy"


def _numba_enabled(config: TrackerConfig) -> bool:
    return bool(config.use_numba and nb is not None)


if nb is not None:

    @nb.njit(cache=True, fastmath=True)
    def _pairwise_iou_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
        for i in range(a.shape[0]):
            ax1, ay1, ax2, ay2 = a[i, 0], a[i, 1], a[i, 2], a[i, 3]
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            for j in range(b.shape[0]):
                bx1, by1, bx2, by2 = b[j, 0], b[j, 1], b[j, 2], b[j, 3]
                area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                x1 = max(ax1, bx1)
                y1 = max(ay1, by1)
                x2 = min(ax2, bx2)
                y2 = min(ay2, by2)
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = area_a + area_b - inter + 1e-6
                out[i, j] = inter / union
        return out

    @nb.njit(cache=True, fastmath=True)
    def _xyxy_to_xywh_batch_numba(xyxy: np.ndarray) -> np.ndarray:
        out = np.zeros((xyxy.shape[0], 4), dtype=np.float32)
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i, 0], xyxy[i, 1], xyxy[i, 2], xyxy[i, 3]
            out[i, 0] = x1 + (x2 - x1) / 2.0
            out[i, 1] = y1 + (y2 - y1) / 2.0
            out[i, 2] = x2 - x1
            out[i, 3] = y2 - y1
        return out

    @nb.njit(cache=True)
    def _greedy_match_numba(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, m = cost_matrix.shape
        match_cap = min(n, m)
        matches = np.empty((match_cap, 2), dtype=np.int64)
        used_a = np.zeros(n, dtype=np.uint8)
        used_b = np.zeros(m, dtype=np.uint8)
        match_count = 0
        while True:
            min_cost = thresh + 1.0
            min_i = -1
            min_j = -1
            for i in range(n):
                if used_a[i] != 0:
                    continue
                for j in range(m):
                    if used_b[j] != 0:
                        continue
                    cost = cost_matrix[i, j]
                    if cost < min_cost:
                        min_cost = cost
                        min_i = i
                        min_j = j
            if min_i < 0:
                break
            used_a[min_i] = 1
            used_b[min_j] = 1
            matches[match_count, 0] = min_i
            matches[match_count, 1] = min_j
            match_count += 1
        unmatched_a = np.zeros(n, dtype=np.int64)
        unmatched_b = np.zeros(m, dtype=np.int64)
        ua_count = 0
        ub_count = 0
        for i in range(n):
            if used_a[i] == 0:
                unmatched_a[ua_count] = i
                ua_count += 1
        for j in range(m):
            if used_b[j] == 0:
                unmatched_b[ub_count] = j
                ub_count += 1
        return matches[:match_count], unmatched_a[:ua_count], unmatched_b[:ub_count]


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.score = 0.0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    @staticmethod
    def reset_id() -> None:
        BaseTrack._count = 0


def _xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([x1 + (x2 - x1) / 2.0, y1 + (y2 - y1) / 2.0, x2 - x1, y2 - y1], dtype=np.float32)


def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def _pairwise_iou(a: np.ndarray, b: np.ndarray, use_numba: bool = False) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    if use_numba and nb is not None:
        return _pairwise_iou_numba(a, b)
    a = a[:, None, :]
    b = b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def _iou_distance(tracks: List["Track"], detections: List["Track"], use_numba: bool = False) -> np.ndarray:
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)
    a = np.stack([t.xyxy for t in tracks], axis=0)
    b = np.stack([d.xyxy for d in detections], axis=0)
    return 1.0 - _pairwise_iou(a, b, use_numba=use_numba)


def _fuse_score(cost_matrix: np.ndarray, detections: List["Track"]) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.array([det.score for det in detections], dtype=np.float32)[None, :]
    fused = iou_sim * det_scores
    return 1.0 - fused


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
    sim = np.dot(a_norm, b_norm.T)
    return np.maximum(0.0, 1.0 - sim)


def _linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
    use_numba: bool = False,
    use_hungarian: bool = False,
) -> Tuple[List[List[int]], List[int], List[int]]:
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    if use_numba and nb is not None and not use_hungarian:
        matches_arr, unmatched_a_arr, unmatched_b_arr = _greedy_match_numba(cost_matrix, float(thresh))
        return matches_arr.tolist(), unmatched_a_arr.tolist(), unmatched_b_arr.tolist()
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append([int(r), int(c)])
        unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in row_ind]
        unmatched_b = [i for i in range(cost_matrix.shape[1]) if i not in col_ind]
        return matches, unmatched_a, unmatched_b
    except Exception:
        pairs = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                pairs.append((cost_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0])
        used_a = set()
        used_b = set()
        matches = []
        for cost, i, j in pairs:
            if cost > thresh:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            matches.append([i, j])
        unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in used_a]
        unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in used_b]
        return matches, unmatched_a, unmatched_b


class KalmanFilterXYWH:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean.astype(np.float32), covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        projected_cov += np.diag(np.square(std))
        kalman_gain = np.linalg.multi_dot((covariance, self._update_mat.T, np.linalg.inv(projected_cov)))
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_cov = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_cov


class Track(BaseTrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, xywh: np.ndarray, score: float, cls: int, idx: int, feat: Optional[np.ndarray] = None):
        super().__init__()
        self._tlwh = _xywh_to_xyxy(xywh).astype(np.float32)
        self.kalman_filter: Optional[KalmanFilterXYWH] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.score = float(score)
        self.cls = int(cls)
        self.idx = int(idx)
        self.curr_feat = feat
        self.smooth_feat = feat
        self.hits = 0
        self.age = 0

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return _xyxy_to_xywh(self._tlwh).copy()
        ret = self.mean[:4].copy()
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        return _xywh_to_xyxy(self.mean[:4])

    def activate(self, kalman_filter: KalmanFilterXYWH, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.hits = 1
        self.age = 1

    def re_activate(self, new_track: "Track", frame_id: int, new_id: bool = False) -> None:
        if self.kalman_filter is None:
            self.kalman_filter = self.shared_kalman
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.tlwh)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

    def update(self, new_track: "Track", frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        if self.kalman_filter is None:
            self.kalman_filter = self.shared_kalman
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.tlwh)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.hits += 1
        self.age += 1

    def predict(self) -> None:
        if self.mean is None or self.covariance is None:
            return
        if self.state != TrackState.Tracked:
            self.mean[6:] = 0
        self.mean, self.covariance = self.shared_kalman.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update_features(self, feat: np.ndarray) -> None:
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = 0.9 * self.smooth_feat + 0.1 * feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-6

    @property
    def result(self) -> List[float]:
        coords = self.xyxy
        return [coords[0], coords[1], coords[2], coords[3], self.track_id, self.score, self.cls, self.idx]


class Detections:
    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        idx: np.ndarray,
        use_numba: bool = False,
    ):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.idx = idx
        self.use_numba = use_numba
        self._xywh_cache: Optional[np.ndarray] = None

    @property
    def xywh(self) -> np.ndarray:
        if self.xyxy.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if self._xywh_cache is None or self._xywh_cache.shape[0] != self.xyxy.shape[0]:
            if self.use_numba and nb is not None:
                self._xywh_cache = _xyxy_to_xywh_batch_numba(self.xyxy)
            else:
                out = np.empty((self.xyxy.shape[0], 4), dtype=np.float32)
                for i in range(self.xyxy.shape[0]):
                    out[i] = _xyxy_to_xywh(self.xyxy[i])
                self._xywh_cache = out
        return self._xywh_cache

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __getitem__(self, mask):
        return Detections(
            self.xyxy[mask],
            self.conf[mask],
            self.cls[mask],
            self.idx[mask],
            use_numba=self.use_numba,
        )


def _extract_color_features(image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
    try:
        import cv2
    except Exception:
        return [np.zeros((48,), dtype=np.float32) for _ in range(len(boxes))]
    features = []
    h, w = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            features.append(np.zeros((48,), dtype=np.float32))
            continue
        crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 3], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten().astype(np.float32)
        hist /= np.linalg.norm(hist) + 1e-6
        features.append(hist)
    return features


class ReIDEncoder:
    """Optional appearance encoder for DeepSORT/BoT-SORT using batched embeddings."""

    def __init__(self, model_path: str, device: Optional[str], batch_size: int) -> None:
        self.model_path = Path(model_path)
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.logger = get_logger("tracking.reid")
        self._input_name = None
        self._output_name = None
        self._input_size = (256, 128)
        self._use_torch = False
        self._trt = None
        self._session = None
        self._model = None
        self._batch_buf: Optional[np.ndarray] = None
        self._gpu_batch_buf = None
        self._gpu_batch_shape: Optional[tuple[int, int, int, int]] = None
        self._gpu_stream = None
        self._gpu_cv_stream = None
        self._gpu_upload = []
        self._gpu_resize = []
        self._gpu_preproc_enabled = False
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        self._mean_gpu = None
        self._std_gpu = None

        suffix = self.model_path.suffix.lower()
        if suffix == ".engine":
            self._init_tensorrt()
        elif suffix == ".onnx":
            self._init_onnx()
        elif suffix in {".pt", ".pth"}:
            self._init_torchscript()
        else:
            raise ValueError(f"Unsupported ReID model format: {self.model_path}")

        if self._trt is not None:
            self._setup_gpu_preproc()

    def _init_tensorrt(self) -> None:
        try:
            import tensorrt as trt
            import cupy as cp
            import cupyx
        except Exception as exc:
            raise RuntimeError("TensorRT and CuPy are required for ReID TRT inference") from exc

        device_id = 0
        if self.device and "cuda" in str(self.device).lower():
            try:
                device_id = int(str(self.device).split(":")[1])
            except (IndexError, ValueError):
                device_id = 0
        cp.cuda.Device(device_id).use()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as engine_file, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(engine_file.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine for ReID")
        self._trt = _ReIDTensorRT(engine, device_id=device_id)
        self._input_size = self._trt.input_hw
        self.logger.info("ReID TensorRT loaded: %s (input=%sx%s)", self.model_path, self._input_size[1], self._input_size[0])

    def _setup_gpu_preproc(self) -> None:
        try:
            import cv2
            import cupy as cp
        except Exception as exc:
            self.logger.warning("ReID GPU preproc unavailable: %s", exc)
            return
        if not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            self.logger.warning("ReID GPU preproc unavailable: OpenCV CUDA not detected")
            return
        self._gpu_preproc_enabled = True
        self._gpu_cv_stream = cv2.cuda.Stream()
        stream_ptr = None
        for attr in ("cudaPtr", "ptr"):
            if hasattr(self._gpu_cv_stream, attr):
                try:
                    value = getattr(self._gpu_cv_stream, attr)
                    stream_ptr = int(value() if callable(value) else value)
                    break
                except Exception:
                    continue
        if stream_ptr:
            self._gpu_stream = cp.cuda.ExternalStream(stream_ptr)
        else:
            self._gpu_cv_stream = None
            self._gpu_stream = cp.cuda.Stream.null
        self._mean_gpu = cp.asarray(self._mean)
        self._std_gpu = cp.asarray(self._std)
        self.logger.info("ReID GPU preproc enabled (opencv_cuda=%s stream_ptr=%s)", True, bool(stream_ptr))

    def _init_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("onnxruntime is required for ReID ONNX models") from exc
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.device and self.device.lower().startswith("cpu"):
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        shape = self._session.get_inputs()[0].shape
        height = int(shape[2]) if len(shape) > 2 and shape[2] is not None else 256
        width = int(shape[3]) if len(shape) > 3 and shape[3] is not None else 128
        self._input_size = (height, width)
        self.logger.info("ReID ONNX loaded: %s (input=%sx%s)", self.model_path, width, height)

    def _init_torchscript(self) -> None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required for ReID TorchScript models") from exc
        if not self.device:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model = torch.jit.load(str(self.model_path), map_location=self.device)
        self._model.eval()
        self._use_torch = True
        self.logger.info("ReID TorchScript loaded: %s (device=%s)", self.model_path, self.device)

    def _ensure_batch_buf(self, batch: int) -> np.ndarray:
        if self._batch_buf is None or self._batch_buf.shape[0] < batch:
            height, width = self._input_size
            self._batch_buf = np.empty((batch, 3, height, width), dtype=np.float32)
        return self._batch_buf[:batch]

    def _ensure_gpu_buffers(self, batch: int) -> None:
        if not self._gpu_preproc_enabled:
            return
        import cv2

        while len(self._gpu_upload) < max(1, batch):
            self._gpu_upload.append(cv2.cuda_GpuMat())
        while len(self._gpu_resize) < batch:
            self._gpu_resize.append(cv2.cuda_GpuMat())

    @staticmethod
    def _gpu_mat_info(mat) -> tuple[int, int, int]:
        size = None
        if hasattr(mat, "size"):
            try:
                size = mat.size() if callable(mat.size) else mat.size
            except Exception:
                size = None
        if size is not None and len(size) >= 2:
            cols, rows = int(size[0]), int(size[1])
        else:
            rows = getattr(mat, "rows", None)
            cols = getattr(mat, "cols", None)
            rows = rows() if callable(rows) else rows
            cols = cols() if callable(cols) else cols
            if rows is None or cols is None:
                raise AttributeError("cv2.cuda.GpuMat missing size/rows/cols")
        step = getattr(mat, "step", None)
        step = step() if callable(step) else step
        if isinstance(step, (tuple, list)):
            step = step[0]
        if step is None:
            raise AttributeError("cv2.cuda.GpuMat missing step")
        return rows, cols, int(step)

    def _ensure_gpu_batch(self, batch: int) -> None:
        if not self._gpu_preproc_enabled:
            return
        height, width = self._input_size
        shape = (batch, 3, height, width)
        if self._gpu_batch_buf is None or self._gpu_batch_shape != shape:
            import cupy as cp

            self._gpu_batch_buf = cp.empty(shape, dtype=cp.float32)
            self._gpu_batch_shape = shape

    def _preprocess_gpu(self, image: np.ndarray, boxes: np.ndarray):
        import cv2
        import cupy as cp

        height, width = self._input_size
        batch = len(boxes)
        self.logger.debug("ReID GPU preproc start (batch=%d input=%sx%s)", batch, width, height)
        self._ensure_gpu_buffers(batch)
        self._ensure_gpu_batch(batch)
        output = self._gpu_batch_buf[:batch]

        scale = 1.0 / 255.0
        ih, iw = image.shape[:2]
        gpu_src = self._gpu_upload[0]
        if self._gpu_cv_stream is not None:
            gpu_src.upload(image, self._gpu_cv_stream)
        else:
            gpu_src.upload(image)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(x1, iw - 1))
            x2 = max(0, min(x2, iw))
            y1 = max(0, min(y1, ih - 1))
            y2 = max(0, min(y2, ih))
            gpu_dst = self._gpu_resize[i]
            if x2 <= x1 or y2 <= y1:
                self.logger.debug("ReID GPU preproc skip empty crop idx=%d box=%s", i, box)
                output[i].fill(0)
                continue
            roi = gpu_src.rowRange(y1, y2).colRange(x1, x2)
            if self._gpu_cv_stream is not None:
                cv2.cuda.resize(
                    roi,
                    (width, height),
                    gpu_dst,
                    0,
                    0,
                    cv2.INTER_LINEAR,
                    stream=self._gpu_cv_stream,
                )
            else:
                cv2.cuda.resize(roi, (width, height), gpu_dst, 0, 0, cv2.INTER_LINEAR)
            mat = gpu_dst
            try:
                rows, cols, step = self._gpu_mat_info(mat)
            except Exception:
                self.logger.debug("ReID GPU preproc invalid GpuMat idx=%d box=%s", i, box)
                output[i].fill(0)
                continue
            ptr = int(mat.cudaPtr()) if hasattr(mat, "cudaPtr") else 0
            if rows <= 0 or cols <= 0 or step <= 0 or ptr == 0:
                self.logger.debug(
                    "ReID GPU preproc empty GpuMat idx=%d box=%s rows=%s cols=%s step=%s ptr=%s",
                    i,
                    box,
                    rows,
                    cols,
                    step,
                    ptr,
                )
                output[i].fill(0)
                continue
            mem = cp.cuda.UnownedMemory(ptr, int(step * rows), mat)
            memptr = cp.cuda.MemoryPointer(mem, 0)
            g_bgr = cp.ndarray((rows, cols, 3), dtype=cp.uint8, memptr=memptr, strides=(step, 3, 1))
            g_rgb = g_bgr[..., ::-1]
            g_rgb = cp.transpose(g_rgb, (2, 0, 1))
            with self._gpu_stream:
                output[i] = (g_rgb.astype(cp.float32) * scale - self._mean_gpu) / self._std_gpu

        return output

    def _preprocess(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("cv2 is required for ReID preprocessing") from exc
        height, width = self._input_size
        batch = self._ensure_batch_buf(len(boxes))
        ih, iw = image.shape[:2]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(x1, iw - 1))
            x2 = max(0, min(x2, iw))
            y1 = max(0, min(y1, ih - 1))
            y2 = max(0, min(y2, ih))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                batch[i].fill(0)
                continue
            crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.astype(np.float32) / 255.0
            crop = np.transpose(crop, (2, 0, 1))
            crop = (crop - self._mean) / self._std
            batch[i] = crop
        return batch

    def extract(self, image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        if image is None or boxes.size == 0:
            return []
        feats: List[np.ndarray] = []
        for start in range(0, len(boxes), self.batch_size):
            chunk = boxes[start : start + self.batch_size]
            self.logger.debug("ReID extract chunk start=%d size=%d", start, len(chunk))
            if self._trt is not None and self._gpu_preproc_enabled:
                try:
                    gpu_batch = self._preprocess_gpu(image, chunk)
                    output = self._trt.infer_device(gpu_batch)
                except Exception as exc:
                    self.logger.exception("ReID GPU preproc failed; falling back to CPU")
                    self._gpu_preproc_enabled = False
                    batch = self._preprocess(image, chunk)
                    output = self._trt.infer(batch)
            else:
                batch = self._preprocess(image, chunk)
                if self._trt is not None:
                    output = self._trt.infer(batch)
                elif self._use_torch:
                    import torch

                    with torch.no_grad():
                        tensor = torch.from_numpy(batch).to(self.device)
                        output = self._model(tensor)
                        if isinstance(output, (tuple, list)):
                            output = output[0]
                        output = output.detach().float().cpu().numpy()
                else:
                    output = self._session.run([self._output_name], {self._input_name: batch})[0]
            if output.ndim > 2:
                output = output.reshape((output.shape[0], -1))
            norm = np.linalg.norm(output, axis=1, keepdims=True) + 1e-6
            output = output / norm
            for row in output:
                feats.append(row.astype(np.float32))
        return feats


class FeatureExtractor:
    """Selects between ReID embeddings and lightweight color histogram features."""

    def __init__(self, reid: Optional[ReIDEncoder]) -> None:
        self.reid = reid

    def extract(self, image: np.ndarray, boxes: np.ndarray) -> Optional[List[np.ndarray]]:
        if image is None or boxes is None:
            return None
        if self.reid is not None:
            return self.reid.extract(image, boxes)
        return _extract_color_features(image, boxes)


class _ReIDTensorRT:
    def __init__(self, engine, device_id: int) -> None:
        try:
            import tensorrt as trt
            import cupy as cp
            import cupyx
        except Exception as exc:
            raise RuntimeError("TensorRT and CuPy are required for ReID TRT inference") from exc

        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.is_trt10 = not hasattr(engine, "num_bindings")
        self.input_name = None
        self.output_names: List[str] = []
        self.output_dtypes: List[np.dtype] = []
        self.input_dtype = None
        self.input_shape = None
        self.max_batch_size = 1

        class HostDeviceMem:
            def __init__(self, size, dtype):
                if size <= 0:
                    raise RuntimeError("Invalid TensorRT buffer size")
                self.size = size
                self.dtype = dtype
                self.host = cupyx.zeros_pinned(size, dtype)
                self.device = cp.zeros(size, dtype)

            @property
            def devptr(self):
                return self.device.data.ptr

            def copy_dtoh_async(self, stream):
                self.device.data.copy_to_host_async(self.host.ctypes.data, self.host.nbytes, stream)

        self._HostDeviceMem = HostDeviceMem
        self.input = None
        self.outputs: List[HostDeviceMem] = []
        self.bindings: List[int] = []
        self.input_index = None
        self.output_indices: List[int] = []

        if self.is_trt10:
            for idx in range(engine.num_io_tensors):
                name = engine.get_tensor_name(idx)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                mode = engine.get_tensor_mode(name)
                shape = tuple(int(v) for v in engine.get_tensor_shape(name))
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                    if -1 in shape:
                        shape = tuple(int(v) for v in engine.get_tensor_profile_shape(name, 0)[2])
                    self.input_shape = shape
                    self.max_batch_size = shape[0]
                    self.input_dtype = dtype
                else:
                    self.output_names.append(name)
                    self.output_dtypes.append(dtype)

            if not self.input_name or self.input_shape is None:
                raise RuntimeError("TensorRT ReID engine has no valid input tensor")

            self.context.set_input_shape(self.input_name, self.input_shape)
            self.input = HostDeviceMem(int(np.prod(self.input_shape)), self.input_dtype)
            self.bindings.append(self.input.devptr)
            for idx, name in enumerate(self.output_names):
                shape = tuple(int(v) for v in self.context.get_tensor_shape(name))
                if any(dim <= 0 for dim in shape):
                    shape = tuple(int(v) for v in engine.get_tensor_profile_shape(name, 0)[2])
                self.outputs.append(HostDeviceMem(int(np.prod(shape)), self.output_dtypes[idx]))
                self.bindings.append(self.outputs[-1].devptr)
        else:
            for idx in range(engine.num_bindings):
                name = engine.get_binding_name(idx)
                dtype = trt.nptype(engine.get_binding_dtype(idx))
                shape = tuple(int(v) for v in engine.get_binding_shape(idx))
                if engine.binding_is_input(idx):
                    self.input_index = idx
                    if -1 in shape:
                        shape = tuple(int(v) for v in engine.get_profile_shape(0, idx)[2])
                    self.input_shape = shape
                    self.max_batch_size = shape[0]
                    self.input_dtype = dtype
                else:
                    self.output_indices.append(idx)
                    self.output_dtypes.append(dtype)

            if self.input_index is None or self.input_shape is None:
                raise RuntimeError("TensorRT ReID engine has no valid input binding")

            self.context.set_binding_shape(self.input_index, self.input_shape)
            self.input = HostDeviceMem(int(np.prod(self.input_shape)), self.input_dtype)
            self.bindings = [self.input.devptr]
            for idx, binding in enumerate(self.output_indices):
                shape = tuple(int(v) for v in self.context.get_binding_shape(binding))
                if any(dim <= 0 for dim in shape):
                    shape = tuple(int(v) for v in engine.get_profile_shape(0, binding)[2])
                self.outputs.append(HostDeviceMem(int(np.prod(shape)), self.output_dtypes[idx]))
                self.bindings.append(self.outputs[-1].devptr)

        self.input_hw = (int(self.input_shape[2]), int(self.input_shape[3]))

    def infer(self, batch: np.ndarray) -> np.ndarray:
        import cupy as cp

        if self.input is None:
            raise RuntimeError("TensorRT ReID engine is not initialized")
        if batch.ndim != 4:
            raise ValueError(f"Invalid ReID input shape {batch.shape}")
        batch_size = int(batch.shape[0])
        if tuple(batch.shape[1:]) != tuple(self.input_shape[1:]):
            raise ValueError(
                f"ReID input shape {tuple(batch.shape[1:])} does not match engine {tuple(self.input_shape[1:])}"
            )
        if batch_size > self.max_batch_size:
            raise RuntimeError(f"ReID batch size {batch_size} exceeds engine max {self.max_batch_size}")

        input_shape = (batch_size, *self.input_shape[1:])
        if self.is_trt10:
            self.context.set_input_shape(self.input_name, input_shape)
        else:
            self.context.set_binding_shape(self.input_index, input_shape)

        with self.stream:
            input_view = self.input.device[: batch.size].reshape(batch.shape)
            if batch.dtype != input_view.dtype:
                batch = batch.astype(input_view.dtype, copy=False)
            cp.copyto(input_view, batch)

        if self.is_trt10:
            for idx, name in enumerate(self.output_names):
                self.context.set_tensor_address(name, self.outputs[idx].devptr)
            self.context.set_tensor_address(self.input_name, self.input.devptr)
            self.context.execute_async_v3(stream_handle=self.stream.ptr)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.ptr)

        for out in self.outputs:
            out.copy_dtoh_async(self.stream)
        self.stream.synchronize()
        if self.is_trt10:
            output_shape = tuple(int(v) for v in self.context.get_tensor_shape(self.output_names[0]))
        else:
            output_shape = tuple(int(v) for v in self.context.get_binding_shape(self.output_indices[0]))
        size = int(np.prod(output_shape))
        return self.outputs[0].host[:size].reshape(output_shape).copy()

    def infer_device(self, batch) -> np.ndarray:
        import cupy as cp

        if self.input is None:
            raise RuntimeError("TensorRT ReID engine is not initialized")
        if batch.ndim != 4:
            raise ValueError(f"Invalid ReID input shape {batch.shape}")
        batch_size = int(batch.shape[0])
        if tuple(batch.shape[1:]) != tuple(self.input_shape[1:]):
            raise ValueError(
                f"ReID input shape {tuple(batch.shape[1:])} does not match engine {tuple(self.input_shape[1:])}"
            )
        if batch_size > self.max_batch_size:
            raise RuntimeError(f"ReID batch size {batch_size} exceeds engine max {self.max_batch_size}")

        input_shape = (batch_size, *self.input_shape[1:])
        if self.is_trt10:
            self.context.set_input_shape(self.input_name, input_shape)
        else:
            self.context.set_binding_shape(self.input_index, input_shape)

        with self.stream:
            input_view = self.input.device[: batch.size].reshape(batch.shape)
            if batch.dtype != input_view.dtype:
                batch = batch.astype(input_view.dtype, copy=False)
            cp.copyto(input_view, batch)

        if self.is_trt10:
            for idx, name in enumerate(self.output_names):
                self.context.set_tensor_address(name, self.outputs[idx].devptr)
            self.context.set_tensor_address(self.input_name, self.input.devptr)
            self.context.execute_async_v3(stream_handle=self.stream.ptr)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.ptr)

        for out in self.outputs:
            out.copy_dtoh_async(self.stream)
        self.stream.synchronize()
        if self.is_trt10:
            output_shape = tuple(int(v) for v in self.context.get_tensor_shape(self.output_names[0]))
        else:
            output_shape = tuple(int(v) for v in self.context.get_binding_shape(self.output_indices[0]))
        size = int(np.prod(output_shape))
        return self.outputs[0].host[:size].reshape(output_shape).copy()


class ByteTracker:
    def __init__(self, config: TrackerConfig, feature_extractor: Optional[FeatureExtractor] = None):
        self.config = config
        self.tracked: List[Track] = []
        self.lost: List[Track] = []
        self.removed: List[Track] = []
        self.frame_id = 0
        self.max_time_lost = int(config.track_buffer)
        self.kalman_filter = KalmanFilterXYWH()
        self.feature_extractor = feature_extractor
        self.use_numba = _numba_enabled(config)
        self.use_hungarian = str(config.matching).lower() == "hungarian"
        BaseTrack.reset_id()

    def reset(self) -> None:
        self.tracked = []
        self.lost = []
        self.removed = []
        self.frame_id = 0
        BaseTrack.reset_id()

    def update(self, detections: Detections, image: Optional[np.ndarray] = None) -> np.ndarray:
        self.frame_id += 1
        activated = []
        refind = []
        lost = []
        removed = []

        scores = detections.conf
        remain_inds = scores >= self.config.track_high_thresh
        inds_low = scores > self.config.track_low_thresh
        inds_high = scores < self.config.track_high_thresh
        inds_second = inds_low & inds_high
        detections_second = detections[inds_second]
        detections = detections[remain_inds]

        det_tracks = self._init_tracks(detections, image)
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self._joint_tracks(tracked_stracks, self.lost)
        self._multi_predict(strack_pool)
        dists = self._get_dists(strack_pool, det_tracks)
        matches, u_track, u_detection = _linear_assignment(
            dists,
            self.config.match_thresh,
            use_numba=self.use_numba,
            use_hungarian=self.use_hungarian,
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = det_tracks[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        det_tracks_second = self._init_tracks(detections_second, image)
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = _iou_distance(r_tracked, det_tracks_second, use_numba=self.use_numba)
        matches, u_track, _ = _linear_assignment(
            dists,
            0.5,
            use_numba=self.use_numba,
            use_hungarian=self.use_hungarian,
        )
        for itracked, idet in matches:
            track = r_tracked[itracked]
            det = det_tracks_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        for it in u_track:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        detections_unmatched = [det_tracks[i] for i in u_detection]
        dists = _iou_distance(unconfirmed, detections_unmatched, use_numba=self.use_numba)
        matches, u_unconfirmed, u_detection = _linear_assignment(
            dists,
            0.7,
            use_numba=self.use_numba,
            use_hungarian=self.use_hungarian,
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_unmatched[idet], self.frame_id)
            activated.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed.append(track)

        for inew in u_detection:
            track = detections_unmatched[inew]
            if track.score < self.config.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        for track in self.lost:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        self.tracked = self._joint_tracks(self.tracked, activated)
        self.tracked = self._joint_tracks(self.tracked, refind)
        self.lost = self._sub_tracks(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = self._sub_tracks(self.lost, self.removed)
        self.tracked, self.lost = self._remove_duplicates(self.tracked, self.lost, use_numba=self.use_numba)
        self.removed.extend(removed)
        if len(self.removed) > 1000:
            self.removed = self.removed[-1000:]

        return np.asarray([t.result for t in self.tracked if t.is_activated], dtype=np.float32)

    def _init_tracks(self, detections: Detections, image: Optional[np.ndarray]) -> List[Track]:
        if len(detections) == 0:
            return []
        feats = None
        if image is not None and self.feature_extractor is not None:
            feats = self.feature_extractor.extract(image, detections.xyxy)
        tracks = []
        for xywh, score, cls, idx in zip(detections.xywh, detections.conf, detections.cls, detections.idx):
            feat = feats.pop(0) if feats is not None else None
            tracks.append(Track(xywh, float(score), int(cls), int(idx), feat=feat))
        return tracks

    def _get_dists(self, tracks: List[Track], detections: List[Track]) -> np.ndarray:
        dists = _iou_distance(tracks, detections, use_numba=self.use_numba)
        if self.config.fuse_score:
            dists = _fuse_score(dists, detections)
        return dists

    def _multi_predict(self, tracks: List[Track]) -> None:
        for t in tracks:
            t.predict()

    @staticmethod
    def _joint_tracks(a: List[Track], b: List[Track]) -> List[Track]:
        exists = {}
        res = []
        for t in a:
            exists[t.track_id] = True
            res.append(t)
        for t in b:
            if not exists.get(t.track_id, False):
                exists[t.track_id] = True
                res.append(t)
        return res

    @staticmethod
    def _sub_tracks(a: List[Track], b: List[Track]) -> List[Track]:
        track_ids = {t.track_id for t in b}
        return [t for t in a if t.track_id not in track_ids]

    @staticmethod
    def _remove_duplicates(a: List[Track], b: List[Track], use_numba: bool = False) -> Tuple[List[Track], List[Track]]:
        pdist = _iou_distance(a, b, use_numba=use_numba)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = a[p].frame_id - a[p].start_frame
            timeq = b[q].frame_id - b[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(a) if i not in dupa]
        resb = [t for i, t in enumerate(b) if i not in dupb]
        return resa, resb


class BoTSORT(ByteTracker):
    def _get_dists(self, tracks: List[Track], detections: List[Track]) -> np.ndarray:
        dists = _iou_distance(tracks, detections, use_numba=self.use_numba)
        if not tracks or not detections:
            return dists
        track_feats = np.stack([t.smooth_feat for t in tracks if t.smooth_feat is not None], axis=0)
        det_feats = np.stack([d.curr_feat for d in detections if d.curr_feat is not None], axis=0)
        if track_feats.size and det_feats.size:
            emb_dists = _cosine_distance(track_feats, det_feats)
            iou_mask = dists > (1.0 - self.config.proximity_thresh)
            emb_dists[emb_dists > (1.0 - self.config.appearance_thresh)] = 1.0
            emb_dists[iou_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        if self.config.fuse_score:
            dists = _fuse_score(dists, detections)
        return dists


class DeepSORT:
    def __init__(self, config: TrackerConfig, feature_extractor: Optional[FeatureExtractor] = None):
        self.config = config
        self.tracks: List[Track] = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYWH()
        self.feature_extractor = feature_extractor
        self.use_numba = _numba_enabled(config)
        self.use_hungarian = str(config.matching).lower() == "hungarian"
        BaseTrack.reset_id()

    def reset(self) -> None:
        self.tracks = []
        self.frame_id = 0
        BaseTrack.reset_id()

    def update(self, detections: Detections, image: Optional[np.ndarray]) -> np.ndarray:
        self.frame_id += 1
        det_tracks = self._init_tracks(detections, image)
        for track in self.tracks:
            track.predict()

        confirmed = [t for t in self.tracks if t.is_activated]
        dists = self._appearance_distance(confirmed, det_tracks)
        matches, u_track, u_detection = _linear_assignment(
            dists,
            1.0 - self.config.appearance_thresh,
            use_numba=self.use_numba,
            use_hungarian=self.use_hungarian,
        )
        for itracked, idet in matches:
            confirmed[itracked].update(det_tracks[idet], self.frame_id)

        unmatched_tracks = [confirmed[i] for i in u_track]
        remaining_dets = [det_tracks[i] for i in u_detection]
        dists = _iou_distance(unmatched_tracks, remaining_dets, use_numba=self.use_numba)
        matches, u_track, u_detection = _linear_assignment(
            dists,
            1.0 - self.config.iou_thresh,
            use_numba=self.use_numba,
            use_hungarian=self.use_hungarian,
        )
        for itracked, idet in matches:
            unmatched_tracks[itracked].update(remaining_dets[idet], self.frame_id)

        for i in u_track:
            track = unmatched_tracks[i]
            track.time_since_update += 1
            if track.time_since_update > self.config.max_age:
                track.mark_removed()

        for i in u_detection:
            track = remaining_dets[i]
            track.activate(self.kalman_filter, self.frame_id)
            if track.hits >= self.config.min_hits:
                track.is_activated = True
            self.tracks.append(track)

        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        return np.asarray([t.result for t in self.tracks if t.is_activated], dtype=np.float32)

    def _init_tracks(self, detections: Detections, image: Optional[np.ndarray]) -> List[Track]:
        if len(detections) == 0:
            return []
        feats = None
        if image is not None and self.feature_extractor is not None:
            feats = self.feature_extractor.extract(image, detections.xyxy)
        tracks = []
        for xywh, score, cls, idx in zip(detections.xywh, detections.conf, detections.cls, detections.idx):
            feat = feats.pop(0) if feats is not None else None
            tracks.append(Track(xywh, float(score), int(cls), int(idx), feat=feat))
        return tracks

    @staticmethod
    def _appearance_distance(tracks: List[Track], detections: List[Track]) -> np.ndarray:
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)
        track_feats = np.stack([t.smooth_feat for t in tracks], axis=0)
        det_feats = np.stack([d.curr_feat for d in detections], axis=0)
        return _cosine_distance(track_feats, det_feats)


class TrackerAdapter:
    def reset(self) -> None:
        raise NotImplementedError

    def update(self, detections: List[Detection], image: Optional[np.ndarray]) -> Dict[int, int]:
        raise NotImplementedError


class TrackerManager(TrackerAdapter):
    def __init__(self, tracker_type: str, config: Optional[TrackerConfig] = None) -> None:
        tracker_type = (tracker_type or "bytetrack").strip().lower()
        self.config = config or TrackerConfig()
        self._history_len = max(1, int(self.config.track_buffer))
        self._frame_id = 0
        self._track_history: Dict[int, deque] = {}
        self._track_last_seen: Dict[int, int] = {}
        self.reid_encoder = None
        if self.config.reid_model:
            resolved = os.path.expanduser(self.config.reid_model)
            if os.path.exists(resolved):
                self.reid_encoder = ReIDEncoder(
                    resolved,
                    device=self.config.reid_device,
                    batch_size=self.config.reid_batch_size,
                )
            else:
                raise FileNotFoundError(f"ReID model not found: {resolved}")
        feature_extractor = FeatureExtractor(self.reid_encoder) if tracker_type in {"botsort", "deepsort"} else None
        if tracker_type == "botsort":
            self.tracker = BoTSORT(self.config, feature_extractor=feature_extractor)
        elif tracker_type == "deepsort":
            self.tracker = DeepSORT(self.config, feature_extractor=feature_extractor)
        else:
            self.tracker = ByteTracker(self.config, feature_extractor=None)
        self.logger = get_logger("tracking")
        self.logger.info("Tracking enabled: %s", tracker_type)
        self.logger.info("Tracking accel: numba=%s reid=%s", _numba_enabled(self.config), bool(self.reid_encoder))
        self._xyxy_buf = np.empty((0, 4), dtype=np.float32)
        self._conf_buf = np.empty((0,), dtype=np.float32)
        self._cls_buf = np.empty((0,), dtype=np.float32)
        self._idx_buf = np.empty((0,), dtype=np.int32)

    def reset(self) -> None:
        self.tracker.reset()
        self._frame_id = 0
        self._track_history.clear()
        self._track_last_seen.clear()

    def history(self) -> Dict[int, List[Tuple[int, int]]]:
        return {track_id: list(points) for track_id, points in self._track_history.items()}

    def _update_history(self, tracks: np.ndarray) -> None:
        for row in tracks:
            row_size = row.size if hasattr(row, "size") else len(row)
            if row_size < 8:
                continue
            track_id = int(row[-4])
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            cx = int(round((x1 + x2) * 0.5))
            cy = int(round((y1 + y2) * 0.5))
            history = self._track_history.get(track_id)
            if history is None:
                history = deque(maxlen=self._history_len)
                self._track_history[track_id] = history
            history.append((cx, cy))
            self._track_last_seen[track_id] = self._frame_id

    def _prune_history(self) -> None:
        if not self._track_last_seen:
            return
        expire_after = self._frame_id - self._history_len
        if expire_after <= 0:
            return
        stale = [track_id for track_id, last in self._track_last_seen.items() if last <= expire_after]
        for track_id in stale:
            self._track_last_seen.pop(track_id, None)
            self._track_history.pop(track_id, None)

    def update(self, detections: List[Detection], image: Optional[np.ndarray]) -> Dict[int, int]:
        self._frame_id += 1
        if not detections:
            self._prune_history()
            return {}
        count = len(detections)
        if self._xyxy_buf.shape[0] < count:
            new_size = max(count, int(self._xyxy_buf.shape[0] * 1.5) + 1)
            self._xyxy_buf = np.empty((new_size, 4), dtype=np.float32)
            self._conf_buf = np.empty((new_size,), dtype=np.float32)
            self._cls_buf = np.empty((new_size,), dtype=np.float32)
            self._idx_buf = np.empty((new_size,), dtype=np.int32)
        xyxy = self._xyxy_buf[:count]
        conf = self._conf_buf[:count]
        cls = self._cls_buf[:count]
        idx = self._idx_buf[:count]
        for i, det in enumerate(detections):
            bbox = det.bbox
            xyxy[i, 0] = bbox[0]
            xyxy[i, 1] = bbox[1]
            xyxy[i, 2] = bbox[2]
            xyxy[i, 3] = bbox[3]
            conf[i] = det.score
            cls[i] = det.class_id
            idx[i] = i
        results = Detections(xyxy, conf, cls, idx, use_numba=self.config.use_numba)
        tracks = self.tracker.update(results, image)
        self._update_history(tracks)
        self._prune_history()
        track_ids: Dict[int, int] = {}
        for row in tracks:
            row_size = row.size if hasattr(row, "size") else len(row)
            if row_size < 8:
                continue
            det_index = int(row[-1])
            track_id = int(row[-4])
            track_ids[det_index] = track_id
        return track_ids


def create_tracker(tracker_type: str, config: Optional[TrackerConfig] = None) -> TrackerManager:
    return TrackerManager(tracker_type, config=config)
