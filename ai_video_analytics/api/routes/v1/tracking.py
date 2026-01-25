import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.responses import StreamingResponse

from ai_video_analytics.core.processing import ProcessingDep
from ai_video_analytics.core.utils.logging import get_logger
from ai_video_analytics.schemas import TrackingStream


router = APIRouter()
logger = get_logger("api.tracking")


def _overlay_fps(frame, fps: float) -> None:
    height, width = frame.shape[:2]
    line_width = max(int(round((height + width) / 2 * 0.003)), 2)
    font_scale = max(line_width / 3, 0.6)
    font_thickness = max(line_width - 1, 1)
    label = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    x = max(10, line_width)
    y = max(th + line_width, 10 + th)
    x = min(x, max(width - tw - line_width, x))
    y = min(y, max(height - line_width, y))
    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 0),
        font_thickness,
        lineType=cv2.LINE_AA,
    )


def _resize_stream_frame(frame: cv2.Mat, target: int = 640) -> cv2.Mat:
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame
    if width >= height:
        target_w = target
        target_h = int(round(height * (target / float(width))))
    else:
        target_h = target
        target_w = int(round(width * (target / float(height))))
    target_w = max(1, target_w)
    target_h = max(1, target_h)
    if target_w == width and target_h == height:
        return frame
    interp = cv2.INTER_AREA if target_w < width or target_h < height else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_w, target_h), interpolation=interp)


def _parse_source(source: Union[str, int]) -> Union[str, int]:
    if isinstance(source, int):
        return source
    value = str(source or "").strip()
    if value.isdigit():
        return int(value)
    return value


def _open_capture(source: Union[str, int]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    return cap


def _build_tracker(processing, pose: bool):
    if not processing or not getattr(processing, "tracking_enabled", False):
        return None
    config = processing.pose_config if pose and getattr(processing, "pose_config", None) else processing.config
    if not config:
        return None
    from ai_video_analytics.core.tracking import TrackerConfig, create_tracker

    tracker_type = getattr(config.features, "tracker_type", "bytetrack")
    tracker_config = TrackerConfig(
        use_numba=getattr(config.features, "tracker_numba", True),
        reid_model=getattr(config.features, "tracker_reid_model", None),
        reid_batch_size=getattr(config.features, "tracker_reid_batch_size", 32),
        reid_device=getattr(config.features, "tracker_reid_device", None),
        matching=getattr(config.features, "tracker_matching", "greedy"),
    )
    return create_tracker(tracker_type, tracker_config)


def _frame_stream(
    capture: cv2.VideoCapture,
    processing,
    pose: bool,
    threshold: float,
    draw_scores: bool,
    draw_sizes: bool,
    limit_people: int,
    min_person_size: int,
    jpeg_quality: int,
    fps_limit: float,
    show_fps: bool,
):
    engine = processing.pose_engine if pose else processing.engine
    if engine is None:
        raise RuntimeError("Inference engine is not initialized")
    tracker = _build_tracker(processing, pose)
    if tracker:
        tracker.reset()
    frame_interval = 1.0 / fps_limit if fps_limit and fps_limit > 0 else 0.0
    last_t = time.time()
    fps = 0.0
    try:
        while True:
            t0 = time.time()
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame = _resize_stream_frame(frame, target=640)
            dets = engine.infer_batch([frame])[0]
            if tracker:
                track_ids = tracker.update(dets, frame)
                for det_index, track_id in track_ids.items():
                    if 0 <= det_index < len(dets):
                        dets[det_index].track_id = track_id
                tracks = processing._serialize_tracks(tracker.history())
                if tracks:
                    frame = processing._draw_track_trails(frame, tracks)
            filtered = processing._filter_detections(dets, threshold, limit_people, min_person_size)
            if pose:
                frame = processing._draw_pose_detections(frame, filtered, draw_scores=draw_scores, draw_sizes=draw_sizes)
            else:
                frame = processing._draw_detections(frame, filtered, draw_scores=draw_scores, draw_sizes=draw_sizes)
            if show_fps:
                now = time.time()
                dt = now - last_t
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
                last_t = now
                _overlay_fps(frame, fps)
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            if frame_interval > 0:
                elapsed = time.time() - t0
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
    finally:
        capture.release()


@router.post("/tracking/stream", tags=["Tracking"])
async def stream_tracking(data: TrackingStream, processing: ProcessingDep):
    """
    Stream annotated detections with tracking overlays as MJPEG.

       - **source**: Camera index, file path, HTTP/RTSP URL (*required*)
       - **pose**: Use pose detection instead of person detection. Default: False (*optional*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **draw_sizes**: Draw detection sizes Default: True (*optional*)
       - **limit_people**: Maximum number of detections to be processed.  0 for unlimited number. Default: 0 (*optional*)
       - **fps_limit**: Output FPS limit. 0 disables throttling. Default: 0 (*optional*)
        - **jpeg_quality**: JPEG quality 1-100. Default: 80 (*optional*)
       - **show_fps**: Overlay FPS on frames. Default: True (*optional*)
       \f
    """
    try:
        source = _parse_source(data.source)
        capture = _open_capture(source)
        quality = int(max(1, min(100, data.jpeg_quality or 80)))
        stream = _frame_stream(
            capture,
            processing,
            pose=bool(data.pose),
            threshold=float(data.threshold or 0.6),
            draw_scores=bool(data.draw_scores),
            draw_sizes=bool(data.draw_sizes),
            limit_people=int(data.limit_people or 0),
            min_person_size=int(data.min_person_size or 0),
            jpeg_quality=quality,
            fps_limit=float(data.fps_limit or 0.0),
            show_fps=bool(data.show_fps),
        )
        return StreamingResponse(stream, media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as exc:
        logger.exception("Tracking stream failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/tracking/stream/upload", tags=["Tracking"])
async def stream_tracking_upload(
    processing: ProcessingDep,
    file: UploadFile = File(...),
    pose: bool = Form(False),
    threshold: float = Form(0.6),
    draw_scores: bool = Form(True),
    draw_sizes: bool = Form(True),
    limit_people: int = Form(0),
    min_person_size: int = Form(0),
    fps_limit: float = Form(0.0),
    jpeg_quality: int = Form(80),
    show_fps: bool = Form(True),
):
    """
    Stream annotated detections for an uploaded video file as MJPEG.
    """
    tmp_path: Optional[Path] = None
    try:
        payload = await file.read()
        suffix = Path(file.filename or "").suffix or ".mp4"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(payload)
        tmp_file.flush()
        tmp_file.close()
        tmp_path = Path(tmp_file.name)
        capture = _open_capture(str(tmp_path))
        quality = int(max(1, min(100, jpeg_quality)))
        stream = _frame_stream(
            capture,
            processing,
            pose=bool(pose),
            threshold=float(threshold),
            draw_scores=bool(draw_scores),
            draw_sizes=bool(draw_sizes),
            limit_people=int(limit_people),
            min_person_size=int(min_person_size),
            jpeg_quality=quality,
            fps_limit=float(fps_limit),
            show_fps=bool(show_fps),
        )
        return StreamingResponse(stream, media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as exc:
        logger.exception("Tracking upload stream failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                logger.warning("Failed to remove temp video: %s", tmp_path)
