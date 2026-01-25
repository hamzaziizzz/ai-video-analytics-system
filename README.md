# AI Video Analytics System (Detection, Pose, Tracking)

A production-ready REST API for person detection with optional pose and tracking.
The pipeline mirrors the InsightFace-REST architecture, tuned for people analytics.
API responses return `people` (and `poses` when enabled) with optional tracking metadata.

## Key Features

- FastAPI REST API for person detection, pose, and tracking.
- Custom TensorRT and ONNX runtimes plus Ultralytics fallback.
- Batch inference and GPU preproc/postproc options.
- Optional ReID for BoT-SORT/DeepSORT tracking.
- MJPEG tracking streams for webcam, file, or HTTP/RTSP sources.
- Docker compose configs for CPU, GPU, and multi-GPU deployments.
- Optional msgpack responses for faster payload transfer.

## Quick Start (Docker)

1. Create a `.env` under `compose/` (see `compose/cpu.env` as a starting point).
2. Use a compose file that matches your target:
   - `compose/docker-compose.yml` (single GPU)
   - `compose/docker-compose-cpu.yml` (CPU)
   - `compose/docker-compose-multi-gpu.yml` (multi-GPU)
3. Run:
   ```bash
   docker compose -f compose/docker-compose.yml up --build
   ```
4. Open API docs at `http://localhost:18080/docs`.

See `compose/README.md` for the full matrix of compose options.

## API Endpoints (v1)

- `POST /v1/detect` - JSON inference, returns `people`.
- `POST /v1/pose/detect` - JSON pose inference, returns `poses`.
- `POST /v1/draw` - draw detections on images.
- `POST /v1/pose/draw` - draw pose detections on images.
- `POST /v1/multipart/draw` - multipart file upload (detection).
- `POST /v1/multipart/pose/draw` - multipart file upload (pose).
- `POST /v1/tracking/stream` - MJPEG tracking stream from webcam/file/URL.
- `POST /v1/tracking/stream/upload` - MJPEG tracking stream from uploaded video.
- `GET /v1/health` - service health.

## Response Format

`people` is a list of detections, each with:
`bbox`, `prob`, `class_id`, `class_name`, `num_det` (0-based index), and optional `track_id`.
If enabled, `persondata` includes base64-encoded crops.

`poses` entries include `bbox`, `prob`, `keypoints`, `class_id`, `class_name`,
`num_det` (0-based index), and optional `track_id`.

When tracking is enabled, each image response includes `tracks` with per-ID
track history (list of points). Tracking streams draw trail overlays.

## Clients

Use `ai_video_analytics_clients` for sync/async Python clients and helpers.

## References

This project is based on the architecture of:
- https://github.com/SthPhoenix/InsightFace-REST
- https://github.com/deepinsight/insightface
