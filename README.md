# AI Video Analytics System (Person Detection)

a production-ready REST API for person detection. This repo mirrors the InsightFace-REST
architecture, but the pipeline is focused on people only. API responses return
a `people` array (each entry includes bbox, prob, size, class_id, class_name).

## Key Features

- FastAPI REST API for person detection.
- Ultralytics YOLO models with optional ONNX/TensorRT/OpenVINO export.
- Batch inference support.
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
4. Open API docs at `http://localhost:18081/docs`.

See `compose/README.md` for the full matrix of compose options.

## API Endpoints (v1)

- `POST /v1/detect` - JSON inference, returns `people`.
- `POST /v1/draw_detections` or `POST /v1/draw` - draw detections on images.
- `POST /v1/multipart/draw_detections` or `POST /v1/multipart/draw` - multipart file upload.
- `GET /v1/health` - service health.

## Response Format

`people` is a list of detections, each with:
`bbox`, `prob`, `size`, `class_id`, `class_name`.
If enabled, `persondata` includes base64-encoded crops.

## Clients

Use `ai_video_analytics_clients` for sync/async Python clients and helpers.

## References

This project is based on the architecture of:
- https://github.com/SthPhoenix/InsightFace-REST
- https://github.com/deepinsight/insightface
