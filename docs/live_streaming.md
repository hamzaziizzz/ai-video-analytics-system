# Live Streaming

This system can expose a live MJPEG feed and snapshot per camera when streaming is enabled in config.

## Config

Add or update the `streaming` block:

```
"streaming": {
  "enabled": true,
  "draw_boxes": true,
  "draw_all_detections": false,
  "jpeg_quality": 80,
  "fps_limit": 5
}
```

## Endpoints

- Snapshot (latest frame):
  - `GET /snapshot/{camera_id}`
- MJPEG stream:
  - `GET /stream/{camera_id}`
- Simple preview page:
  - `GET /preview/{camera_id}`

Example:

```
http://localhost:8000/stream/cam_entrance
```
