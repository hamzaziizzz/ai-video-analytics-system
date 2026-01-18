from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from ..service import VideoAnalyticsService
from ..utils.config import AppConfig
from ..utils.logging import get_logger, setup_logging


def create_app(config: AppConfig) -> FastAPI:
    setup_logging(config.app.log_level)
    logger = get_logger("api")

    app = FastAPI(title="AI Video Analytics System")
    service = VideoAnalyticsService(config)
    app.state.service = service

    @app.on_event("startup")
    def on_startup() -> None:
        logger.info("API startup")
        service.start()

    @app.on_event("shutdown")
    def on_shutdown() -> None:
        logger.info("API shutdown")
        service.stop()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/events")
    def events(limit: int = Query(100, ge=1, le=1000)) -> dict:
        return {"events": service.get_events(limit=limit)}

    @app.get("/cameras")
    def cameras() -> dict:
        return {"cameras": service.get_camera_status()}

    @app.get("/snapshot/{camera_id}")
    def snapshot(camera_id: str):
        if camera_id not in service.get_camera_status():
            raise HTTPException(status_code=404, detail="Camera not found")
        if not service.streaming_enabled():
            raise HTTPException(status_code=404, detail="Streaming is disabled")
        frame = service.get_latest_frame(camera_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="No frame available yet")
        return Response(content=frame, media_type="image/jpeg")

    @app.get("/stream/{camera_id}")
    def stream(camera_id: str):
        if camera_id not in service.get_camera_status():
            raise HTTPException(status_code=404, detail="Camera not found")
        if not service.streaming_enabled():
            raise HTTPException(status_code=404, detail="Streaming is disabled")
        return StreamingResponse(
            service.stream_frames(camera_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/preview/{camera_id}")
    def preview(camera_id: str):
        if camera_id not in service.get_camera_status():
            raise HTTPException(status_code=404, detail="Camera not found")
        if not service.streaming_enabled():
            raise HTTPException(status_code=404, detail="Streaming is disabled")
        html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Live Preview - {camera_id}</title>
    <style>
      body {{ margin: 0; background: #0b0c0f; color: #f0f0f0; font-family: Arial, sans-serif; }}
      header {{ padding: 12px 16px; background: #141722; }}
      .frame {{ display: flex; justify-content: center; padding: 16px; }}
      img {{ max-width: 100%; border: 1px solid #2b2f3a; }}
    </style>
  </head>
  <body>
    <header>Camera: {camera_id}</header>
    <div class="frame">
      <img src="/stream/{camera_id}" alt="Live stream" />
    </div>
  </body>
</html>"""
        return HTMLResponse(content=html)

    return app
