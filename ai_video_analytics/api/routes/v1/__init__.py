from fastapi import APIRouter

from ai_video_analytics.api.routes.v1.detection import router as detection_router
from ai_video_analytics.api.routes.v1.service import router as service_router

v1_router = APIRouter()
v1_router.include_router(detection_router)
v1_router.include_router(service_router)
