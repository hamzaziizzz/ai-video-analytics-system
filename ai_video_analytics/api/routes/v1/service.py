import os

from fastapi import APIRouter, HTTPException
from starlette.responses import RedirectResponse

from ai_video_analytics.schemas import Images
from ai_video_analytics.settings import Settings
from ai_video_analytics.core.processing import ProcessingDep

settings = Settings()
router = APIRouter()

__version__ = os.getenv("AVAS_VERSION", "0.1.0")


@router.get("/info", tags=["Utility"])
def info():
    """
    Enlist container configuration.

    """
    try:
        about = dict(
            version=__version__,
            tensorrt_version=os.getenv("TRT_VERSION", os.getenv("TENSORRT_VERSION")),
            log_level=settings.log_level,
            models=settings.models.dict(),
            defaults=settings.defaults.dict(),
        )
        about["models"].pop("device", None)
        return about
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health", tags=["Utility"])
async def check_health(processing: ProcessingDep):
    """
    Execute detection request with default parameters to verify system is working.
    """
    data = Images(urls=["test_images/person.jpg"])

    try:
        await processing.extract(images=data)
        return {"status": "ok"}
    except Exception:
        raise HTTPException(500, detail="self check failed")


@router.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")
