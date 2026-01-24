import logging
import os
import ssl
import warnings
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from fastapi_offline import FastAPIOffline
except ImportError:  # pragma: no cover - optional dependency
    FastAPIOffline = FastAPI

from ai_video_analytics.api.routes import v1_router
warnings.filterwarnings(
    "ignore",
    message=r"The value of the smallest subnormal.*",
    category=UserWarning,
)

from ai_video_analytics.core.processing import get_processing
from ai_video_analytics.logger import logger
from ai_video_analytics.settings import Settings

__version__ = os.getenv("AVAS_VERSION", "0.1.0")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s - %(message)s",
        datefmt="[%H:%M:%S]",
    )
    logger.info("Starting processing module...")
    try:
        timeout = ClientTimeout(total=60.0)
        if settings.defaults.sslv3_hack:
            ssl_context = ssl._create_unverified_context()
            ssl_context.set_ciphers("DEFAULT")
            dl_client = aiohttp.ClientSession(timeout=timeout, connector=TCPConnector(ssl=ssl_context))
        else:
            dl_client = aiohttp.ClientSession(timeout=timeout, connector=TCPConnector(ssl=False))
        processing = await get_processing()
        await processing.start(dl_client=dl_client)
        logger.info("Processing module ready!")
    except Exception as exc:
        logger.error(exc)
        raise SystemExit(1)
    yield


def get_app() -> FastAPI:
    application = FastAPIOffline(
        title="AI Video Analytics System",
        description="Person detection REST API",
        version=__version__,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(v1_router)
    return application


app = get_app()
