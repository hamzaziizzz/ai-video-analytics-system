import asyncio
import base64
import io
import os
import traceback
from typing import List, Optional

import aiofiles
import aiohttp
import numpy as np

from .logging import get_logger

_logger = get_logger("image_provider")

_nvjpeg = None
_turbojpeg = None


def _get_jpeg_decoder():
    global _nvjpeg, _turbojpeg
    if _nvjpeg is None:
        if os.getenv("USE_NVJPEG", "false").lower() in {"1", "true", "yes"}:
            try:
                from nvjpeg import NvJpeg

                _nvjpeg = NvJpeg()
            except Exception:
                _nvjpeg = False
        else:
            _nvjpeg = False
    if _nvjpeg:
        return _nvjpeg

    if _turbojpeg is None:
        try:
            from turbojpeg import TurboJPEG

            _turbojpeg = TurboJPEG()
        except Exception:
            _turbojpeg = False
    return _turbojpeg


def _decode_bytes(im_bytes: bytes):
    decoder = _get_jpeg_decoder()
    if decoder:
        try:
            if hasattr(decoder, "decode"):
                return decoder.decode(im_bytes)
        except Exception:
            pass
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for image decode fallback") from exc
    data = np.frombuffer(im_bytes, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


async def _read_bytes(path: str) -> bytes:
    async with aiofiles.open(path, mode="rb") as handle:
        return await handle.read()


async def _fetch_bytes(url: str, session: aiohttp.ClientSession, headers: Optional[dict] = None) -> bytes:
    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        return await resp.read()


def _b64_to_bytes(blob: str, b64_decode: bool) -> bytes:
    if not b64_decode:
        return blob if isinstance(blob, bytes) else blob.encode("utf-8")
    payload = blob.split(",")[-1]
    return base64.b64decode(payload)


async def get_images(
    images,
    decode: bool = True,
    session: Optional[aiohttp.ClientSession] = None,
    b64_decode: bool = True,
    headers: Optional[dict] = None,
):
    results = []
    from ai_video_analytics.settings import Settings

    settings = Settings()
    headers = headers or settings.defaults.img_req_headers
    urls = images.urls or []
    data_list = images.data or []

    if not urls and not data_list:
        return results

    if session is None:
        timeout = aiohttp.ClientTimeout(total=30.0)
        session = aiohttp.ClientSession(timeout=timeout)
        close_session = True
    else:
        close_session = False

    async def _handle_bytes(payload: bytes):
        if not decode:
            return {"data": payload, "traceback": None}
        try:
            img = _decode_bytes(payload)
            if img is None:
                raise RuntimeError("Failed to decode image")
            return {"data": img, "traceback": None}
        except Exception:
            return {"data": None, "traceback": traceback.format_exc()}

    try:
        for entry in data_list:
            try:
                raw = _b64_to_bytes(entry, b64_decode=b64_decode) if isinstance(entry, str) else entry
                results.append(await _handle_bytes(raw))
            except Exception:
                results.append({"data": None, "traceback": traceback.format_exc()})

        for url in urls:
            try:
                if url.startswith("http://") or url.startswith("https://"):
                    raw = await _fetch_bytes(url, session=session, headers=headers)
                else:
                    local_path = url
                    if not os.path.isabs(local_path):
                        local_path = os.path.join(settings.root_images_dir, local_path)
                    raw = await _read_bytes(local_path)
                results.append(await _handle_bytes(raw))
            except Exception:
                results.append({"data": None, "traceback": traceback.format_exc()})
    finally:
        if close_session:
            await session.close()

    return results
