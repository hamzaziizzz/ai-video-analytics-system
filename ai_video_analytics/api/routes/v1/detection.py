import io
from typing import Annotated, List, Optional

import msgpack
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import UJSONResponse
from starlette.responses import PlainTextResponse, StreamingResponse

from ai_video_analytics.schemas import Images, PeopleDraw, PeopleExtract
from ai_video_analytics.core.processing import ProcessingDep
from ai_video_analytics.api.routes.v1.msgpack import MsgpackRoute
from ai_video_analytics.api.routes.v1.image_utils import tile_images


router = APIRouter(route_class=MsgpackRoute)


@router.post("/detect", tags=["Detection"])
async def extract(
    data: PeopleExtract,
    processing: ProcessingDep,
    accept: Optional[List[str]] = Header(None),
    content_type: Annotated[str | None, Header()] = None,
):
    """
    Person detection endpoint accepts json with
    parameters in following format:

       - **images**: dict containing either links or data lists. (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **return_person_data**: Return crops encoded in base64. Default: False (*optional*)
       - **limit_people**: Maximum number of detections to be processed.  0 for unlimited number. Default: 0 (*optional*)
       - **reset_tracking**: Reset tracker state before processing this request. Default: False (*optional*)
       - **verbose_timings**: Return all timings. Default: False (*optional*)
       - **msgpack**: Serialize output to msgpack format for transfer. Default: False (*optional*)
       \f

       :return:
       List[List[dict]]
    """
    try:
        b64_decode = True
        if content_type == "application/msgpack":
            b64_decode = False
        output = await processing.extract(
            data.images,
            return_person_data=data.return_person_data,
            threshold=data.threshold,
            limit_people=data.limit_people,
            min_person_size=data.min_person_size,
            reset_tracking=data.reset_tracking,
            verbose_timings=data.verbose_timings,
            b64_decode=b64_decode,
            img_req_headers=data.img_req_headers,
        )

        if data.msgpack or "application/x-msgpack" in accept:
            return PlainTextResponse(msgpack.dumps(output, use_single_float=True), media_type="application/x-msgpack")
        return UJSONResponse(output)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/draw", tags=["Detection"])
async def draw(data: PeopleDraw, processing: ProcessingDep):
    """
    Return image with drawn detections for testing purposes.

       - **images**: dict containing either links or data lists. (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **draw_sizes**: Draw detection sizes Default: True (*optional*)
       - **limit_people**: Maximum number of detections to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """
    try:
        output = await processing.draw(
            data.images,
            threshold=data.threshold,
            draw_scores=data.draw_scores,
            limit_people=data.limit_people,
            min_person_size=data.min_person_size,
            draw_sizes=data.draw_sizes,
        )
        output.seek(0)
        return StreamingResponse(output, media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/multipart/draw", tags=["Detection"])
async def draw_upl(
    processing: ProcessingDep,
    files: List[UploadFile] = File(...),
    threshold: float = Form(0.6),
    draw_scores: bool = Form(True),
    draw_sizes: bool = Form(True),
    limit_people: int = Form(0),
    use_rotation: bool = Form(False),
):
    """
    Return image with drawn detections for testing purposes.

       - **files**: Image file(s) (*required*, multiple files return a tiled image)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **draw_sizes**: Draw detection sizes Default: True (*optional*)
       - **limit_people**: Maximum number of detections to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """
    try:
        if len(files) == 1:
            payload = await files[0].read()
            output = await processing.draw(
                payload,
                threshold=threshold,
                draw_scores=draw_scores,
                draw_sizes=draw_sizes,
                limit_people=limit_people,
                multipart=True,
            )
            output.seek(0)
            return StreamingResponse(output, media_type="image/jpg")

        annotated = []
        for upload in files:
            payload = await upload.read()
            output = await processing.draw(
                payload,
                threshold=threshold,
                draw_scores=draw_scores,
                draw_sizes=draw_sizes,
                limit_people=limit_people,
                multipart=True,
            )
            output.seek(0)
            data = np.frombuffer(output.read(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to decode annotated image for {upload.filename or 'upload'}")
            annotated.append(image)
        tiled = tile_images(annotated)
        ok, buffer = cv2.imencode(".jpg", tiled)
        if not ok:
            raise RuntimeError("Failed to encode tiled image")
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/multipart/detect", tags=["Detection"])
async def extract_upl(
    processing: ProcessingDep,
    files: List[UploadFile] = File(...),
    threshold: float = Form(0.6),
    return_person_data: bool = Form(False),
    limit_people: int = Form(0),
    min_person_size: int = Form(0),
    reset_tracking: bool = Form(False),
    verbose_timings: bool = Form(False),
    msgpack: bool = Form(False),
    accept: Optional[List[str]] = Header(None),
):
    """
    Person detection endpoint accepts multipart data with
    parameters in following format:

       - **files**: Image file(s) (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **return_person_data**: Return crops encoded in base64. Default: False (*optional*)
       - **limit_people**: Maximum number of detections to be processed.  0 for unlimited number. Default: 0 (*optional*)
       - **reset_tracking**: Reset tracker state before processing this request. Default: False (*optional*)
       - **verbose_timings**: Return all timings. Default: False (*optional*)
       - **msgpack**: Serialize output to msgpack format for transfer. Default: False (*optional*)
       \f

       :return:
       List[List[dict]]
    """
    try:
        payloads = [await upload.read() for upload in files]
        images = Images(data=payloads)
        output = await processing.extract(
            images,
            return_person_data=return_person_data,
            threshold=threshold,
            limit_people=limit_people,
            min_person_size=min_person_size,
            reset_tracking=reset_tracking,
            verbose_timings=verbose_timings,
            b64_decode=False,
        )
        if msgpack or (accept and "application/x-msgpack" in accept):
            return PlainTextResponse(msgpack.dumps(output, use_single_float=True), media_type="application/x-msgpack")
        return UJSONResponse(output)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
