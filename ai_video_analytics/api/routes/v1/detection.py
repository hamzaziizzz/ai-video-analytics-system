from typing import Annotated, Callable, List, Optional

import msgpack
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, Response
from fastapi.responses import UJSONResponse
from fastapi.routing import APIRoute
from starlette.responses import PlainTextResponse, StreamingResponse

from ai_video_analytics.schemas import PeopleDraw, PeopleExtract
from ai_video_analytics.core.processing import ProcessingDep


class MsgPackRequest(Request):
    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            body = await super().body()
            if "application/msgpack" in self.headers.getlist("Content-Type"):
                body = msgpack.unpackb(body)
            self._body = body
        return self._body


class MsgpackRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            request = MsgPackRequest(request.scope, request.receive)
            return await original_route_handler(request)

        return custom_route_handler


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
    file: bytes = File(...),
    threshold: float = Form(0.6),
    draw_scores: bool = Form(True),
    draw_sizes: bool = Form(True),
    limit_people: int = Form(0),
    use_rotation: bool = Form(False),
):
    """
    Return image with drawn detections for testing purposes.

       - **file**: Image file (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **draw_sizes**: Draw detection sizes Default: True (*optional*)
       - **limit_people**: Maximum number of detections to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """
    try:
        output = await processing.draw(
            file,
            threshold=threshold,
            draw_scores=draw_scores,
            draw_sizes=draw_sizes,
            limit_people=limit_people,
            multipart=True,
        )
        output.seek(0)
        return StreamingResponse(output, media_type="image/jpg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
