from typing import Optional, List, Union

import pydantic
from pydantic import BaseModel

from ai_video_analytics.settings import Settings

example_img = "test_images/person.jpg"
settings = Settings()


class Images(BaseModel):
    data: Optional[List[str] | List[bytes]] = pydantic.Field(
        default=None,
        json_schema_extra={"example": None},
        description="List of base64 encoded images",
    )
    urls: Optional[List[str]] = pydantic.Field(
        default=None,
        json_schema_extra={"example": [example_img]},
        description="List of image urls or file paths",
    )


class PeopleExtract(BaseModel):
    images: Images

    threshold: Optional[float] = pydantic.Field(
        default=settings.defaults.det_thresh,
        json_schema_extra={"example": settings.defaults.det_thresh},
        description="Detector threshold",
    )
    return_person_data: Optional[bool] = pydantic.Field(
        default=settings.defaults.return_person_data,
        json_schema_extra={"example": settings.defaults.return_person_data},
        description="Return crops encoded in base64",
    )
    limit_people: Optional[int] = pydantic.Field(
        default=0,
        json_schema_extra={"example": 0},
        description="Maximum number of detections to return",
    )
    min_person_size: Optional[int] = pydantic.Field(
        default=0,
        json_schema_extra={"example": 0},
        description="Ignore detections smaller than this size",
    )
    reset_tracking: Optional[bool] = pydantic.Field(
        default=False,
        json_schema_extra={"example": False},
        description="Reset tracker state before processing this request",
    )
    verbose_timings: Optional[bool] = pydantic.Field(
        default=False,
        json_schema_extra={"example": True},
        description="Return all timings.",
    )
    msgpack: Optional[bool] = pydantic.Field(
        default=False,
        json_schema_extra={"example": False},
        description="Use MSGPACK for response serialization",
    )
    img_req_headers: Optional[dict] = pydantic.Field(
        default={},
        json_schema_extra={"example": {}},
        description="Custom headers for image retrieving from remote servers",
    )


class PoseExtract(BaseModel):
    images: Images

    threshold: Optional[float] = pydantic.Field(
        default=settings.defaults.det_thresh,
        json_schema_extra={"example": settings.defaults.det_thresh},
        description="Detector threshold",
    )
    limit_people: Optional[int] = pydantic.Field(
        default=0,
        json_schema_extra={"example": 0},
        description="Maximum number of detections to return",
    )
    min_person_size: Optional[int] = pydantic.Field(
        default=0,
        json_schema_extra={"example": 0},
        description="Ignore detections smaller than this size",
    )
    reset_tracking: Optional[bool] = pydantic.Field(
        default=False,
        json_schema_extra={"example": False},
        description="Reset tracker state before processing this request",
    )
    verbose_timings: Optional[bool] = pydantic.Field(
        default=False,
        json_schema_extra={"example": True},
        description="Return all timings.",
    )
    msgpack: Optional[bool] = pydantic.Field(
        default=False,
        json_schema_extra={"example": False},
        description="Use MSGPACK for response serialization",
    )
    img_req_headers: Optional[dict] = pydantic.Field(
        default={},
        json_schema_extra={"example": {}},
        description="Custom headers for image retrieving from remote servers",
    )


class PeopleDraw(BaseModel):
    images: Images

    threshold: Optional[float] = pydantic.Field(
        default=settings.defaults.det_thresh,
        json_schema_extra={"example": settings.defaults.det_thresh},
        description="Detector threshold",
    )
    draw_scores: Optional[bool] = pydantic.Field(
        default=True,
        json_schema_extra={"example": True},
        description="Draw detection scores",
    )
    draw_sizes: Optional[bool] = pydantic.Field(
        default=True,
        json_schema_extra={"example": True},
        description="Draw detection sizes",
    )
    limit_people: Optional[int] = pydantic.Field(
        default=0,
        json_schema_extra={"example": 0},
        description="Maximum number of detections to return",
    )
    min_person_size: Optional[int] = pydantic.Field(
        default=0,
        json_schema_extra={"example": 0},
        description="Ignore detections smaller than this size",
    )


class PoseDraw(BaseModel):
    images: Images

    threshold: Optional[float] = pydantic.Field(
        default=settings.defaults.det_thresh,
        json_schema_extra={"example": settings.defaults.det_thresh},
        description="Detector threshold",
    )
    draw_scores: Optional[bool] = pydantic.Field(
        default=True,
        json_schema_extra={"example": True},
        description="Draw detection scores",
    )
    draw_sizes: Optional[bool] = pydantic.Field(
        default=True,
        json_schema_extra={"example": True},
        description="Draw detection sizes",
    )
    limit_people: Optional[int] = pydantic.Field(
        default=0,
        json_schema_extra={"example": 0},
        description="Maximum number of detections to return",
    )
    min_person_size: Optional[int] = pydantic.Field(
        default=0,
        json_schema_extra={"example": 0},
        description="Ignore detections smaller than this size",
    )


class TrackingStream(BaseModel):
    source: Union[str, int] = pydantic.Field(
        ...,
        json_schema_extra={"example": 0},
        description="Camera index, file path, or HTTP/RTSP URL",
    )
    pose: Optional[bool] = pydantic.Field(
        default=False,
        json_schema_extra={"example": False},
        description="Use pose detection instead of person detection",
    )
    threshold: Optional[float] = pydantic.Field(
        default=settings.defaults.det_thresh,
        json_schema_extra={"example": settings.defaults.det_thresh},
        description="Detector threshold",
    )
    draw_scores: Optional[bool] = pydantic.Field(
        default=True,
        json_schema_extra={"example": True},
        description="Draw detection scores",
    )
    draw_sizes: Optional[bool] = pydantic.Field(
        default=True,
        json_schema_extra={"example": True},
        description="Draw detection sizes",
    )
    limit_people: Optional[int] = pydantic.Field(
        default=0,
        json_schema_extra={"example": 0},
        description="Maximum number of detections to return",
    )
    min_person_size: Optional[int] = pydantic.Field(
        default=0,
        json_schema_extra={"example": 0},
        description="Ignore detections smaller than this size",
    )
    fps_limit: Optional[float] = pydantic.Field(
        default=0.0,
        json_schema_extra={"example": 0.0},
        description="Output FPS limit. Use 0 to disable throttling",
    )
    jpeg_quality: Optional[int] = pydantic.Field(
        default=80,
        json_schema_extra={"example": 80},
        description="JPEG quality for MJPEG streaming",
    )
    show_fps: Optional[bool] = pydantic.Field(
        default=True,
        json_schema_extra={"example": True},
        description="Overlay FPS on the stream",
    )
