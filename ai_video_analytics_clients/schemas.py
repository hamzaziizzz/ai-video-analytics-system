from typing import Optional, List

from pydantic import BaseModel, ConfigDict


class BaseModelMod(BaseModel):
    model_config = ConfigDict(extra='ignore')


class Took(BaseModelMod):
    total_ms: Optional[float]


class PeopleResponse(BaseModelMod):
    num_det: Optional[int]
    prob: Optional[float]
    bbox: Optional[List[int]]
    class_id: Optional[int]
    class_name: Optional[str]
    persondata: Optional[str]


class ImageResponse(BaseModelMod):
    status: str
    took_ms: Optional[float]
    people: Optional[list[PeopleResponse]]


class DetectionResponse(BaseModelMod):
    took: Took
    data: List[ImageResponse]
