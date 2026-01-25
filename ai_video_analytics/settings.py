from typing import Union, Optional, List

import os

from pydantic.v1.env_settings import BaseSettings

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
}


def str_to_int_list(value):
    if isinstance(value, str):
        return list(map(int, value.split(",")))
    return value


class StrToIntList(str):
    @classmethod
    def __get_validators__(cls):
        yield str_to_int_list


class Defaults(BaseSettings):
    return_person_data: bool = False
    det_thresh: float = 0.6
    img_req_headers: dict = headers
    sslv3_hack: bool = False

    class Config:
        env_prefix = "DEF_"


class Models(BaseSettings):
    inference_backend: str = "onnx"
    det_name: str = "yolo26x"
    max_size: Union[StrToIntList, List[int]] = [640, 640]
    det_batch_size: int = 1
    pose_batch_size: Optional[int] = None
    force_fp16: bool = False
    triton_uri: Optional[str] = None
    pose_detection: bool = False
    pose_name: Optional[str] = None

    def apply_aliases(self) -> None:
        det_name = os.getenv("DETECTION_MODEL") or os.getenv("AVAS_DETECTION_MODEL")
        if det_name:
            self.det_name = det_name
        backend = os.getenv("INFERENCE_BACKEND") or os.getenv("AVAS_INFERENCE_BACKEND")
        if backend:
            self.inference_backend = backend
        det_bs = os.getenv("DET_BATCH_SIZE") or os.getenv("BATCH_SIZE") or os.getenv("AVAS_BATCH_SIZE")
        if det_bs and det_bs.isdigit():
            self.det_batch_size = int(det_bs)
        pose_bs = os.getenv("POSE_BATCH_SIZE")
        if pose_bs and pose_bs.isdigit():
            self.pose_batch_size = int(pose_bs)
        max_size = os.getenv("MAX_SIZE")
        if max_size:
            self.max_size = str_to_int_list(max_size)
        force_fp16 = os.getenv("FORCE_FP16")
        if force_fp16 is not None:
            normalized = force_fp16.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                self.force_fp16 = True
            elif normalized in {"0", "false", "no", "n", "off"}:
                self.force_fp16 = False

        pose_detection = os.getenv("POSE_DETECTION")
        if pose_detection is not None:
            normalized = pose_detection.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                self.pose_detection = True
            elif normalized in {"0", "false", "no", "n", "off"}:
                self.pose_detection = False

        pose_name = os.getenv("POSE_DETECTION_MODEL")
        if not pose_name:
            pose_name = os.getenv("POSE_MODEL") or os.getenv("POSE_MODEL_PATH")
        if pose_name:
            self.pose_name = pose_name


class Settings(BaseSettings):
    log_level: str = "INFO"
    root_images_dir: str = "/images"
    port: int = 18080
    models = Models()
    defaults = Defaults()

    def __init__(self, **values):
        super().__init__(**values)
        self.models.apply_aliases()
        port = os.getenv("PORT")
        if port and port.isdigit():
            self.port = int(port)
        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            self.log_level = log_level
