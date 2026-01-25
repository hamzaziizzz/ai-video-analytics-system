import sys
from pathlib import Path
from typing import Iterable, List, Optional

import tensorrt as trt

from ..utils.logging import get_logger
from .yolo import letterbox

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_paths: List[Path],
        batch_size: int,
        input_hw: tuple[int, int],
        cache_path: Optional[Path],
    ) -> None:
        super().__init__()
        try:
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError("pycuda is required for INT8 calibration") from exc

        self.image_paths = image_paths
        self.batch_size = max(1, batch_size)
        self.input_hw = input_hw
        self.cache_path = cache_path
        self._index = 0
        self.logger = get_logger("inference.trt_calib")
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * input_hw[0] * input_hw[1] * 4)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: Iterable[str]):  # pragma: no cover - TRT calls
        try:
            import cv2
            import numpy as np
            import pycuda.driver as cuda
        except ImportError:
            return None

        if self._index + self.batch_size > len(self.image_paths):
            return None

        batch_paths = self.image_paths[self._index : self._index + self.batch_size]
        self._index += self.batch_size
        images = []
        for path in batch_paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
            img, _, _ = letterbox(image, self.input_hw)
            images.append(img)
        if len(images) < self.batch_size:
            return None

        batch = np.stack(images, axis=0).astype(np.float32)
        batch = batch[..., ::-1]
        batch = np.transpose(batch, (0, 3, 1, 2))
        batch = batch / 255.0
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        return [int(self.device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_path and self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(cache)


def _collect_calibration_images(data_path: Path, limit: int) -> List[Path]:
    if data_path.is_dir():
        images = sorted(
            p for p in data_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        return images[:limit] if limit > 0 else images

    if data_path.suffix.lower() in {".txt"}:
        lines = [Path(line.strip()) for line in data_path.read_text().splitlines() if line.strip()]
        return lines[:limit] if limit > 0 else lines

    if data_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            from ultralytics.utils.yaml import yaml_load
        except Exception as exc:
            raise RuntimeError("ultralytics is required to parse calibration yaml") from exc
        data = yaml_load(data_path)
        candidates = []
        for key in ("val", "train"):
            value = data.get(key)
            if not value:
                continue
            if isinstance(value, (list, tuple)):
                candidates.extend([Path(v) for v in value])
            else:
                candidates.append(Path(value))
        images: List[Path] = []
        for item in candidates:
            path = item if item.is_absolute() else data_path.parent / item
            if path.is_dir():
                images.extend(path.rglob("*"))
            elif path.exists():
                images.append(path)
        images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        images = sorted(images)
        return images[:limit] if limit > 0 else images

    raise RuntimeError(f"Unsupported calibration data path: {data_path}")


def _resolve_profile_shapes(
    input_shape: List[int],
    max_batch: int,
    input_size: tuple[int, int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Build TRT min/opt/max profile shapes with fixed H/W and optional dynamic batch."""
    if len(input_shape) < 4:
        raise RuntimeError(f"Unsupported ONNX input rank: {input_shape}")

    min_shape = list(input_shape)
    opt_shape = list(input_shape)
    max_shape = list(input_shape)

    batch_dim = input_shape[0]
    if batch_dim < 1:
        min_shape[0] = 1
        opt_shape[0] = max(1, max_batch // 2)
        max_shape[0] = max_batch
    else:
        min_shape[0] = batch_dim
        opt_shape[0] = batch_dim
        max_shape[0] = batch_dim

    h, w = input_size
    min_h = h
    min_w = w
    max_h = h
    max_w = w
    for idx, dim in enumerate(input_shape[1:], start=1):
        if dim >= 1:
            min_shape[idx] = dim
            opt_shape[idx] = dim
            max_shape[idx] = dim
            continue
        if idx == 2:
            min_shape[idx] = min_h
            opt_shape[idx] = h
            max_shape[idx] = max_h
        elif idx == 3:
            min_shape[idx] = min_w
            opt_shape[idx] = w
            max_shape[idx] = max_w
        else:
            min_shape[idx] = 1
            opt_shape[idx] = 1
            max_shape[idx] = 1

    return tuple(min_shape), tuple(opt_shape), tuple(max_shape)


def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool,
    max_batch_size: int,
    input_size: Optional[tuple[int, int]] = None,
    int8: bool = False,
    gpu_id: int = 0,
    calib_data: Optional[Path] = None,
    calib_images: int = 300,
    calib_cache: Optional[Path] = None,
    workspace_mb: int = 2048,
) -> None:
    logger = get_logger("inference.trt_builder")
    logger.info(
        "Building TensorRT engine (fp16=%s int8=%s batch=%s gpu=%s workspace=%sMB)",
        fp16,
        int8,
        max_batch_size,
        gpu_id,
        workspace_mb,
    )
    logger.info("Using ONNX: %s", onnx_path)
    logger.info("Target engine: %s", engine_path)
    cuda_context = None
    if int8:
        try:
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError("pycuda is required for INT8 calibration") from exc
        cuda.init()
        cuda_context = cuda.Device(int(gpu_id)).make_context()
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        has_fp16 = builder.platform_has_fast_fp16
        if fp16 or has_fp16:
            if not has_fp16 and fp16:
                logger.warning("Builder reports no fast FP16 support; performance may drop.")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logger.info("Building engine in FP32 mode.")

        if int8:
            if not calib_data:
                raise RuntimeError("INT8 calibration data is required for INT8 TensorRT build")
            config.set_flag(trt.BuilderFlag.INT8)

        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
            trt10 = True
        else:
            config.max_workspace_size = workspace_mb * 1024 * 1024
            trt10 = False

        logger.info("Parsing ONNX...")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for idx in range(parser.num_errors):
                    logger.error("ONNX parse error: %s", parser.get_error(idx))
                sys.exit(1)

        max_batch = max(1, int(max_batch_size))
        input_tensor = network.get_input(0)
        input_shape = list(input_tensor.shape)
        if input_size is None:
            input_size = (640, 640)
        min_shape, opt_shape, max_shape = _resolve_profile_shapes(input_shape, max_batch, input_size)
        profile = builder.create_optimization_profile()
        logger.info(
            "Optimization profile: min=%s opt=%s max=%s",
            min_shape,
            opt_shape,
            max_shape,
        )
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        if int8:
            input_hw = (max_shape[2], max_shape[3])
            images = _collect_calibration_images(calib_data, calib_images)
            if len(images) < max_batch:
                raise RuntimeError(
                    f"INT8 calibration dataset has {len(images)} images, requires >= batch size {max_batch}."
                )
            cache_path = calib_cache or engine_path.with_suffix(".calib")
            config.int8_calibrator = Int8Calibrator(
                image_paths=images,
                batch_size=max_batch,
                input_hw=input_hw,
                cache_path=cache_path,
            )
            logger.info("INT8 calibration: %d images, cache=%s", len(images), cache_path)

        logger.info("Building TensorRT engine (this may take several minutes)...")
        if trt10:
            engine_bytes = builder.build_serialized_network(network, config)
        else:
            engine = builder.build_engine(network, config=config)
            engine_bytes = engine.serialize() if engine else None

        if engine_bytes is None:
            raise RuntimeError("Failed to build TensorRT engine")

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)

        logger.info("Built TensorRT engine: %s", engine_path)
    if cuda_context is not None:
        try:
            cuda_context.pop()
            cuda_context.detach()
        except Exception:
            pass
