import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

from onnxsim import simplify

from ..utils.logging import get_logger
from ..utils.model_assets import download_file
from ..utils.model_registry import get_registry
from ..utils.device import resolve_torch_device

DEFAULT_ULTRALYTICS_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def prepare_yolo_inference_assets(config) -> None:
    if config.models.algorithm.lower() != "yolo":
        return

    engine = config.inference.engine.lower()
    trt_impl = (getattr(config.inference, "trt_implementation", "custom") or "custom").lower()
    model_ref = config.models.detection_model or config.inference.model_path
    model_path = Path(config.inference.model_path)
    expected_suffix = _expected_suffix(engine)
    explicit_path = _is_explicit_path(model_ref)
    if engine == "openvino" and explicit_path:
        resolved_dir = _resolve_openvino_dir(model_path)
        if resolved_dir:
            config.inference.model_path = str(resolved_dir)
            return
    if explicit_path and model_path.exists():
        if engine == "openvino":
            if model_path.is_dir() or model_path.suffix == ".xml":
                return
        elif model_path.suffix == expected_suffix:
            return

    models_dir = _resolve_models_dir()
    if engine in {"ultralytics", "pt", "pytorch"}:
        if explicit_path and model_path.exists() and model_path.suffix == ".pt":
            pt_path = model_path
        else:
            pt_path = ensure_yolo_pt(model_ref, models_dir)
        config.inference.model_path = str(pt_path)
        return

    if engine in {"onnx", "openvino", "tensorrt", "trt"}:
        if model_path.exists() and model_path.suffix == ".pt":
            pt_path = model_path
        else:
            pt_path = ensure_yolo_pt(model_ref, models_dir)
        if engine in {"tensorrt", "trt"} and config.inference.int8 and not config.inference.int8_calib_data:
            get_logger("models.export").warning(
                "INT8 TensorRT build requested but INT8_CALIB_DATA is not set; disabling INT8."
            )
            config.inference.int8 = False
        export_format, suffix = _export_format(engine)
        gpu_id = _parse_gpu_id(config.inference.device)
        if gpu_id is None and engine in {"tensorrt", "trt"}:
            gpu_id = 0
        extra_tags = []
        if trt_impl in {"ultralytics", "ultra", "yolo"}:
            extra_tags.append("ultralytics")
        if config.inference.trt_dynamic_shapes:
            extra_tags.append("dyn")
        if not extra_tags:
            extra_tags = None
        output_path = _resolve_output_path(
            model_path,
            pt_path,
            suffix,
            engine,
            models_dir=models_dir,
            gpu_id=gpu_id,
            batch_size=config.inference.batch_size,
            fp16=config.inference.fp16,
            int8=config.inference.int8,
            input_size=_resolve_export_size(config),
            prefer_model_path=explicit_path,
            extra_tags=extra_tags,
        )
        if engine == "openvino":
            if output_path.exists():
                config.inference.model_path = str(output_path)
                return
        elif output_path.exists():
            if engine in {"tensorrt", "trt"} and trt_impl in {"ultralytics", "ultra", "yolo"}:
                if _has_ultralytics_metadata(output_path):
                    config.inference.model_path = str(output_path)
                    return
            else:
                config.inference.model_path = str(output_path)
                return
        if engine in {"tensorrt", "trt"} and trt_impl in {"ultralytics", "ultra", "yolo"}:
            exported_path = export_yolo(
                pt_path,
                output_path=output_path,
                export_format="engine",
                device=config.inference.device,
                fp16=config.inference.fp16,
                int8=config.inference.int8,
                imgsz=_resolve_export_size(config),
                batch_size=config.inference.batch_size,
                nms=config.inference.trt_ultralytics_nms,
                dynamic=config.inference.trt_dynamic_shapes,
            )
            config.inference.model_path = str(exported_path)
            return
        elif output_path.exists():
            config.inference.model_path = str(output_path)
            return
        if engine in {"tensorrt", "trt"}:
            onnx_dir = models_dir / "onnx"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = onnx_dir / f"{pt_path.stem}.onnx"
            if not onnx_path.exists():
                export_yolo(
                    pt_path,
                    output_path=onnx_path,
                    export_format="onnx",
                    device=config.inference.device,
                    fp16=config.inference.fp16,
                    int8=False,
                    imgsz=_resolve_export_size(config),
                    batch_size=config.inference.batch_size,
                    opset=17,
                    dynamic=config.inference.trt_dynamic_shapes,
                )
            from ..inference.trt_builder import build_trt_engine

            build_trt_engine(
                onnx_path=onnx_path,
                engine_path=output_path,
                fp16=config.inference.fp16,
                max_batch_size=config.inference.batch_size,
                input_size=config.inference.input_size,
                dynamic_shapes=config.inference.trt_dynamic_shapes,
                dynamic_min_size=config.inference.trt_dynamic_min_size,
                dynamic_max_size=config.inference.trt_dynamic_max_size,
                dynamic_stride=config.inference.trt_dynamic_stride,
                int8=config.inference.int8,
                gpu_id=gpu_id or 0,
                calib_data=Path(config.inference.int8_calib_data) if config.inference.int8_calib_data else None,
                calib_images=config.inference.int8_calib_images,
                calib_cache=Path(config.inference.int8_calib_cache) if config.inference.int8_calib_cache else None,
            )
            config.inference.model_path = str(output_path)
        else:
            exported_path = export_yolo(
                pt_path,
                output_path=output_path,
                export_format=export_format,
                device=config.inference.device,
                fp16=config.inference.fp16,
                int8=config.inference.int8,
                imgsz=_resolve_export_size(config),
                batch_size=config.inference.batch_size,
                dynamic=config.inference.trt_dynamic_shapes,
            )
            if engine == "openvino":
                config.inference.model_path = str(output_path if output_path.exists() else exported_path)
            else:
                config.inference.model_path = str(exported_path)
        return

    raise RuntimeError(f"Unsupported inference backend for YOLO assets: {config.inference.engine}")


def ensure_yolo_pt(model_ref: str, models_dir: Path) -> Path:
    path_ref = Path(model_ref)
    pytorch_dir = models_dir / "pytorch-models"
    if path_ref.suffix == ".pt":
        dest = path_ref if path_ref.is_absolute() else pytorch_dir / path_ref.name
        filename = dest.name
    else:
        filename = f"{_normalize_model_name(model_ref)}.pt"
        dest = pytorch_dir / filename

    if dest.exists():
        return dest

    url = _resolve_ultralytics_url(filename)
    download_file(url, dest, expected_sha256=None)
    return dest


def export_yolo(
    model_path: Path,
    output_path: Optional[Path],
    export_format: str,
    device: str,
    fp16: bool,
    int8: bool,
    imgsz,
    batch_size: int,
    opset: Optional[int] = None,
    nms: bool = True,
    dynamic: Optional[bool] = None,
) -> Path:
    logger = get_logger("models.export")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required to export YOLO models") from exc

    resolved_device = resolve_torch_device(device)
    if export_format == "engine" and resolved_device.startswith("cpu"):
        raise RuntimeError("TensorRT export requires a CUDA-capable device")

    model = YOLO(str(model_path))
    if dynamic is None:
        dynamic = batch_size > 1
    export_kwargs = dict(
        format=export_format,
        half=fp16,
        int8=int8,
        device=resolved_device,
        imgsz=imgsz,
        batch=batch_size,
        dynamic=bool(dynamic),
        nms=bool(nms),
        simplify=True,
        verbose=False,
    )
    if opset is not None and export_format == "onnx":
        export_kwargs["opset"] = opset

    exported = model.export(
        **export_kwargs,
    )

    exported_path = Path(exported)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if exported_path.resolve() != output_path.resolve():
            source_path = exported_path
            if output_path.exists():
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()
            shutil.move(str(source_path), str(output_path))
            exported_path = output_path
            if export_format == "engine":
                _relocate_onnx_artifact(model_path, output_path)

    logger.info("Exported %s model to %s", export_format, exported_path)
    return exported_path


def _resolve_export_size(config) -> tuple[int, int]:
    """Pick export size for dynamic TRT builds (max size to cover profile)."""
    size = tuple(config.inference.input_size)
    if config.inference.trt_dynamic_shapes and config.inference.trt_dynamic_max_size:
        return tuple(config.inference.trt_dynamic_max_size)
    return size


def _relocate_onnx_artifact(model_path: Path, engine_path: Path) -> None:
    onnx_src = model_path.with_suffix(".onnx")
    if not onnx_src.exists():
        return
    onnx_dir = engine_path.parent.parent / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_dst = onnx_dir / onnx_src.name
    if onnx_dst.exists():
        onnx_dst.unlink()
    shutil.move(str(onnx_src), str(onnx_dst))


def _resolve_models_dir() -> Path:
    env_dir = os.environ.get("AVAS_MODELS_DIR") or os.environ.get("MODELS_DIR") or "models"
    return Path(env_dir).resolve()


def _has_ultralytics_metadata(engine_path: Path) -> bool:
    try:
        with open(engine_path, "rb") as handle:
            header = handle.read(4)
            if len(header) != 4:
                return False
            meta_len = int.from_bytes(header, byteorder="little")
            size = engine_path.stat().st_size
            if meta_len <= 0 or meta_len > size - 4:
                return False
            meta_raw = handle.read(meta_len)
        json.loads(meta_raw.decode("utf-8"))
        return True
    except Exception:
        return False


def _normalize_model_name(model_ref: str) -> str:
    name = Path(model_ref).name
    if name.endswith((".pt", ".onnx", ".engine", ".xml")):
        return Path(name).stem
    return name


def _resolve_ultralytics_url(filename: str) -> str:
    registry = get_registry()
    provider = registry.provider("ultralytics") if registry else None
    base_url = DEFAULT_ULTRALYTICS_BASE_URL
    if provider and provider.get("base_url"):
        base_url = provider["base_url"]
    return f"{base_url.rstrip('/')}/{filename}"


def _export_format(engine: str) -> tuple[str, str]:
    normalized = engine.lower()
    if normalized in {"tensorrt", "trt"}:
        return "engine", ".engine"
    if normalized == "onnx":
        return "onnx", ".onnx"
    if normalized == "openvino":
        return "openvino", ".xml"
    return "onnx", ".onnx"


def _resolve_output_path(
    model_path: Path,
    pt_path: Path,
    suffix: str,
    engine: str,
    models_dir: Path,
    gpu_id: Optional[int],
    batch_size: int,
    fp16: bool,
    int8: bool,
    input_size: Optional[tuple[int, int]],
    prefer_model_path: bool,
    extra_tags: Optional[List[str]] = None,
) -> Path:
    if prefer_model_path and model_path.suffix and model_path.suffix == suffix:
        return model_path
    base_dir = models_dir
    if engine.lower() in {"tensorrt", "trt"}:
        tag = _engine_tag(gpu_id, batch_size, fp16, int8, input_size=input_size, extra_tags=extra_tags)
        return base_dir / "trt-engines" / f"{pt_path.stem}{tag}{suffix}"
    if engine.lower() == "onnx":
        return base_dir / "onnx" / f"{pt_path.stem}{suffix}"
    if engine.lower() == "openvino":
        tag = _engine_tag(None, batch_size, fp16, int8)
        return base_dir / "openvino" / f"{pt_path.stem}{tag}_openvino_model"
    return base_dir / f"{pt_path.stem}{suffix}"


def _expected_suffix(engine: str) -> str:
    normalized = engine.lower()
    if normalized in {"tensorrt", "trt"}:
        return ".engine"
    if normalized == "onnx":
        return ".onnx"
    if normalized == "openvino":
        return ""
    return ".pt"


def _engine_tag(
    gpu_id: Optional[int],
    batch_size: int,
    fp16: bool,
    int8: bool,
    input_size: Optional[tuple[int, int]] = None,
    extra_tags: Optional[List[str]] = None,
) -> str:
    tags = []
    if gpu_id is not None:
        tags.append(f"gpu{gpu_id}")
    if batch_size and batch_size > 0:
        tags.append(f"bs{batch_size}")
    if input_size:
        height, width = input_size
        tags.append(f"sz{width}x{height}")
    if fp16:
        tags.append("fp16")
    if int8:
        tags.append("int8")
    if extra_tags:
        tags.extend(tag for tag in extra_tags if tag)
    if not tags:
        return ""
    return "_" + "_".join(tags)


def _parse_gpu_id(device: str) -> Optional[int]:
    if not device:
        return None
    value = device.strip().lower()
    if value.startswith("cuda:"):
        parts = value.split(":", 1)
        try:
            return int(parts[1])
        except (ValueError, TypeError):
            return None
    return None


def _resolve_openvino_dir(model_path: Path) -> Optional[Path]:
    if model_path.is_dir():
        if any(model_path.glob("*.xml")):
            if model_path.name.endswith("_openvino_model"):
                return model_path
            target = model_path.parent / f"{model_path.name}_openvino_model"
            if target.exists():
                return target
            shutil.move(str(model_path), str(target))
            return target
    if model_path.suffix == ".xml":
        parent = model_path.parent
        if parent.exists() and any(parent.glob("*.xml")):
            if parent.name.endswith("_openvino_model"):
                return parent
            target = parent.parent / f"{parent.name}_openvino_model"
            if target.exists():
                return target
            shutil.move(str(parent), str(target))
            return target
    return None


def _is_explicit_path(value: str) -> bool:
    if not value:
        return False
    if "/" in value or "\\" in value:
        return True
    return value.endswith((".pt", ".onnx", ".engine", ".xml"))
