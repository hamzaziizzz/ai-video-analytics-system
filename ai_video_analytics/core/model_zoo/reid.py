import os
import re
from pathlib import Path
from typing import Optional, Tuple

import torch

from ai_video_analytics.core.utils.logging import get_logger

DEFAULT_REID_VARIANT = "osnet_x0_25"
DEFAULT_REID_INPUT_HW = (256, 128)
DEFAULT_REID_OPSET = 18


def prepare_reid_assets(config) -> Optional[str]:
    """Ensure the configured ReID model exists for the selected backend."""
    logger = get_logger("models.reid")
    model_ref = getattr(config.features, "tracker_reid_model", None)
    if not model_ref:
        logger.info("ReID model not configured; skipping export.")
        return None
    if str(model_ref).strip().lower() in {"auto", "default", "osnet"}:
        model_ref = DEFAULT_REID_VARIANT

    backend = (config.inference.engine or "").lower()
    if backend in {"tensorrt", "trt"}:
        target_ext = ".engine"
    elif backend == "onnx":
        target_ext = ".onnx"
    else:
        logger.info("ReID export skipped: unsupported backend '%s'", backend)
        return None

    target_path, onnx_path, variant = _resolve_paths(
        model_ref,
        target_ext,
        fp16=config.inference.fp16,
    )
    config.features.tracker_reid_model = str(target_path)

    if target_path.exists():
        logger.info("ReID model ready: %s", target_path)
        return str(target_path)

    if not onnx_path.exists():
        logger.info("Exporting ReID ONNX: %s (variant=%s)", onnx_path, variant)
        _export_reid_onnx(variant, onnx_path, batch=1, input_hw=DEFAULT_REID_INPUT_HW, opset=DEFAULT_REID_OPSET)

    if target_ext == ".onnx":
        logger.info("ReID model ready: %s", onnx_path)
        return str(onnx_path)

    from ai_video_analytics.core.inference.trt_builder import build_trt_engine

    max_batch = max(1, int(getattr(config.features, "tracker_reid_batch_size", 1)))
    logger.info("Building ReID TensorRT engine: %s", target_path)
    build_trt_engine(
        onnx_path=onnx_path,
        engine_path=target_path,
        fp16=config.inference.fp16,
        max_batch_size=max_batch,
        input_size=DEFAULT_REID_INPUT_HW,
        int8=False,
        gpu_id=_parse_gpu_id(config.inference.device) or 0,
    )
    return str(target_path)


def _resolve_paths(model_ref: str, target_ext: str, fp16: bool) -> tuple[Path, Path, str]:
    models_dir = _resolve_models_dir() / "reid"
    models_dir.mkdir(parents=True, exist_ok=True)

    raw_path = Path(os.path.expanduser(model_ref))
    variant = _parse_variant(raw_path.stem) or DEFAULT_REID_VARIANT

    if raw_path.suffix.lower() == target_ext:
        target_path = raw_path
    else:
        target_path = models_dir / _default_filename(variant, target_ext, fp16=fp16)

    if raw_path.suffix.lower() == ".onnx":
        onnx_path = raw_path
    else:
        onnx_path = models_dir / f"{variant}.onnx"

    return target_path, onnx_path, variant


def _default_filename(variant: str, suffix: str, fp16: bool) -> str:
    if suffix == ".engine":
        tag = "_fp16" if fp16 else "_fp32"
        return f"{variant}{tag}{suffix}"
    return f"{variant}{suffix}"


def _parse_variant(name: str) -> Optional[str]:
    match = re.search(r"osnet_[a-z0-9_]+", name.lower())
    return match.group(0) if match else None


def _resolve_models_dir() -> Path:
    env_dir = os.environ.get("AVAS_MODELS_DIR") or os.environ.get("MODELS_DIR") or "models"
    return Path(env_dir).resolve()


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


def _strip_classifier(model: torch.nn.Module) -> torch.nn.Module:
    for attr in ("classifier", "fc", "head"):
        if hasattr(model, attr):
            setattr(model, attr, torch.nn.Identity())
    return model


class _ReIDWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def _export_reid_onnx(variant: str, output: Path, batch: int, input_hw: Tuple[int, int], opset: int) -> None:
    model = _load_osnet(variant)
    model = _strip_classifier(model)
    model = _ReIDWrapper(model)
    model.eval()

    height, width = input_hw
    dummy = torch.zeros((batch, 3, height, width), dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        export_params=True,
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
    )


def _load_osnet(variant: str) -> torch.nn.Module:
    try:
        import torchreid  # type: ignore
    except Exception as exc:
        raise RuntimeError("torchreid is required to export the ReID model") from exc

    _ensure_gdown()
    return torchreid.models.build_model(
        name=variant,
        num_classes=1000,
        loss="softmax",
        pretrained=True,
        use_gpu=torch.cuda.is_available(),
    )


def _ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except Exception as exc:
        raise RuntimeError("gdown is required to download OSNet pretrained weights") from exc
