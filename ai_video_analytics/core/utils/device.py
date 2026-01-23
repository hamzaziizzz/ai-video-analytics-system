from typing import Optional


def resolve_torch_device(device: Optional[str]) -> str:
    if not device:
        return "cpu"
    value = device.strip().lower()
    if value != "auto":
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
