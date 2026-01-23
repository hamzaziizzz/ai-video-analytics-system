import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.request import urlopen

from .logging import get_logger
from .model_registry import ModelFile, ModelRegistry, get_registry


def normalize_sha256(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.split("sha256:", 1)[1]
    return normalized or None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: Optional[str]) -> bool:
    if not path.exists():
        raise FileNotFoundError(path)
    expected = normalize_sha256(expected_sha256)
    if not expected:
        return True
    actual = sha256_file(path)
    return actual == expected


def download_file(
    url: str,
    dest_path: Path,
    expected_sha256: Optional[str],
    overwrite: bool = False,
    timeout_seconds: float = 60.0,
    chunk_size: int = 1024 * 1024,
) -> None:
    logger = get_logger("models.download")
    if dest_path.exists() and not overwrite:
        logger.info("File exists, skipping download: %s", dest_path)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    logger.info("Downloading %s -> %s", url, dest_path)
    with urlopen(url, timeout=timeout_seconds) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)

    os.replace(tmp_path, dest_path)

    if expected_sha256:
        if not verify_file(dest_path, expected_sha256):
            dest_path.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch for {dest_path}")


def resolve_model_files(
    model_name: str,
    file_keys: Optional[Iterable[str]] = None,
    registry: Optional[ModelRegistry] = None,
) -> List[ModelFile]:
    registry = registry or get_registry()
    if file_keys:
        resolved: List[ModelFile] = []
        for key in file_keys:
            entry = registry.file(model_name, key)
            if entry:
                resolved.append(entry)
        return resolved
    return registry.files(model_name)


def ensure_model_files(
    model_name: str,
    file_keys: Optional[Iterable[str]] = None,
    download: bool = False,
    verify: bool = True,
    overwrite: bool = False,
    timeout_seconds: float = 60.0,
    registry: Optional[ModelRegistry] = None,
) -> List[ModelFile]:
    logger = get_logger("models.ensure")
    registry = registry or get_registry()
    files = resolve_model_files(model_name, file_keys=file_keys, registry=registry)
    if not files:
        raise RuntimeError(f"No files found for model '{model_name}'")

    for entry in files:
        if not entry.path.exists():
            if not download or not entry.url:
                raise FileNotFoundError(entry.path)
            download_file(
                entry.url,
                entry.path,
                entry.sha256,
                overwrite=overwrite,
                timeout_seconds=timeout_seconds,
            )
        if verify:
            if entry.sha256:
                if not verify_file(entry.path, entry.sha256):
                    raise RuntimeError(f"SHA256 mismatch for {entry.path}")
                logger.info("Verified %s", entry.path)
            else:
                logger.warning("No checksum for %s", entry.path)
    return files
