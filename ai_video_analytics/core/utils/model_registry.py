import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .logging import get_logger


@dataclass(frozen=True)
class ModelFile:
    key: str
    path: Path
    sha256: Optional[str] = None
    url: Optional[str] = None


class ModelRegistry:
    def __init__(self, models_dir: Optional[str] = None) -> None:
        if not models_dir:
            models_dir = os.environ.get("AVAS_MODELS_DIR") or os.environ.get("MODELS_DIR") or "models"
        self.models_dir = Path(models_dir)
        self.logger = get_logger("registry")
        self._models, self._providers = self._load_registry()

    def _load_registry(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        base_path = self.models_dir / "models.json"
        override_path = self.models_dir / "models.override.json"
        if not base_path.exists():
            self.logger.warning("Model registry not found: %s", base_path)
            return {}, {}

        base = json.loads(base_path.read_text(encoding="utf-8"))
        if override_path.exists():
            override = json.loads(override_path.read_text(encoding="utf-8"))
            base = self._merge_dicts(base, override)

        providers = {}
        if isinstance(base, dict):
            providers = base.pop("providers", {}) if isinstance(base.get("providers"), dict) else {}

        models: Dict[str, Dict[str, Any]] = {}
        if isinstance(base, dict):
            for key, value in base.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, dict):
                    models[key] = value
        return models, providers

    def _merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def get(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self._models.get(model_name)

    def model_names(self) -> List[str]:
        return sorted(self._models.keys())

    def provider_names(self) -> List[str]:
        return sorted(self._providers.keys())

    def provider(self, name: str) -> Optional[Dict[str, Any]]:
        entry = self._providers.get(name)
        if isinstance(entry, dict):
            return entry
        return None

    def resolve_model_name(self, model_path: str) -> str:
        path = Path(model_path)
        name = path.stem
        return name

    def output_layout(self, model_name: str) -> Optional[str]:
        entry = self.get(model_name) or {}
        return entry.get("outputs", {}).get("layout")

    def output_shape(self, model_name: str) -> Optional[Tuple[int, ...]]:
        entry = self.get(model_name) or {}
        shape = entry.get("outputs", {}).get("shape")
        if shape:
            return tuple(shape)
        return None

    def file(self, model_name: str, file_key: str) -> Optional[ModelFile]:
        entry = self.get(model_name) or {}
        files = entry.get("files") or {}
        if not isinstance(files, dict):
            return None
        file_entry = files.get(file_key)
        if file_entry is None:
            return None
        return self._normalize_file_entry(model_name, file_key, file_entry)

    def files(self, model_name: str) -> List[ModelFile]:
        entry = self.get(model_name) or {}
        files = entry.get("files") or {}
        if not isinstance(files, dict):
            return []
        results: List[ModelFile] = []
        for key, value in files.items():
            file_entry = self._normalize_file_entry(model_name, key, value)
            if file_entry:
                results.append(file_entry)
        return results

    def _normalize_file_entry(self, model_name: str, key: str, value: Any) -> Optional[ModelFile]:
        path_str: Optional[str] = None
        url: Optional[str] = None
        sha256: Optional[str] = None
        provider: Optional[str] = None
        model_ref: Optional[str] = None
        url_template: Optional[str] = None

        if isinstance(value, str):
            path_str = value
        elif isinstance(value, dict):
            path_str = value.get("path") or value.get("file")
            url = value.get("url")
            sha256 = self._normalize_sha256(value.get("sha256"))
            provider = value.get("provider")
            model_ref = value.get("model") or value.get("model_name")
            url_template = value.get("url_template")

        if not path_str:
            return None

        entry = self.get(model_name) or {}
        if not sha256:
            sha256 = self._normalize_sha256(self._lookup_hash(entry, key))
        if not url:
            url = self._lookup_url(entry, key)

        if not provider:
            provider = entry.get("provider") if isinstance(entry.get("provider"), str) else None
        if not model_ref:
            model_ref = entry.get("model") or entry.get("model_name")
        if not url_template and isinstance(entry.get("url_template"), str):
            url_template = entry.get("url_template")

        if not url:
            url = self._resolve_provider_url(provider, model_ref, url_template)

        path = self._resolve_path(path_str)
        return ModelFile(key=key, path=path, sha256=sha256, url=url)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        base = self.models_dir.parent
        return (base / path).resolve()

    def _lookup_hash(self, entry: Dict[str, Any], key: str) -> Optional[str]:
        hashes = entry.get("hashes")
        if isinstance(hashes, dict):
            if "sha256" in hashes and isinstance(hashes.get("sha256"), dict):
                return hashes["sha256"].get(key)
            return hashes.get(key)
        return None

    def _lookup_url(self, entry: Dict[str, Any], key: str) -> Optional[str]:
        urls = entry.get("urls")
        if isinstance(urls, dict):
            return urls.get(key)
        return None

    def _resolve_provider_url(
        self,
        provider_name: Optional[str],
        model_ref: Optional[str],
        url_template: Optional[str],
    ) -> Optional[str]:
        if not model_ref:
            return None

        base_url = None
        template = url_template
        if provider_name:
            provider = self.provider(provider_name)
            if provider:
                if not template:
                    template = provider.get("url_template")
                base_url = provider.get("base_url")

        if template:
            if "{base}" in template and base_url:
                return template.format(base=base_url, model=model_ref)
            return template.format(model=model_ref)
        if base_url:
            return f"{base_url.rstrip('/')}/{model_ref}"
        return None

    def _normalize_sha256(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        normalized = value.strip().lower()
        if normalized.startswith("sha256:"):
            normalized = normalized.split("sha256:", 1)[1]
        return normalized or None


_DEFAULT_REGISTRY = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _DEFAULT_REGISTRY
