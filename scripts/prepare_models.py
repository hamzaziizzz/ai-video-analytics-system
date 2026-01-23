import argparse
import sys

from src.core.utils.logging import setup_logging
from src.core.utils.model_assets import ensure_model_files, resolve_model_files
from src.core.utils.model_registry import ModelRegistry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/verify model assets")
    parser.add_argument("--model", action="append", help="Model name (repeatable)")
    parser.add_argument("--file", action="append", dest="files", help="File key (repeatable)")
    parser.add_argument("--download", action="store_true", help="Download missing files when URL is present")
    parser.add_argument("--verify", action="store_true", help="Verify files with SHA256 if available")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files when downloading")
    parser.add_argument("--timeout", type=float, default=60.0, help="Download timeout seconds")
    parser.add_argument("--models-dir", default="models", help="Path to models directory")
    parser.add_argument("--list", action="store_true", help="List registry entries and exit")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    return parser.parse_args()


def _list_registry(registry: ModelRegistry) -> None:
    provider_names = registry.provider_names()
    if provider_names:
        print("Providers:")
        for name in provider_names:
            provider = registry.provider(name) or {}
            base_url = provider.get("base_url", "")
            print(f"  {name}: {base_url}")

    model_names = registry.model_names()
    if model_names:
        print("Models:")
        for name in model_names:
            files = registry.files(name)
            keys = ", ".join(sorted({entry.key for entry in files}))
            print(f"  {name}: {keys}")


def main() -> None:
    args = _parse_args()
    setup_logging(args.log_level)
    registry = ModelRegistry(models_dir=args.models_dir)

    if args.list:
        _list_registry(registry)
        return

    model_names = args.model or registry.model_names()
    if not model_names:
        print("No models found in registry", file=sys.stderr)
        raise SystemExit(1)

    if not args.download and not args.verify:
        args.verify = True

    for model_name in model_names:
        if args.verify or args.download:
            ensure_model_files(
                model_name,
                file_keys=args.files,
                download=args.download,
                verify=args.verify,
                overwrite=args.overwrite,
                timeout_seconds=args.timeout,
                registry=registry,
            )
        else:
            resolved = resolve_model_files(model_name, file_keys=args.files, registry=registry)
            if not resolved:
                print(f"{model_name}: no files", file=sys.stderr)
            else:
                for entry in resolved:
                    print(f"{model_name}:{entry.key} -> {entry.path}")


if __name__ == "__main__":
    main()
