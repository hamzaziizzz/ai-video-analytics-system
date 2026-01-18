import argparse

import uvicorn

from .api.app import create_app
from .utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Video Analytics System")
    parser.add_argument("--config", default="configs/default.json", help="Path to JSON config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    app = create_app(config)
    uvicorn.run(app, host=config.app.host, port=config.app.port, log_level=config.app.log_level.lower())


if __name__ == "__main__":
    main()
