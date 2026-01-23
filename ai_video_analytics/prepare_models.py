import logging

from ai_video_analytics.core.model_zoo.yolo import prepare_yolo_inference_assets
from ai_video_analytics.core.settings import Settings, build_default_config
from ai_video_analytics.core.utils.logging import get_logger, setup_logging


def prepare_models() -> None:
    settings = Settings()
    config = build_default_config(settings)
    setup_logging(config.app.log_level)
    logging.basicConfig(
        level=config.app.log_level,
        format="%(asctime)s %(levelname)s - %(message)s",
        datefmt="[%H:%M:%S]",
    )
    logger = get_logger("models.prepare")
    logger.info("Preparing YOLO assets...")
    prepare_yolo_inference_assets(config)
    logger.info("Model assets ready.")


if __name__ == "__main__":
    prepare_models()
