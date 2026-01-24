import copy
import logging
from pathlib import Path

from ai_video_analytics.core.model_zoo.yolo import prepare_yolo_inference_assets
from ai_video_analytics.core.settings import Settings as CoreSettings, build_default_config
from ai_video_analytics.core.utils.logging import get_logger, setup_logging
from ai_video_analytics.settings import Settings as RuntimeSettings


def prepare_models() -> None:
    core_settings = CoreSettings()
    config = build_default_config(core_settings)
    setup_logging(config.app.log_level)
    logging.basicConfig(
        level=config.app.log_level,
        format="%(asctime)s %(levelname)s - %(message)s",
        datefmt="[%H:%M:%S]",
    )
    logger = get_logger("models.prepare")
    logger.info("Preparing YOLO assets...")
    prepare_yolo_inference_assets(config)

    runtime_settings = RuntimeSettings()
    pose_name = runtime_settings.models.pose_name
    if (
        runtime_settings.models.pose_detection
        and pose_name
        and str(pose_name).strip().lower() not in {"none", "off", "false", "0"}
    ):
        pose_config = copy.deepcopy(config)
        pose_config.models.detection_model = pose_name
        pose_path = Path(str(pose_name))
        if pose_path.suffix or str(pose_name).strip().startswith(("/", ".")):
            pose_config.inference.model_path = str(pose_path)
        logger.info("Preparing YOLO pose assets...")
        prepare_yolo_inference_assets(pose_config)
    logger.info("Model assets ready.")


if __name__ == "__main__":
    prepare_models()
