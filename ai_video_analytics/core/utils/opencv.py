import os

from .logging import get_logger


def configure_opencv_logging(suppress: bool) -> None:
    if not suppress:
        return

    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")
    os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")
    logger = get_logger("opencv")

    try:
        import cv2
    except ImportError:
        return

    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
            logger.info("OpenCV logging set to silent")
            return
    except Exception:
        pass

    try:
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)
            logger.info("OpenCV logging level set to 0")
    except Exception:
        pass
