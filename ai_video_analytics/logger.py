import logging
import os

import colorlog


def configure_logger(name: str):
    log_level = os.getenv("LOG_LEVEL", "INFO")

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.isatty = lambda: True

    formatter = colorlog.ColoredFormatter(
        "[{asctime}]{log_color}[{levelname:^8s}] ({filename}:{lineno} ({funcName})): {message}",
        style="{",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


logger = configure_logger(__name__)
