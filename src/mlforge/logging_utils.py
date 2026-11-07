from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_ROOT_NAME = "mlforge"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level.upper())
    formatter = logging.Formatter(_LOG_FORMAT)

    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        target = str(log_file.resolve())
        already = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == target
            for h in logger.handlers
        )
        if not already:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    if name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")