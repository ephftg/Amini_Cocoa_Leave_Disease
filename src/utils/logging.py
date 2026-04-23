import logging
import logging.config
import os

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"}
    },
    "handlers": {
        "pipeline": {
            "class": "logging.handlers.RotatingFileHandler",  # when max mb of logs added, will write to next file
            "filename": "logs/pipeline.log",
            "formatter": "standard",
            "maxBytes": 5_000_000,  # 5 MB per file
            "backupCount": 3,  # keep 3 old files
        },
        "inference": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/inference.log",
            "formatter": "standard",
            "maxBytes": 5_000_000,
            "backupCount": 3,
        },
        "console": {
            "class": "logging.StreamHandler",  # print to console
            "formatter": "standard",
        },
    },
    "loggers": {
        "pipeline": {
            "handlers": ["pipeline", "console"],
            "level": "INFO",
            "propagate": False,
        },  # for pipeline
        "inference": {
            "handlers": ["inference", "console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def setup_logging():
    """Helper function to configure the logging config."""
    os.makedirs("logs", exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
