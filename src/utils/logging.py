import logging
import logging.config
import os

# each container need to mount the logs to persist it
# volumes:
#       - ./logs:/app/logs        # host ./logs ← container writes here

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
        "frontend": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/frontend.log",
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
        },  # for pipeline
        "frontend": {"handlers": ["frontend", "console"], "level": "INFO"},
    },
}


def setup_logging():
    """Helper function to configure the logging config."""
    os.makedirs("logs", exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
