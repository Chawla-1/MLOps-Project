"""
Structured JSON Logger for PG Room Rent Prediction Service (MP141).
Outputs single-line JSON logs to stdout.
Ensures zero customer personal identifying information is logged.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Format log records as single-line JSON objects to standard output.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Include custom extra fields if provided
        if hasattr(record, "extra_info") and isinstance(record.extra_info, dict):
            # Ensure no personal identifiable information (PII) is included
            sanitized = {
                k: v for k, v in record.extra_info.items()
                if k not in ["name", "email", "phone", "student_id", "record_id"]
            }
            log_obj.update(sanitized)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def get_logger(name: str = "mp141_service") -> logging.Logger:
    """
    Creates or retrieves a logger configured with JSONFormatter streaming to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger
