"""Dependency-free structured (JSON) logging for the API, worker, and beat.

Wire it up before starting a process:

    from .log import configure_json_logging
    configure_json_logging()

Each emit becomes one JSON line on stdout: ``{"ts", "level", "logger",
"message"}`` plus, when present, ``path``/``method``/``status_code``,
``exc`` (traceback text), and ``event`` (for heartbeat-style records).
"""

import json
import logging
import sys
from datetime import datetime, timezone

_EXTRA_FIELDS = (
    "path",
    "method",
    "status_code",
    "elapsed_ms",
    "event",
    "task_id",
    "pid",
)

_APP_LOGGER_NAMES = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "uvicorn.asgi",
    "celery",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_json_logging(level: int = logging.INFO) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    for name in _APP_LOGGER_NAMES:
        logger = logging.getLogger(name)
        # kombu installs a NullHandler on 'celery' at import; Celery treats any
        # pre-existing handler as "logging already configured" and skips its own
        # console handler. Drop the no-op so ours is the only one in place when
        # celery.main() runs.
        logger.handlers = [
            h for h in logger.handlers if not isinstance(h, logging.NullHandler)
        ]
        logger.handlers.append(handler)
        logger.propagate = False
        logger.setLevel(level)

    return handler