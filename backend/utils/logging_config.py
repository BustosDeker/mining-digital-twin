"""Logging estructurado para todo el backend.

Regla del proyecto: logging estructurado en todos los módulos. En desarrollo
se usa un formato humano-legible; en producción (`LOG_JSON=true`) se emite
JSON por línea, apto para agregadores (ELK, CloudWatch, etc.).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from backend.utils.config import get_settings


class JSONFormatter(logging.Formatter):
    """Formatea cada registro de log como una línea JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra_keys = set(record.__dict__) - _STANDARD_LOG_RECORD_KEYS
        for key in extra_keys:
            payload[key] = record.__dict__[key]
        return json.dumps(payload, ensure_ascii=False, default=str)


_STANDARD_LOG_RECORD_KEYS = set(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
    ).__dict__.keys()
)

_HUMAN_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def configure_logging() -> None:
    """Configura el logging raíz una sola vez para todo el proceso."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)

    # Evita handlers duplicados si algo más (uvicorn, pytest) ya configuró logging.
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if settings.LOG_JSON:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger configurado para el módulo llamante.

    Uso estándar en todo el proyecto:
        logger = get_logger(__name__)
    """
    configure_logging()
    return logging.getLogger(name)
