"""Structured (JSON) logging configuration for the triagem service.

Every log event is emitted as a single line of JSON, which is what
container log collectors (CloudWatch, Grafana Loki, etc.) expect: no
multi-line stack-trace surprises, one parseable object per line.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render each :class:`logging.LogRecord` as one line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as a single-line JSON object.

        Always includes ``timestamp``, ``level``, ``logger`` and
        ``message``; appends ``exc_info`` when an exception is attached.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger to emit one JSON object per event.

    Args:
        level: Log level name (e.g. ``"INFO"``, ``"DEBUG"``). Defaults to
            ``Settings.log_level`` when omitted.
    """
    from triagem.config import get_settings

    resolved_level = (level or get_settings().log_level).upper()

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)
