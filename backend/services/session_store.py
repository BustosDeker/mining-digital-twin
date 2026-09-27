"""Historial de sesiones: persistencia y consulta.

Cada sesión terminada (detenida o finalizada sola) se guarda como JSON en
`settings.DATA_SESSIONS_DIR`. Este es también el dataset propio del
proyecto (Componente de "dataset generado por el sistema"): agregando aquí
las corridas ABM puras y, en el futuro, las sesiones con intervención
humana real, se construye el dataset primario del artículo.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


def persist_session_record(snapshot: dict[str, Any]) -> Path:
    settings = get_settings()
    session_id = snapshot["session_id"]
    record = {
        **snapshot,
        "persisted_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = settings.DATA_SESSIONS_DIR / f"{session_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, default=str)
    logger.info("Sesión persistida", extra={"session_id": session_id, "path": str(out_path)})
    return out_path


def list_session_records() -> list[dict[str, Any]]:
    settings = get_settings()
    records = []
    for path in sorted(settings.DATA_SESSIONS_DIR.glob("*.json"), reverse=True):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        records.append(
            {
                "session_id": data.get("session_id"),
                "scenario_name": data.get("scenario_name"),
                "router_name": data.get("router_name"),
                "status": data.get("status"),
                "step": data.get("step"),
                "persisted_at": data.get("persisted_at"),
            }
        )
    return records


def get_session_record(session_id: str) -> dict[str, Any]:
    settings = get_settings()
    path = settings.DATA_SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No hay historial persistido para la sesión {session_id}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
