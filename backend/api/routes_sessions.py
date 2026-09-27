"""Rutas REST de historial de sesiones (persistidas al detenerse/finalizar)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.services import session_store

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def list_sessions() -> list[dict]:
    return session_store.list_session_records()


@router.get("/{session_id}")
def get_session(session_id: str) -> dict:
    try:
        return session_store.get_session_record(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
