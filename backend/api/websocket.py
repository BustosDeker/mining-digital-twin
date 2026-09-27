"""WebSocket de estado del gemelo digital en tiempo real.

Contrato: al conectar, el cliente recibe inmediatamente el snapshot actual
completo (`DigitalTwinState.snapshot()` + metadatos de sesión). Mientras la
sesión esté en estado `running`, recibe un nuevo snapshot cada vez que el
modelo avanza un paso (ver `SessionManager._playback_loop`); si está
`paused`, solo recibe actualizaciones cuando alguien llama a `/step`.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.session_manager import session_manager
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/simulations/{session_id}")
async def simulation_state_websocket(websocket: WebSocket, session_id: str) -> None:
    try:
        session = session_manager.get_session(session_id)
    except KeyError:
        await websocket.close(code=4004, reason=f"Sesión no encontrada: {session_id}")
        return

    await websocket.accept()
    session_manager.subscribe(session_id, websocket)
    logger.info("Cliente WebSocket conectado", extra={"session_id": session_id})

    try:
        await websocket.send_json(session.snapshot())
        while True:
            # Mantiene viva la conexión y permite comandos futuros del
            # cliente por el mismo canal (no usado en esta fase; el control
            # va por REST). Si el cliente cierra, se lanza WebSocketDisconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("Cliente WebSocket desconectado", extra={"session_id": session_id})
    finally:
        session_manager.unsubscribe(session_id, websocket)
