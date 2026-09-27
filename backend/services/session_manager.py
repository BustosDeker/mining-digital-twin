"""Gestor de sesiones de simulación en tiempo real.

Cada sesión envuelve un `DigitalTwinState` + `MineEvacuationModel` y expone
el ciclo de vida que el frontend controla (Iniciar/Pausar/Detener/Reiniciar/
Paso +1s, ver mockup de referencia): reproducción automática en tiempo real
(un paso de simulación cada `ABM_STEP_SECONDS` segundos reales) o avance
manual paso a paso. Cualquier WebSocket suscrito a la sesión recibe el
snapshot del gemelo digital cada vez que el estado cambia.

Este módulo es el único punto de la API con estado mutable en memoria
(las sesiones); todo lo demás en `api/` es sin estado.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from backend.digital_twin.graph_models import MineLayout
from backend.digital_twin.layout_generator import LayoutGenerator, LayoutGeneratorConfig
from backend.digital_twin.state import DigitalTwinState, HazardType
from backend.services.session_store import persist_session_record
from backend.simulation.model import MineEvacuationModel
from backend.simulation.routing import AdaptiveShortestPathRouter, QLearningRouter, Router
from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class SessionStatus(str, Enum):
    READY = "ready"  # creada, aún no iniciada
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FINISHED = "finished"  # terminó sola (todos los agentes en estado terminal)


ROUTER_FACTORIES: dict[str, Any] = {
    "adaptive_astar": lambda: AdaptiveShortestPathRouter(),
    "q_learning": lambda: QLearningRouter(),  # sin entrenar por defecto; ver nota en create_session
}


@dataclass
class SimulationSession:
    session_id: str
    layout: MineLayout
    state: DigitalTwinState
    model: MineEvacuationModel
    router_name: str
    status: SessionStatus = SessionStatus.READY
    scenario_name: str = "default"
    _subscribers: set[Any] = field(default_factory=set)  # WebSocket-like objetos con .send_json
    _playback_task: asyncio.Task | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        payload = self.state.snapshot()
        payload["session_id"] = self.session_id
        payload["status"] = self.status.value
        payload["router_name"] = self.router_name
        payload["scenario_name"] = self.scenario_name
        return payload

    async def broadcast(self) -> None:
        if not self._subscribers:
            return
        payload = self.snapshot()
        dead: list[Any] = []
        for ws in self._subscribers:
            try:
                await ws.send_json(payload)
            except Exception:  # conexión cerrada u otro error de transporte
                dead.append(ws)
        for ws in dead:
            self._subscribers.discard(ws)


class SessionManager:
    """Registro en memoria de todas las sesiones activas del proceso."""

    def __init__(self) -> None:
        self._sessions: dict[str, SimulationSession] = {}

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def create_session(
        self,
        scenario_name: str = "default",
        n_agents: int | None = None,
        router_name: str = "adaptive_astar",
        hazard_type: HazardType | None = HazardType.FIRE,
        hazard_intensity: float = 0.9,
        layout_config: LayoutGeneratorConfig | None = None,
    ) -> SimulationSession:
        settings = get_settings()
        layout = LayoutGenerator(layout_config or LayoutGeneratorConfig()).generate(scenario_name)
        state = DigitalTwinState(layout)

        if hazard_type is not None:
            from backend.digital_twin.graph_models import NodeType

            risk_nodes = [n.node_id for n in layout.nodes if n.node_type == NodeType.RISK_ZONE]
            if risk_nodes:
                state.spawn_hazard(hazard_type, origin_node_id=risk_nodes[0], intensity=hazard_intensity)

        if router_name not in ROUTER_FACTORIES:
            raise ValueError(f"Router desconocido: {router_name}. Disponibles: {list(ROUTER_FACTORIES)}")
        router: Router = ROUTER_FACTORIES[router_name]()

        model = MineEvacuationModel(state, n_agents=n_agents or settings.ABM_DEFAULT_N_AGENTS, path_provider=router)

        session_id = uuid.uuid4().hex[:12]
        session = SimulationSession(
            session_id=session_id,
            layout=layout,
            state=state,
            model=model,
            router_name=router_name,
            scenario_name=scenario_name,
        )
        self._sessions[session_id] = session
        logger.info("Sesión de simulación creada", extra={"session_id": session_id, "scenario": scenario_name})
        return session

    def get_session(self, session_id: str) -> SimulationSession:
        if session_id not in self._sessions:
            raise KeyError(f"Sesión no encontrada: {session_id}")
        return self._sessions[session_id]

    def list_sessions(self) -> list[SimulationSession]:
        return list(self._sessions.values())

    # ------------------------------------------------------------------
    # Control de reproducción (Iniciar/Pausar/Detener/Reiniciar/Paso)
    # ------------------------------------------------------------------
    async def start(self, session_id: str) -> SimulationSession:
        session = self.get_session(session_id)
        if session.status == SessionStatus.RUNNING:
            return session

        session.status = SessionStatus.RUNNING
        if session._playback_task is None or session._playback_task.done():
            session._playback_task = asyncio.create_task(self._playback_loop(session))
        return session

    def pause(self, session_id: str) -> SimulationSession:
        session = self.get_session(session_id)
        if session.status == SessionStatus.RUNNING:
            session.status = SessionStatus.PAUSED
        return session

    def stop(self, session_id: str) -> SimulationSession:
        session = self.get_session(session_id)
        session.status = SessionStatus.STOPPED
        persist_session_record(session.snapshot())
        return session

    def reset(self, session_id: str) -> SimulationSession:
        session = self.get_session(session_id)
        session.status = SessionStatus.READY
        session.state.reset_dynamic_state()
        # Se re-crea el modelo (nuevos agentes) conservando el mismo grafo/estado.
        settings = get_settings()
        router: Router = ROUTER_FACTORIES[session.router_name]()
        session.model = MineEvacuationModel(
            session.state, n_agents=len(session.model.schedule.agents), path_provider=router
        )
        return session

    async def step_once(self, session_id: str) -> SimulationSession:
        """Avanza exactamente un paso, sin importar el estado de reproducción
        (usado por el botón "Paso +1s" incluso si la sesión está pausada).
        """
        session = self.get_session(session_id)
        session.model.step()
        if not session.model.running:
            session.status = SessionStatus.FINISHED
            persist_session_record(session.snapshot())
        await session.broadcast()
        return session

    async def _playback_loop(self, session: SimulationSession) -> None:
        settings = get_settings()
        interval = max(0.05, session.model.step_seconds)
        try:
            while session.status == SessionStatus.RUNNING:
                session.model.step()
                await session.broadcast()
                if not session.model.running:
                    session.status = SessionStatus.FINISHED
                    persist_session_record(session.snapshot())
                    break
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # WebSocket subscription
    # ------------------------------------------------------------------
    def subscribe(self, session_id: str, websocket: Any) -> None:
        self.get_session(session_id)._subscribers.add(websocket)

    def unsubscribe(self, session_id: str, websocket: Any) -> None:
        session = self._sessions.get(session_id)
        if session:
            session._subscribers.discard(websocket)


# Instancia única del proceso (la API es single-process en esta fase; para
# despliegues multi-worker, el estado de sesión debería externalizarse a
# Redis u otro almacén compartido — documentado como trabajo futuro).
session_manager = SessionManager()
