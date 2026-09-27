"""Estado del gemelo digital: fuente única de verdad.

`DigitalTwinState` envuelve el `MineLayout` (topología estática) junto con
todo lo que cambia en tiempo real: estado/riesgo dinámico de cada arista,
eventos de emergencia activos y el paso de simulación actual. Es el objeto
que la API serializa hacia el frontend (WebSocket) y que el ABM (Fase 3) y
el enrutamiento (Fase 4) consultan y mutan.

No se importa nada de `simulation/` aquí para evitar dependencias
circulares: este módulo solo conoce el grafo y el riesgo, no las reglas de
comportamiento de los agentes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.digital_twin.graph_models import EdgeStatus, MineEdge, MineLayout, MineNode
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class HazardType(str, Enum):
    FIRE = "fire"
    COLLAPSE = "collapse"
    GAS_LEAK = "gas_leak"


class HazardEvent(BaseModel):
    """Un evento de emergencia activo que se propaga sobre el grafo."""

    event_id: str
    hazard_type: HazardType
    origin_node_id: str
    started_at_step: int
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)
    affected_edges: set[str] = Field(default_factory=set)

    model_config = {"arbitrary_types_allowed": True}


class DigitalTwinState:
    """Estado mutable y en tiempo real del gemelo digital.

    Esta clase es intencionalmente la única con permiso para mutar
    `layout.edges[*].status` y `.current_risk`. Cualquier otro módulo debe
    pasar por sus métodos, nunca mutar `MineEdge` directamente, de modo que
    quede un único punto auditable de cambios de estado.
    """

    def __init__(self, layout: MineLayout) -> None:
        # Copia profunda obligatoria: sin esto, dos `DigitalTwinState`
        # construidos sobre el mismo `MineLayout` compartirían los mismos
        # objetos `MineEdge` (Pydantic, mutables), y las mutaciones de uno
        # (bloqueos, riesgo) se filtrarían silenciosamente al otro. Esto es
        # crítico para Monte Carlo y para el entrenamiento offline del
        # `QLearningRouter`, donde se instancian cientos de estados a partir
        # del mismo layout base y cada uno debe ser independiente.
        self.layout = layout.model_copy(deep=True)
        self._node_map: dict[str, MineNode] = self.layout.node_map()
        self._edge_map: dict[str, MineEdge] = self.layout.edge_map()
        self._edges_by_endpoints: dict[tuple[str, str], MineEdge] = self.layout.edges_by_endpoints()
        self.current_step: int = 0
        self.active_hazards: dict[str, HazardEvent] = {}
        self.agent_snapshots: dict[str, dict[str, Any]] = {}
        self._event_counter = 0

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------
    def get_node(self, node_id: str) -> MineNode:
        return self._node_map[node_id]

    def get_edge(self, edge_id: str) -> MineEdge:
        return self._edge_map[edge_id]

    def get_edge_between(self, node_a: str, node_b: str) -> MineEdge | None:
        return self._edges_by_endpoints.get((node_a, node_b))

    @property
    def blocked_edges(self) -> list[str]:
        return [e.edge_id for e in self._edge_map.values() if e.status == EdgeStatus.BLOCKED]

    @property
    def risk_by_edge(self) -> dict[str, float]:
        return {e.edge_id: e.current_risk for e in self._edge_map.values()}

    # ------------------------------------------------------------------
    # Mutaciones (único punto auditable de cambio de estado del grafo)
    # ------------------------------------------------------------------
    def set_edge_status(self, edge_id: str, status: EdgeStatus, risk: float | None = None) -> None:
        edge = self._edge_map[edge_id]
        previous_status = edge.status
        edge.status = status
        if risk is not None:
            edge.current_risk = max(0.0, min(1.0, risk))
        if previous_status != status:
            logger.info(
                "Cambio de estado de arista",
                extra={"edge_id": edge_id, "from": previous_status.value, "to": status.value},
            )

    def spawn_hazard(self, hazard_type: HazardType, origin_node_id: str, intensity: float = 1.0) -> HazardEvent:
        self._event_counter += 1
        event = HazardEvent(
            event_id=f"HZ{self._event_counter}",
            hazard_type=hazard_type,
            origin_node_id=origin_node_id,
            started_at_step=self.current_step,
            intensity=intensity,
        )
        self.active_hazards[event.event_id] = event
        logger.warning(
            "Nuevo evento de emergencia",
            extra={"event_id": event.event_id, "type": hazard_type.value, "origin": origin_node_id},
        )
        return event

    def clear_hazard(self, event_id: str) -> None:
        self.active_hazards.pop(event_id, None)

    def advance_step(self) -> None:
        self.current_step += 1

    def update_agent_snapshot(self, agent_id: str, snapshot: dict[str, Any]) -> None:
        self.agent_snapshots[agent_id] = snapshot

    def reset_dynamic_state(self) -> None:
        """Reinicia riesgo/estado dinámico y eventos, preservando la topología."""
        for edge in self._edge_map.values():
            edge.status = EdgeStatus.CLEAR
            edge.current_risk = edge.base_risk
        self.active_hazards.clear()
        self.agent_snapshots.clear()
        self.current_step = 0

    # ------------------------------------------------------------------
    # Serialización (contrato hacia frontend / cualquier cliente)
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Representación JSON-serializable completa del estado actual."""
        return {
            "step": self.current_step,
            "layout_id": self.layout.layout_id,
            "nodes": [n.model_dump() for n in self.layout.nodes],
            "edges": [e.model_dump() for e in self.layout.edges],
            "active_hazards": [h.model_dump(mode="json") for h in self.active_hazards.values()],
            "agents": self.agent_snapshots,
        }

    def graph_state_delta(self) -> dict[str, Any]:
        """Representación mínima usada para el contrato de estado en tiempo
        real (equivalente al antiguo campo `graph_state` del contrato Unity):
        solo aristas bloqueadas y riesgo por arista, para minimizar payload.
        """
        return {
            "blocked_edges": self.blocked_edges,
            "risk_level": self.risk_by_edge,
        }
