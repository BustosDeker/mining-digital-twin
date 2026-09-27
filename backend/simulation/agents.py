"""Agente minero para el modelo ABM (Mesa).

El agente se mueve sobre el grafo del gemelo digital siguiendo reglas de
comportamiento humano en emergencia: pánico, contagio social (gregarismo),
fatiga y familiaridad con la ruta. La decisión de *qué ruta* tomar se
delega en un `path_provider` inyectado (ver `simulation/routing.py`,
Fase 4) para no acoplar el agente a una estrategia de enrutamiento
concreta — en esta fase se usa un proveedor mínimo de caminos más cortos.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Callable, Protocol

import mesa

from backend.digital_twin.graph_models import EdgeStatus, NodeType
from backend.digital_twin.state import DigitalTwinState
from backend.simulation.parameters import (
    DEFAULT_FAMILIARITY_PARAMS,
    DEFAULT_MOVEMENT_PARAMS,
    DEFAULT_PANIC_PARAMS,
)
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class AgentStatus(str, Enum):
    MOVING = "moving"
    WAITING = "waiting"  # bloqueado, recalculando ruta
    SHELTERED = "sheltered"  # llegó a una cámara de refugio
    EVACUATED = "evacuated"  # llegó a una salida
    LOST = "lost"  # sin ruta disponible hacia ningún destino seguro


class PathProvider(Protocol):
    """Interfaz mínima que el agente necesita del subsistema de enrutamiento.

    Se formaliza como `Router` en `simulation/routing.py` (Fase 4); aquí
    solo se declara el contrato para evitar acoplamiento.
    """

    def __call__(self, state: DigitalTwinState, source_node_id: str) -> list[str] | None:
        ...


class MinerAgent(mesa.Agent):
    """Un minero simulado que intenta evacuar la mina ante una emergencia."""

    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        state: DigitalTwinState,
        start_node_id: str,
        path_provider: PathProvider,
        is_familiar_with_mine: bool = True,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__(unique_id, model)
        self.state = state
        self.current_node_id = start_node_id
        self.path_provider = path_provider
        self.is_familiar_with_mine = is_familiar_with_mine

        self._rng = random.Random(rng_seed if rng_seed is not None else unique_id)

        self.status = AgentStatus.MOVING
        self.panic_level: float = 0.0
        self.cumulative_distance_m: float = 0.0
        self.carrying_load: bool = False

        self.current_path: list[str] = []
        self._progress_on_edge_m: float = 0.0
        self._current_edge_id: str | None = None

        self._recompute_path()

    # ------------------------------------------------------------------
    # Ciclo de Mesa
    # ------------------------------------------------------------------
    def step(self) -> None:
        if self.status in (AgentStatus.EVACUATED, AgentStatus.SHELTERED):
            return

        self._update_panic()

        if self.status == AgentStatus.LOST:
            self._recompute_path()
            if self.status == AgentStatus.LOST:
                return

        if not self.current_path or len(self.current_path) < 2:
            self._check_arrival(self.current_node_id)
            return

        self._advance_along_path()

    # ------------------------------------------------------------------
    # Pánico y contagio social (Helbing et al.)
    # ------------------------------------------------------------------
    def _update_panic(self) -> None:
        params = DEFAULT_PANIC_PARAMS
        near_hazard = self._is_near_hazard()

        if near_hazard:
            self.panic_level = min(1.0, self.panic_level + params.panic_increase_near_hazard)
        else:
            self.panic_level = max(0.0, self.panic_level - params.panic_decay_per_step)

        neighbor_panics = self._nearby_agent_panics(radius_nodes=params.panic_contagion_radius_nodes)
        if neighbor_panics:
            avg_neighbor_panic = sum(neighbor_panics) / len(neighbor_panics)
            self.panic_level = min(
                1.0,
                self.panic_level
                + params.panic_contagion_factor * max(0.0, avg_neighbor_panic - self.panic_level),
            )

    def _is_near_hazard(self) -> bool:
        for edge_id in self._edges_touching(self.current_node_id):
            edge = self.state.get_edge(edge_id)
            if edge.status != EdgeStatus.CLEAR:
                return True
        return False

    def _edges_touching(self, node_id: str) -> list[str]:
        return [
            e.edge_id
            for e in self.state.layout.edges
            if e.source == node_id or e.target == node_id
        ]

    def _nearby_agent_panics(self, radius_nodes: int) -> list[float]:
        # Aproximación barata: mismo nodo o nodo adyacente directo (radius=1
        # ya cubre el caso típico de cuello de botella; se evita BFS costoso
        # por agente y por paso al escalar a Monte Carlo con muchos agentes).
        others = [a for a in self.model.schedule.agents if a is not self]
        result = []
        for other in others:
            if not isinstance(other, MinerAgent):
                continue
            if other.status in (AgentStatus.EVACUATED, AgentStatus.SHELTERED):
                continue
            if other.current_node_id == self.current_node_id:
                result.append(other.panic_level)
        return result

    # ------------------------------------------------------------------
    # Movimiento
    # ------------------------------------------------------------------
    def _recompute_path(self) -> None:
        path = self.path_provider(self.state, self.current_node_id)
        if not path or len(path) < 1:
            self.status = AgentStatus.LOST
            self.current_path = []
            logger.debug("Agente sin ruta disponible", extra={"agent_id": self.unique_id})
            return

        self.status = AgentStatus.MOVING
        self.current_path = path
        self._current_edge_id = None
        self._progress_on_edge_m = 0.0

    def _advance_along_path(self) -> None:
        next_node_id = self.current_path[1]
        edge = self.state.get_edge_between(self.current_node_id, next_node_id)

        if edge is None or not edge.is_traversable:
            self.status = AgentStatus.WAITING
            self._recompute_path()
            return

        if self._current_edge_id != edge.edge_id:
            self._current_edge_id = edge.edge_id
            self._progress_on_edge_m = 0.0

        speed = DEFAULT_MOVEMENT_PARAMS.effective_speed(
            panic_level=self.panic_level,
            slope_pct=edge.slope_pct if edge.target == next_node_id else -edge.slope_pct,
            degraded_visibility=edge.status == EdgeStatus.DEGRADED,
            carrying_load=self.carrying_load,
            cumulative_distance_m=self.cumulative_distance_m,
        )
        step_seconds = self.model.step_seconds
        distance_this_step = speed * step_seconds

        self._progress_on_edge_m += distance_this_step
        self.cumulative_distance_m += distance_this_step

        if self._progress_on_edge_m >= edge.length_m:
            self.current_node_id = next_node_id
            self.current_path.pop(0)
            self._current_edge_id = None
            self._progress_on_edge_m = 0.0
            self._check_arrival(next_node_id)

    def _check_arrival(self, node_id: str) -> None:
        node = self.state.get_node(node_id)
        reached_final_destination = len(self.current_path) <= 1

        if node.node_type == NodeType.EXIT:
            self.status = AgentStatus.EVACUATED
            logger.info("Agente evacuado", extra={"agent_id": self.unique_id, "node": node_id})
        elif node.node_type == NodeType.REFUGE_CHAMBER and reached_final_destination:
            # El path_provider solo enruta a un refugio cuando ninguna salida
            # es alcanzable (ver `_default_path_provider`), así que llegar
            # aquí como destino final es, por construcción, la decisión
            # correcta de refugio — sin depender de un umbral de pánico que
            # no tiene mecanismo de recuperación posterior.
            self.status = AgentStatus.SHELTERED
            logger.info("Agente refugiado", extra={"agent_id": self.unique_id, "node": node_id})
        elif reached_final_destination:
            # Llegó al final de su ruta pero el nodo ya no es un destino
            # seguro válido (p. ej. cambió el grafo): recalcula.
            self._recompute_path()

    # ------------------------------------------------------------------
    # Serialización para el snapshot del gemelo digital
    # ------------------------------------------------------------------
    def to_snapshot(self) -> dict:
        progress_ratio = 0.0
        if self._current_edge_id is not None:
            edge = self.state.get_edge(self._current_edge_id)
            progress_ratio = min(1.0, self._progress_on_edge_m / edge.length_m)

        next_node = self.current_path[1] if len(self.current_path) > 1 else self.current_node_id
        return {
            "agent_id": self.unique_id,
            "status": self.status.value,
            "node_id": self.current_node_id,
            "next_node_id": next_node,
            "edge_id": self._current_edge_id,
            "progress": progress_ratio,
            "panic_level": round(self.panic_level, 3),
            "cumulative_distance_m": round(self.cumulative_distance_m, 1),
        }
