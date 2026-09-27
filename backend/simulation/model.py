"""Modelo ABM de evacuación de emergencia (Mesa).

Orquesta agentes mineros, propagación de peligro y avance del tiempo de
simulación. Soporta dos modos de ejecución:

- `run_headless(max_steps)`: corre hasta que todos los agentes terminan
  (evacuados/refugiados/perdidos) o se alcanza `max_steps`; pensado para
  Monte Carlo (N corridas por escenario).
- `step()`: avanza un único paso; pensado para sincronía en tiempo real
  con el frontend vía WebSocket.

El enrutamiento usado por defecto en esta fase es un Dijkstra simple sobre
riesgo/longitud (ver `_default_path_provider`). La Fase 4 formaliza esto
como un `Router` intercambiable (Dijkstra/A* adaptativo vs. Q-learning) sin
requerir cambios en `MinerAgent` ni en este modelo, gracias a la interfaz
`PathProvider` ya definida en `simulation/agents.py`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import mesa
import networkx as nx

from backend.digital_twin.graph_models import NodeType
from backend.digital_twin.state import DigitalTwinState
from backend.simulation.agents import AgentStatus, MinerAgent, PathProvider
from backend.simulation.hazard_propagation import HazardPropagationEngine
from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


def _default_path_provider(state: DigitalTwinState, source_node_id: str) -> list[str] | None:
    """Dijkstra sobre peso = longitud_m * penalización_por_estado, evitando
    aristas bloqueadas. Provee ruta al destino seguro alcanzable más cercano
    (salida preferida; cámara de refugio como respaldo).
    """
    graph = nx.Graph()
    graph.add_nodes_from(n.node_id for n in state.layout.nodes)
    for edge in state.layout.edges:
        if not edge.is_traversable:
            continue
        weight = edge.length_m * edge.traversal_penalty * (1.0 + 2.0 * edge.current_risk)
        graph.add_edge(edge.source, edge.target, weight=weight)

    if source_node_id not in graph:
        return None

    exits = [n.node_id for n in state.layout.nodes if n.node_type == NodeType.EXIT]
    refuges = [n.node_id for n in state.layout.nodes if n.node_type == NodeType.REFUGE_CHAMBER]

    def _best_path_to(destinations: list[str]) -> list[str] | None:
        best_path: list[str] | None = None
        best_cost = float("inf")
        for destination in destinations:
            if destination not in graph:
                continue
            try:
                cost, path = nx.single_source_dijkstra(
                    graph, source_node_id, destination, weight="weight"
                )
            except nx.NetworkXNoPath:
                continue
            if cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path

    # Prioridad estricta: una salida real siempre es preferible a un refugio
    # temporal. Solo se enruta a un refugio cuando NINGUNA salida es
    # alcanzable dado el estado actual del grafo (aristas bloqueadas), lo
    # que hace inequívoca la decisión de refugio al llegar (ver
    # `MinerAgent._check_arrival`): si el agente llegó a un refugio es
    # porque era el único destino seguro alcanzable en ese momento.
    path_to_exit = _best_path_to(exits)
    if path_to_exit is not None:
        return path_to_exit
    return _best_path_to(refuges)


@dataclass
class EvacuationRunMetrics:
    """Métricas de una corrida completa, para Monte Carlo (Fase 7)."""

    total_agents: int
    evacuated: int = 0
    sheltered: int = 0
    lost: int = 0  # sin ninguna ruta posible en algún momento de la corrida
    timed_out: int = 0  # seguía en tránsito cuando se alcanzó max_steps
    steps_taken: int = 0
    evacuation_times_steps: list[float] = field(default_factory=list)
    bottleneck_max_waiting: int = 0

    @property
    def evacuation_rate(self) -> float:
        return self.evacuated / self.total_agents if self.total_agents else 0.0

    @property
    def not_evacuated_rate(self) -> float:
        return 1.0 - self.evacuation_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_agents": self.total_agents,
            "evacuated": self.evacuated,
            "sheltered": self.sheltered,
            "lost": self.lost,
            "timed_out": self.timed_out,
            "steps_taken": self.steps_taken,
            "evacuation_rate": round(self.evacuation_rate, 4),
            "not_evacuated_rate": round(self.not_evacuated_rate, 4),
            "mean_evacuation_time_steps": (
                round(sum(self.evacuation_times_steps) / len(self.evacuation_times_steps), 2)
                if self.evacuation_times_steps
                else None
            ),
            "bottleneck_max_waiting": self.bottleneck_max_waiting,
        }


class MineEvacuationModel(mesa.Model):
    """Modelo ABM completo: agentes + propagación de peligro sobre el gemelo digital."""

    def __init__(
        self,
        state: DigitalTwinState,
        n_agents: int | None = None,
        path_provider: PathProvider | None = None,
        step_seconds: float | None = None,
        random_seed: int | None = None,
        familiarity_ratio: float = 0.7,
    ) -> None:
        super().__init__()
        settings = get_settings()
        self.state = state
        self.step_seconds = step_seconds or settings.ABM_STEP_SECONDS
        self.path_provider = path_provider or _default_path_provider
        self.hazard_engine = HazardPropagationEngine(state)
        self.schedule = mesa.time.RandomActivation(self)
        self._rng = random.Random(random_seed if random_seed is not None else settings.RANDOM_SEED)

        n_agents = n_agents or settings.ABM_DEFAULT_N_AGENTS
        self._spawn_agents(n_agents, familiarity_ratio)

        # Snapshot inicial: el frontend debe poder renderizar a los agentes
        # en su posición de partida incluso antes del primer step().
        for agent in self.schedule.agents:
            if isinstance(agent, MinerAgent):
                self.state.update_agent_snapshot(str(agent.unique_id), agent.to_snapshot())

        self.running = True

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------
    def _spawn_agents(self, n_agents: int, familiarity_ratio: float) -> None:
        spawnable_nodes = [
            n.node_id
            for n in self.state.layout.nodes
            if n.node_type not in (NodeType.EXIT,)
        ]
        if not spawnable_nodes:
            raise ValueError("El layout no tiene nodos válidos donde ubicar agentes.")

        for i in range(n_agents):
            start_node = self._rng.choice(spawnable_nodes)
            is_familiar = self._rng.random() < familiarity_ratio
            agent = MinerAgent(
                unique_id=i,
                model=self,
                state=self.state,
                start_node_id=start_node,
                path_provider=self.path_provider,
                is_familiar_with_mine=is_familiar,
                rng_seed=self._rng.randint(0, 2**31 - 1),
            )
            self.schedule.add(agent)

    # ------------------------------------------------------------------
    # Ejecución paso a paso (uso en tiempo real / WebSocket)
    # ------------------------------------------------------------------
    def step(self) -> None:
        self.hazard_engine.step()
        self.schedule.step()
        self.state.advance_step()

        for agent in self.schedule.agents:
            if isinstance(agent, MinerAgent):
                self.state.update_agent_snapshot(str(agent.unique_id), agent.to_snapshot())

        if self._all_agents_terminal():
            self.running = False

    def _all_agents_terminal(self) -> bool:
        terminal = {AgentStatus.EVACUATED, AgentStatus.SHELTERED, AgentStatus.LOST}
        return all(
            agent.status in terminal
            for agent in self.schedule.agents
            if isinstance(agent, MinerAgent)
        )

    # ------------------------------------------------------------------
    # Ejecución headless (Monte Carlo, Fase 7)
    # ------------------------------------------------------------------
    def run_headless(self, max_steps: int | None = None) -> EvacuationRunMetrics:
        settings = get_settings()
        max_steps = max_steps or settings.ABM_MAX_STEPS

        evacuation_step_by_agent: dict[int, int] = {}

        step_count = 0
        while self.running and step_count < max_steps:
            self.step()
            step_count += 1
            for agent in self.schedule.agents:
                if (
                    isinstance(agent, MinerAgent)
                    and agent.status == AgentStatus.EVACUATED
                    and agent.unique_id not in evacuation_step_by_agent
                ):
                    evacuation_step_by_agent[agent.unique_id] = step_count

        miner_agents = [a for a in self.schedule.agents if isinstance(a, MinerAgent)]
        metrics = EvacuationRunMetrics(total_agents=len(miner_agents), steps_taken=step_count)
        for agent in miner_agents:
            if agent.status == AgentStatus.EVACUATED:
                metrics.evacuated += 1
            elif agent.status == AgentStatus.SHELTERED:
                metrics.sheltered += 1
            elif agent.status == AgentStatus.LOST:
                metrics.lost += 1
            else:  # MOVING o WAITING: seguía en tránsito al cortar la corrida
                metrics.timed_out += 1
        metrics.evacuation_times_steps = list(evacuation_step_by_agent.values())
        metrics.bottleneck_max_waiting = self._max_concurrent_waiting_agents()

        logger.info("Corrida ABM headless finalizada", extra=metrics.to_dict())
        return metrics

    def _max_concurrent_waiting_agents(self) -> int:
        return sum(
            1
            for a in self.schedule.agents
            if isinstance(a, MinerAgent) and a.status == AgentStatus.WAITING
        )
