"""Motor de enrutamiento dinámico: interfaz común + dos estrategias.

Regla del proyecto: se implementan y comparan cuantitativamente dos
estrategias de enrutamiento bajo una interfaz común (`Router`), para que el
Motor IA de evaluación (Fase 7) pueda medirlas sin acoplarse a ninguna:

1. `AdaptiveShortestPathRouter` — A* sobre pesos dinámicos de riesgo, con
   heurística de distancia euclídea (admisible, ya que la distancia recta
   nunca sobreestima la distancia real por galerías).
2. `QLearningRouter` — política aprendida por Q-learning tabular que evita
   zonas de alto riesgo histórico, entrenada offline sobre corridas Monte
   Carlo con distintos orígenes de incendio (ver `train_offline`).

Ambas exponen el mismo método `__call__(state, source_node_id) -> path`,
compatible con el `PathProvider` que ya consume `MinerAgent` (Fase 3), de
modo que sustituir la estrategia del `MineEvacuationModel` no requiere
ningún cambio en el agente ni en el modelo.
"""

from __future__ import annotations

import logging
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import networkx as nx

from backend.digital_twin.graph_models import NodeType
from backend.digital_twin.state import DigitalTwinState
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class Router(Protocol):
    """Interfaz común de enrutamiento (compatible con `PathProvider`)."""

    def __call__(self, state: DigitalTwinState, source_node_id: str) -> list[str] | None:
        ...


def _build_navigable_graph(state: DigitalTwinState) -> nx.Graph:
    """Grafo NetworkX con solo aristas transitables, pesadas por riesgo."""
    graph = nx.Graph()
    graph.add_nodes_from(n.node_id for n in state.layout.nodes)
    for edge in state.layout.edges:
        if not edge.is_traversable:
            continue
        weight = edge.length_m * edge.traversal_penalty * (1.0 + 2.0 * edge.current_risk)
        graph.add_edge(edge.source, edge.target, weight=weight)
    return graph


def _safe_destinations(state: DigitalTwinState, prefer_exits_only: bool) -> tuple[list[str], list[str]]:
    exits = [n.node_id for n in state.layout.nodes if n.node_type == NodeType.EXIT]
    refuges = [] if prefer_exits_only else [
        n.node_id for n in state.layout.nodes if n.node_type == NodeType.REFUGE_CHAMBER
    ]
    return exits, refuges


# ==========================================================================
# Estrategia 1 — Ruta más corta adaptativa (A* con heurística euclídea)
# ==========================================================================
class AdaptiveShortestPathRouter:
    """A* sobre pesos dinámicos de riesgo; recalcula en cada llamada, por lo
    que reacciona de inmediato a cualquier cambio del grafo (arista
    bloqueada/degradada). Estrategia reactiva, no predictiva.
    """

    name = "adaptive_shortest_path"

    def __call__(self, state: DigitalTwinState, source_node_id: str) -> list[str] | None:
        graph = _build_navigable_graph(state)
        if source_node_id not in graph:
            return None

        positions = {n.node_id: n.position for n in state.layout.nodes}

        def heuristic(a: str, b: str) -> float:
            return math.dist(positions[a], positions[b])

        exits, refuges = _safe_destinations(state, prefer_exits_only=False)

        path = self._best_path(graph, source_node_id, exits, heuristic)
        if path is not None:
            return path
        return self._best_path(graph, source_node_id, refuges, heuristic)

    @staticmethod
    def _best_path(
        graph: nx.Graph, source: str, destinations: list[str], heuristic
    ) -> list[str] | None:
        best_path: list[str] | None = None
        best_cost = float("inf")
        for destination in destinations:
            if destination not in graph:
                continue
            try:
                path = nx.astar_path(graph, source, destination, heuristic=heuristic, weight="weight")
                cost = nx.path_weight(graph, path, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path


# ==========================================================================
# Estrategia 2 — Ruta aprendida (Q-learning tabular)
# ==========================================================================
class QLearningRouter:
    """Política aprendida que evita zonas de alto riesgo histórico.

    El estado es el `node_id` actual; las acciones son los nodos vecinos
    alcanzables. La recompensa penaliza fuertemente pasar por aristas de
    alto riesgo histórico (`base_risk`) y bloqueos, y premia acercarse a un
    destino seguro. Se entrena offline (`train_offline`) sobre corridas con
    distintos orígenes de incendio para capturar el riesgo *histórico*
    (no solo el estado instantáneo, a diferencia del router adaptativo).
    """

    name = "q_learning"

    def __init__(
        self,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self._rng = random.Random(random_seed)
        self.q_table: dict[str, dict[str, float]] = defaultdict(dict)
        self._is_trained = False

    # ------------------------------------------------------------------
    # Entrenamiento offline
    # ------------------------------------------------------------------
    def train_offline(
        self,
        layout,
        hazard_origin_node_ids: list[str],
        n_episodes_per_origin: int = 200,
        max_steps_per_episode: int = 60,
    ) -> None:
        """Entrena la Q-table simulando episodios de un único agente que
        navega desde nodos aleatorios hacia una salida, con distintos
        orígenes de incendio activos, para aprender qué aristas conviene
        evitar de forma sistemática (riesgo histórico), no solo reactiva.
        """
        from backend.digital_twin.state import DigitalTwinState, HazardType  # import diferido: evita ciclo
        from backend.simulation.hazard_propagation import HazardPropagationEngine

        exits = [n.node_id for n in layout.nodes if n.node_type == NodeType.EXIT]
        if not exits:
            raise ValueError("El layout no tiene salidas; no se puede entrenar el router.")

        all_node_ids = [n.node_id for n in layout.nodes]

        # Cientos de episodios generarían miles de líneas de log (spawn de
        # eventos, cambios de estado de arista) que no aportan valor durante
        # el entrenamiento offline; se silencian temporalmente y se
        # restauran al finalizar.
        state_logger = logging.getLogger("backend.digital_twin.state")
        previous_level = state_logger.level
        state_logger.setLevel(logging.ERROR)

        for origin in hazard_origin_node_ids:
            for _episode in range(n_episodes_per_origin):
                state = DigitalTwinState(layout)
                state.spawn_hazard(HazardType.FIRE, origin_node_id=origin, intensity=0.9)
                hazard_engine = HazardPropagationEngine(state)

                current_node = self._rng.choice(all_node_ids)
                for _step in range(max_steps_per_episode):
                    hazard_engine.step()
                    state.advance_step()

                    neighbors = self._navigable_neighbors(state, current_node)
                    if not neighbors:
                        break

                    action = self._choose_action(current_node, neighbors, explore=True)
                    reward = self._reward(state, current_node, action, exits)
                    next_neighbors = self._navigable_neighbors(state, action)
                    best_next_q = max(
                        (self.q_table[action].get(n, 0.0) for n in next_neighbors), default=0.0
                    )

                    old_q = self.q_table[current_node].get(action, 0.0)
                    self.q_table[current_node][action] = old_q + self.learning_rate * (
                        reward + self.discount_factor * best_next_q - old_q
                    )

                    current_node = action
                    if current_node in exits:
                        break

        state_logger.setLevel(previous_level)

        self._is_trained = True
        logger.info(
            "QLearningRouter entrenado",
            extra={
                "n_origins": len(hazard_origin_node_ids),
                "n_episodes_total": len(hazard_origin_node_ids) * n_episodes_per_origin,
                "q_table_states": len(self.q_table),
            },
        )

    def _navigable_neighbors(self, state: DigitalTwinState, node_id: str) -> list[str]:
        neighbors = []
        for edge in state.layout.edges:
            if not edge.is_traversable:
                continue
            if edge.source == node_id:
                neighbors.append(edge.target)
            elif edge.target == node_id:
                neighbors.append(edge.source)
        return neighbors

    def _choose_action(self, node_id: str, neighbors: list[str], explore: bool) -> str:
        if explore and self._rng.random() < self.exploration_rate:
            return self._rng.choice(neighbors)
        known = self.q_table[node_id]
        if not known:
            return self._rng.choice(neighbors)
        return max(neighbors, key=lambda n: known.get(n, 0.0))

    def _reward(
        self, state: DigitalTwinState, from_node: str, to_node: str, exits: list[str]
    ) -> float:
        edge = state.get_edge_between(from_node, to_node)
        if edge is None:
            return -10.0
        reward = -1.0 - 5.0 * edge.current_risk - 3.0 * edge.base_risk
        if to_node in exits:
            reward += 100.0
        return reward

    # ------------------------------------------------------------------
    # Persistencia (Model Registry, Fase 6)
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q_table": dict(self.q_table), "is_trained": self._is_trained}, f)

    @classmethod
    def load(cls, path: Path) -> "QLearningRouter":
        router = cls()
        with open(path, "rb") as f:
            payload = pickle.load(f)
        router.q_table = defaultdict(dict, payload["q_table"])
        router._is_trained = payload["is_trained"]
        return router

    # ------------------------------------------------------------------
    # Uso en producción (interfaz Router / PathProvider)
    # ------------------------------------------------------------------
    def __call__(self, state: DigitalTwinState, source_node_id: str) -> list[str] | None:
        exits = {n.node_id for n in state.layout.nodes if n.node_type == NodeType.EXIT}
        refuges = {n.node_id for n in state.layout.nodes if n.node_type == NodeType.REFUGE_CHAMBER}
        destinations = exits | refuges

        path = [source_node_id]
        current = source_node_id
        visited = {current}

        for _ in range(200):  # límite de saltos para evitar bucles infinitos
            if current in destinations:
                return path
            neighbors = [
                n for n in self._navigable_neighbors(state, current) if n not in visited
            ]
            if not neighbors:
                break
            action = self._choose_action(current, neighbors, explore=False)
            path.append(action)
            visited.add(action)
            current = action

        # Si la política aprendida no encuentra salida (grafo cambió tras el
        # entrenamiento), recurre al router adaptativo como respaldo seguro.
        fallback = AdaptiveShortestPathRouter()
        return fallback(state, source_node_id)
