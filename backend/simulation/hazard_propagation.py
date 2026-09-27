"""Propagación dinámica de eventos de emergencia sobre el grafo de la mina.

Un incendio, colapso o fuga de gas se origina en un nodo y se propaga por
las aristas adyacentes con el tiempo, degradando o bloqueando el tránsito
según la distancia (en saltos) al origen y los pasos transcurridos desde el
inicio del evento. Este módulo solo lee/escribe a través de
`DigitalTwinState`, nunca muta `MineEdge` directamente (regla de la Fase 2).
"""

from __future__ import annotations

import networkx as nx

from backend.digital_twin.graph_models import EdgeStatus
from backend.digital_twin.state import DigitalTwinState, HazardEvent, HazardType
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# Velocidad de propagación por tipo de peligro: saltos de grafo por paso de
# simulación que el frente de peligro avanza. Documentado como estimación
# de diseño (no hay dataset público de propagación de incendios en minas
# subterráneas de acceso abierto); ajustable por configuración de escenario.
#
# Calibración: un agente humano tarda típicamente 15-30 pasos (segundos) en
# cruzar un tramo de galería (15-40 m a ~1.2-2.5 m/s). Para que el escenario
# sea un ejercicio de evacuación real (y no un bloqueo instantáneo de toda
# la mina), el frente de peligro debe avanzar en una escala de tiempo
# comparable, no órdenes de magnitud más rápido.
_PROPAGATION_SPEED_HOPS_PER_STEP: dict[HazardType, float] = {
    HazardType.FIRE: 0.035,
    HazardType.GAS_LEAK: 0.05,  # el gas se difunde más rápido que el frente de fuego
    HazardType.COLLAPSE: 0.015,  # un colapso secundario se extiende más lento
}

# Umbrales de distancia (en saltos efectivos) que determinan el estado de
# una arista según qué tan avanzado está el frente de propagación sobre ella.
_DEGRADE_THRESHOLD_HOPS = 1.0
_BLOCK_THRESHOLD_HOPS = 0.35


class HazardPropagationEngine:
    """Aplica, en cada paso, el efecto de todos los eventos activos sobre el grafo."""

    def __init__(self, state: DigitalTwinState) -> None:
        self.state = state
        self._graph_cache: nx.Graph | None = None

    def _build_graph(self) -> nx.Graph:
        if self._graph_cache is not None:
            return self._graph_cache
        g = nx.Graph()
        g.add_nodes_from(n.node_id for n in self.state.layout.nodes)
        for edge in self.state.layout.edges:
            g.add_edge(edge.source, edge.target, edge_id=edge.edge_id, weight=1.0)
        self._graph_cache = g
        return g

    def step(self) -> None:
        """Recalcula el efecto de todos los eventos activos sobre las aristas."""
        if not self.state.active_hazards:
            return

        graph = self._build_graph()
        for event in self.state.active_hazards.values():
            self._apply_event(graph, event)

    def _apply_event(self, graph: nx.Graph, event: HazardEvent) -> None:
        steps_elapsed = max(0, self.state.current_step - event.started_at_step)
        speed = _PROPAGATION_SPEED_HOPS_PER_STEP[event.hazard_type]
        frontier_hops = steps_elapsed * speed * (0.5 + 0.5 * event.intensity)

        try:
            distances = nx.single_source_shortest_path_length(graph, event.origin_node_id)
        except nx.NodeNotFound:
            logger.error("Nodo origen de evento no existe en el grafo", extra={"event_id": event.event_id})
            return

        affected: set[str] = set()
        for node_id, hop_distance in distances.items():
            if hop_distance == 0:
                continue
            remaining = frontier_hops - hop_distance
            if remaining <= -_DEGRADE_THRESHOLD_HOPS:
                continue  # aún no llega el frente a este nodo

            for neighbor in graph.neighbors(node_id):
                edge_data = graph.get_edge_data(node_id, neighbor)
                edge_id = edge_data["edge_id"]
                edge = self.state.get_edge(edge_id)
                if edge.status == EdgeStatus.BLOCKED and remaining < _BLOCK_THRESHOLD_HOPS:
                    continue  # no reabrir aristas ya bloqueadas por debajo del umbral

                if remaining >= _BLOCK_THRESHOLD_HOPS:
                    new_status = EdgeStatus.BLOCKED
                    new_risk = min(1.0, 0.7 + 0.3 * event.intensity)
                elif remaining >= -_DEGRADE_THRESHOLD_HOPS:
                    new_status = EdgeStatus.DEGRADED
                    new_risk = min(1.0, 0.35 + 0.3 * event.intensity)
                else:
                    continue

                # No degradar una arista que ya está en un estado más severo
                # por otro evento superpuesto.
                if _severity_rank(new_status) >= _severity_rank(edge.status):
                    self.state.set_edge_status(edge_id, new_status, risk=max(new_risk, edge.current_risk))
                affected.add(edge_id)

        event.affected_edges |= affected


def _severity_rank(status: EdgeStatus) -> int:
    return {EdgeStatus.CLEAR: 0, EdgeStatus.DEGRADED: 1, EdgeStatus.BLOCKED: 2}[status]
