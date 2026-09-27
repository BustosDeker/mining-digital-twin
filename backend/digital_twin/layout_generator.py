"""Generador procedural de layouts sintéticos de mina subterránea.

No contamos con un layout real anonimizado, por lo que este módulo genera
layouts sintéticos parametrizables, documentados explícitamente como tales
(`MineLayout.source == "synthetic_procedural"`), tal como exige la regla del
proyecto de declarar la procedencia de la geometría.

Algoritmo (resumen):
1. Se generan `n_levels` niveles apilados en profundidad (eje Z negativo).
2. En cada nivel se crece un árbol de galerías por random-walk ramificado
   desde un punto de entrada de nivel, con intersecciones donde el árbol
   ramifica.
3. Los niveles se conectan entre sí por rampas (aristas con pendiente).
4. Se ubican cámaras de refugio en extremos de ramificaciones y salidas en
   el nivel más superficial.
5. Se valida con NetworkX que todo nodo tenga camino hacia al menos una
   salida (grafo funcionalmente evacuable); si no, se repara conectando
   componentes sueltas al nodo más cercano.
"""

from __future__ import annotations

import random

import networkx as nx

from backend.digital_twin.graph_models import EdgeStatus, MineEdge, MineLayout, MineNode, NodeType
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class LayoutGeneratorConfig:
    """Parámetros del generador. Todos con valores por defecto razonables,
    pero pensados para ser variados por escenario (no hardcoded en el algoritmo).
    """

    def __init__(
        self,
        n_levels: int = 3,
        galleries_per_level: int = 5,
        max_branch_depth: int = 4,
        branch_probability: float = 0.45,
        gallery_segment_length_m: tuple[float, float] = (15.0, 40.0),
        level_height_m: float = 25.0,
        n_refuge_chambers: int = 3,
        n_exits: int = 2,
        n_risk_zones: int = 4,
        random_seed: int = 42,
    ) -> None:
        self.n_levels = n_levels
        self.galleries_per_level = galleries_per_level
        self.max_branch_depth = max_branch_depth
        self.branch_probability = branch_probability
        self.gallery_segment_length_m = gallery_segment_length_m
        self.level_height_m = level_height_m
        self.n_refuge_chambers = n_refuge_chambers
        self.n_exits = n_exits
        self.n_risk_zones = n_risk_zones
        self.random_seed = random_seed


class LayoutGenerator:
    """Genera instancias de `MineLayout` de forma determinista dado un seed."""

    def __init__(self, config: LayoutGeneratorConfig | None = None) -> None:
        self.config = config or LayoutGeneratorConfig()
        self._rng = random.Random(self.config.random_seed)
        self._node_counter = 0
        self._edge_counter = 0
        self._nodes: dict[str, MineNode] = {}
        self._edges: dict[str, MineEdge] = {}

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def generate(self, layout_id: str = "synthetic_default") -> MineLayout:
        self._nodes.clear()
        self._edges.clear()
        self._node_counter = 0
        self._edge_counter = 0

        level_entry_nodes: list[str] = []
        for level in range(self.config.n_levels):
            entry_id = self._grow_level(level)
            level_entry_nodes.append(entry_id)

        self._connect_levels(level_entry_nodes)
        self._place_refuge_chambers()
        self._place_exits(level_entry_nodes[0])
        self._place_risk_zones()

        layout = MineLayout(
            layout_id=layout_id,
            source="synthetic_procedural",
            nodes=list(self._nodes.values()),
            edges=list(self._edges.values()),
        )
        layout = self._ensure_connectivity(layout)
        logger.info(
            "Layout sintético generado",
            extra={
                "layout_id": layout_id,
                "n_nodes": len(layout.nodes),
                "n_edges": len(layout.edges),
            },
        )
        return layout

    # ------------------------------------------------------------------
    # Construcción de un nivel (árbol ramificado de galerías)
    # ------------------------------------------------------------------
    def _grow_level(self, level: int) -> str:
        z = -level * self.config.level_height_m
        entry_id = self._new_node(NodeType.INTERSECTION, position=(0.0, 0.0, z), level=level)

        frontier: list[tuple[str, float, float, int]] = [(entry_id, 0.0, 0.0, 0)]
        n_galleries_created = 0

        while frontier and n_galleries_created < self.config.galleries_per_level:
            parent_id, px, py, depth = frontier.pop(0)
            n_branches = 2 if depth == 0 else (2 if self._rng.random() < self.config.branch_probability else 1)

            for _ in range(n_branches):
                if depth >= self.config.max_branch_depth:
                    continue
                angle = self._rng.uniform(0, 6.28318)
                seg_len = self._rng.uniform(*self.config.gallery_segment_length_m)
                nx_ = px + seg_len * _cos(angle)
                ny_ = py + seg_len * _sin(angle)

                node_type = (
                    NodeType.INTERSECTION if self._rng.random() < 0.6 else NodeType.GALLERY
                )
                child_id = self._new_node(node_type, position=(nx_, ny_, z), level=level)
                self._new_edge(parent_id, child_id, slope_pct=0.0)
                frontier.append((child_id, nx_, ny_, depth + 1))
                n_galleries_created += 1

        return entry_id

    def _connect_levels(self, level_entry_nodes: list[str]) -> None:
        """Conecta niveles consecutivos con rampas (aristas con pendiente)."""
        for i in range(len(level_entry_nodes) - 1):
            upper = level_entry_nodes[i]
            lower = level_entry_nodes[i + 1]
            ramp_length = self.config.level_height_m * 3.5  # rampa inclinada, no vertical
            slope = (self.config.level_height_m / ramp_length) * 100.0
            self._new_edge(upper, lower, length_m_override=ramp_length, slope_pct=slope)

    def _place_refuge_chambers(self) -> None:
        candidates = [
            n
            for n in self._nodes.values()
            if n.node_type == NodeType.INTERSECTION
        ]
        self._rng.shuffle(candidates)
        for node in candidates[: self.config.n_refuge_chambers]:
            node.node_type = NodeType.REFUGE_CHAMBER
            node.label = node.label or f"Refugio {node.node_id}"
            node.capacity = max(node.capacity, 10)

    def _place_exits(self, surface_entry_id: str) -> None:
        surface_nodes = [
            n for n in self._nodes.values() if n.level == 0 and n.node_type == NodeType.INTERSECTION
        ]
        self._rng.shuffle(surface_nodes)
        chosen = surface_nodes[: max(1, self.config.n_exits)]
        if surface_entry_id not in [n.node_id for n in chosen] and chosen:
            pass  # la entrada de superficie no tiene por qué ser salida
        for node in chosen:
            node.node_type = NodeType.EXIT
            node.label = node.label or f"Salida {node.node_id}"

    def _place_risk_zones(self) -> None:
        candidates = [n for n in self._nodes.values() if n.node_type == NodeType.GALLERY]
        self._rng.shuffle(candidates)
        for node in candidates[: self.config.n_risk_zones]:
            node.node_type = NodeType.RISK_ZONE
            node.label = node.label or f"Zona de riesgo {node.node_id}"
            for edge in self._edges.values():
                if node.node_id in (edge.source, edge.target):
                    edge.base_risk = max(edge.base_risk, self._rng.uniform(0.3, 0.7))

    # ------------------------------------------------------------------
    # Validación / reparación de conectividad
    # ------------------------------------------------------------------
    def _ensure_connectivity(self, layout: MineLayout) -> MineLayout:
        exits = [n.node_id for n in layout.nodes if n.node_type == NodeType.EXIT]
        if not exits:
            raise ValueError("El layout generado no contiene ninguna salida (EXIT).")

        g = nx.Graph()
        g.add_nodes_from(n.node_id for n in layout.nodes)
        g.add_edges_from((e.source, e.target) for e in layout.edges)

        unreachable: list[str] = []
        for node_id in g.nodes:
            if not any(nx.has_path(g, node_id, exit_id) for exit_id in exits):
                unreachable.append(node_id)

        if not unreachable:
            return layout

        logger.warning(
            "Reparando conectividad del layout: nodos inalcanzables detectados",
            extra={"n_unreachable": len(unreachable)},
        )
        node_positions = {n.node_id: n.position for n in layout.nodes}
        for node_id in unreachable:
            nearest = min(
                (n for n in g.nodes if n != node_id and n not in unreachable),
                key=lambda other: _euclidean(node_positions[node_id], node_positions[other]),
                default=None,
            )
            if nearest is None:
                continue
            self._nodes = layout.node_map()
            self._edges = layout.edge_map()
            self._new_edge(node_id, nearest, slope_pct=0.0)
            layout.edges.append(self._edges[list(self._edges.keys())[-1]])
            g.add_edge(node_id, nearest)

        return layout

    # ------------------------------------------------------------------
    # Helpers de creación de entidades
    # ------------------------------------------------------------------
    def _new_node(
        self, node_type: NodeType, position: tuple[float, float, float], level: int
    ) -> str:
        self._node_counter += 1
        node_id = f"N{self._node_counter}"
        self._nodes[node_id] = MineNode(
            node_id=node_id, node_type=node_type, position=position, level=level
        )
        return node_id

    def _new_edge(
        self,
        source: str,
        target: str,
        slope_pct: float = 0.0,
        length_m_override: float | None = None,
    ) -> str:
        self._edge_counter += 1
        edge_id = f"E{self._edge_counter}"
        length = length_m_override or _euclidean(
            self._nodes[source].position, self._nodes[target].position
        )
        self._edges[edge_id] = MineEdge(
            edge_id=edge_id,
            source=source,
            target=target,
            length_m=max(length, 1.0),
            slope_pct=slope_pct,
            status=EdgeStatus.CLEAR,
        )
        return edge_id


# ----------------------------------------------------------------------
# Utilidades matemáticas puras (evitamos dependencia de `math` repetida)
# ----------------------------------------------------------------------
import math


def _cos(angle: float) -> float:
    return math.cos(angle)


def _sin(angle: float) -> float:
    return math.sin(angle)


def _euclidean(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist(a, b)
