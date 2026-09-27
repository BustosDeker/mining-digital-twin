"""Modelos tipados del grafo de la mina.

El grafo es la fuente única de verdad de la topología de la mina subterránea:
galerías, intersecciones, cámaras de refugio, salidas y zonas de riesgo, junto
con las aristas que las conectan (tramos de galería transitables).

Estos modelos son consumidos por:
- `simulation/` (ABM y enrutamiento) para razonar sobre el grafo.
- `api/` para serializar el estado hacia el frontend por WebSocket/REST.
- El generador de layouts (`layout_generator.py`) para construir instancias.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class NodeType(str, Enum):
    """Tipo semántico de un nodo del grafo de la mina."""

    GALLERY = "gallery"  # punto intermedio de una galería (no es decisión)
    INTERSECTION = "intersection"  # punto donde el agente puede elegir ruta
    REFUGE_CHAMBER = "refuge_chamber"  # cámara de refugio (destino seguro temporal)
    EXIT = "exit"  # salida a superficie (destino final de evacuación)
    RISK_ZONE = "risk_zone"  # zona con alta probabilidad de incidente


class EdgeStatus(str, Enum):
    """Estado de transitabilidad de una arista (tramo de galería)."""

    CLEAR = "clear"  # transitable sin restricciones
    DEGRADED = "degraded"  # transitable con penalización (humo, obstrucción parcial)
    BLOCKED = "blocked"  # intransitable (colapso, fuego, gas)


class MineNode(BaseModel):
    """Un nodo del grafo: una posición discreta dentro de la mina."""

    node_id: str = Field(..., description="Identificador único, p.ej. 'N12'")
    node_type: NodeType
    position: tuple[float, float, float] = Field(
        ..., description="Coordenadas (x, y, z) en metros, z = profundidad/nivel"
    )
    level: int = Field(..., description="Nivel/piso de la mina (0 = más profundo)")
    label: str | None = Field(default=None, description="Etiqueta legible, p.ej. 'Refugio N°2'")
    capacity: int = Field(
        default=1, ge=1, description="Nº máximo de agentes simultáneos en el nodo"
    )

    @field_validator("node_id")
    @classmethod
    def _non_empty_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("node_id no puede estar vacío")
        return v


class MineEdge(BaseModel):
    """Una arista del grafo: un tramo de galería que conecta dos nodos."""

    edge_id: str = Field(..., description="Identificador único, p.ej. 'E5'")
    source: str = Field(..., description="node_id de origen")
    target: str = Field(..., description="node_id de destino")
    length_m: float = Field(..., gt=0, description="Longitud del tramo en metros")
    width_m: float = Field(default=2.5, gt=0, description="Ancho de la galería en metros")
    slope_pct: float = Field(
        default=0.0, description="Pendiente en porcentaje (positiva = subida en dirección target)"
    )
    base_risk: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Riesgo estructural base (0=nulo, 1=máximo)"
    )
    status: EdgeStatus = Field(default=EdgeStatus.CLEAR)
    current_risk: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Riesgo dinámico actual, actualizado por la propagación"
    )

    @property
    def is_traversable(self) -> bool:
        return self.status != EdgeStatus.BLOCKED

    @property
    def traversal_penalty(self) -> float:
        """Multiplicador de tiempo de tránsito según estado (1.0 = sin penalización)."""
        if self.status == EdgeStatus.BLOCKED:
            return float("inf")
        if self.status == EdgeStatus.DEGRADED:
            return 2.5
        return 1.0


class MineLayout(BaseModel):
    """Layout completo de la mina: colección de nodos y aristas.

    Regla del proyecto: debe documentarse explícitamente si el layout es
    sintético parametrizable o un layout real anonimizado (ver `source`).
    """

    layout_id: str
    source: str = Field(
        ..., description="'synthetic_procedural' o 'real_anonymized:<referencia>'"
    )
    nodes: list[MineNode]
    edges: list[MineEdge]

    def node_map(self) -> dict[str, MineNode]:
        return {n.node_id: n for n in self.nodes}

    def edge_map(self) -> dict[str, MineEdge]:
        return {e.edge_id: e for e in self.edges}

    def edges_by_endpoints(self) -> dict[tuple[str, str], MineEdge]:
        result: dict[tuple[str, str], MineEdge] = {}
        for e in self.edges:
            result[(e.source, e.target)] = e
            result[(e.target, e.source)] = e  # las galerías son bidireccionales
        return result
