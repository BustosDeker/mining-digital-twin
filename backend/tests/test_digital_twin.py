"""Tests de la Fase 2: grafo, generador de layout y estado del gemelo digital."""

import networkx as nx
import pytest

from backend.digital_twin.graph_models import EdgeStatus, NodeType
from backend.digital_twin.layout_generator import LayoutGenerator, LayoutGeneratorConfig
from backend.digital_twin.state import DigitalTwinState, HazardType


@pytest.fixture()
def layout():
    config = LayoutGeneratorConfig(
        n_levels=3,
        galleries_per_level=8,
        n_refuge_chambers=3,
        n_exits=2,
        n_risk_zones=4,
        random_seed=42,
    )
    return LayoutGenerator(config).generate(layout_id="pytest_layout")


def test_layout_is_deterministic_given_seed():
    config = LayoutGeneratorConfig(random_seed=7)
    layout_a = LayoutGenerator(config).generate("a")
    layout_b = LayoutGenerator(config).generate("b")
    assert [n.node_id for n in layout_a.nodes] == [n.node_id for n in layout_b.nodes]
    assert [e.edge_id for e in layout_a.edges] == [e.edge_id for e in layout_b.edges]


def test_layout_has_at_least_one_exit(layout):
    exits = [n for n in layout.nodes if n.node_type == NodeType.EXIT]
    assert len(exits) >= 1


def test_layout_is_fully_evacuable(layout):
    g = nx.Graph()
    g.add_nodes_from(n.node_id for n in layout.nodes)
    g.add_edges_from((e.source, e.target) for e in layout.edges)
    exits = [n.node_id for n in layout.nodes if n.node_type == NodeType.EXIT]
    assert nx.number_connected_components(g) == 1
    for node_id in g.nodes:
        assert any(nx.has_path(g, node_id, exit_id) for exit_id in exits)


def test_digital_twin_state_mutations(layout):
    state = DigitalTwinState(layout)
    edge = layout.edges[0]

    state.set_edge_status(edge.edge_id, EdgeStatus.BLOCKED, risk=0.95)
    assert edge.edge_id in state.blocked_edges
    assert state.risk_by_edge[edge.edge_id] == pytest.approx(0.95)

    event = state.spawn_hazard(HazardType.GAS_LEAK, origin_node_id=layout.nodes[0].node_id)
    assert event.event_id in state.active_hazards

    state.advance_step()
    assert state.current_step == 1

    snapshot = state.snapshot()
    assert snapshot["step"] == 1
    assert len(snapshot["active_hazards"]) == 1

    state.reset_dynamic_state()
    assert state.blocked_edges == []
    assert state.active_hazards == {}
    assert state.current_step == 0
