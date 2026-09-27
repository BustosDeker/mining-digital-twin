"""Tests de la Fase 4: Router protocol, A* adaptativo y Q-learning."""

import pytest

from backend.digital_twin.graph_models import EdgeStatus, NodeType
from backend.digital_twin.layout_generator import LayoutGenerator, LayoutGeneratorConfig
from backend.digital_twin.state import DigitalTwinState, HazardType
from backend.simulation.model import MineEvacuationModel
from backend.simulation.routing import AdaptiveShortestPathRouter, QLearningRouter


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
    return LayoutGenerator(config).generate("pytest_routing_layout")


def test_digital_twin_state_instances_are_isolated(layout):
    """Regresión del bug crítico: dos DigitalTwinState sobre el mismo layout
    NO deben compartir objetos de arista mutables.
    """
    state_a = DigitalTwinState(layout)
    state_b = DigitalTwinState(layout)

    edge_id = layout.edges[0].edge_id
    state_a.set_edge_status(edge_id, EdgeStatus.BLOCKED, risk=1.0)

    assert state_a.get_edge(edge_id).status == EdgeStatus.BLOCKED
    assert state_b.get_edge(edge_id).status == EdgeStatus.CLEAR
    assert layout.edge_map()[edge_id].status == EdgeStatus.CLEAR


def test_adaptive_router_reaches_exit_when_clear(layout):
    state = DigitalTwinState(layout)
    router = AdaptiveShortestPathRouter()
    exit_node = next(n.node_id for n in layout.nodes if n.node_type == NodeType.EXIT)
    other_node = next(
        n.node_id
        for n in layout.nodes
        if n.node_type not in (NodeType.EXIT,)
    )
    path = router(state, other_node)
    assert path is not None
    assert path[-1] in [n.node_id for n in layout.nodes if n.node_type == NodeType.EXIT]


def test_qlearning_training_does_not_mutate_original_layout(layout):
    risk_nodes = [n.node_id for n in layout.nodes if n.node_type == NodeType.RISK_ZONE]
    router = QLearningRouter(random_seed=1)
    router.train_offline(
        layout, hazard_origin_node_ids=risk_nodes, n_episodes_per_origin=20, max_steps_per_episode=20
    )
    assert all(e.status == EdgeStatus.CLEAR for e in layout.edges)
    assert len(router.q_table) > 0


def test_qlearning_router_produces_valid_path_after_training(layout):
    risk_nodes = [n.node_id for n in layout.nodes if n.node_type == NodeType.RISK_ZONE]
    router = QLearningRouter(random_seed=1)
    router.train_offline(
        layout, hazard_origin_node_ids=risk_nodes, n_episodes_per_origin=30, max_steps_per_episode=30
    )

    state = DigitalTwinState(layout)
    state.spawn_hazard(HazardType.FIRE, origin_node_id=risk_nodes[0], intensity=0.9)
    for node in layout.nodes:
        if node.node_type == NodeType.EXIT:
            continue
        path = router(state, node.node_id)
        assert path is not None and len(path) >= 1


def test_both_routers_are_interchangeable_in_model(layout):
    """Ambas estrategias deben cumplir el mismo protocolo `Router` /
    `PathProvider` y ser intercambiables sin cambios en el modelo ABM.
    """
    risk_nodes = [n.node_id for n in layout.nodes if n.node_type == NodeType.RISK_ZONE]

    for RouterCls in (AdaptiveShortestPathRouter,):
        state = DigitalTwinState(layout)
        state.spawn_hazard(HazardType.FIRE, origin_node_id=risk_nodes[0], intensity=0.9)
        model = MineEvacuationModel(state, n_agents=10, path_provider=RouterCls(), random_seed=1)
        metrics = model.run_headless(max_steps=500)
        accounted = metrics.evacuated + metrics.sheltered + metrics.lost + metrics.timed_out
        assert accounted == metrics.total_agents

    trained_router = QLearningRouter(random_seed=1)
    trained_router.train_offline(
        layout, hazard_origin_node_ids=risk_nodes, n_episodes_per_origin=20, max_steps_per_episode=20
    )
    state = DigitalTwinState(layout)
    state.spawn_hazard(HazardType.FIRE, origin_node_id=risk_nodes[0], intensity=0.9)
    model = MineEvacuationModel(state, n_agents=10, path_provider=trained_router, random_seed=1)
    metrics = model.run_headless(max_steps=500)
    accounted = metrics.evacuated + metrics.sheltered + metrics.lost + metrics.timed_out
    assert accounted == metrics.total_agents
