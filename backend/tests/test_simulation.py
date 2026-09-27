"""Tests de la Fase 3: parámetros, propagación de peligro y modelo ABM."""

import pytest

from backend.digital_twin.graph_models import EdgeStatus, NodeType
from backend.digital_twin.layout_generator import LayoutGenerator, LayoutGeneratorConfig
from backend.digital_twin.state import DigitalTwinState, HazardType
from backend.simulation.hazard_propagation import HazardPropagationEngine
from backend.simulation.model import MineEvacuationModel
from backend.simulation.parameters import DEFAULT_MOVEMENT_PARAMS


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
    return LayoutGenerator(config).generate("pytest_layout")


def test_effective_speed_decreases_with_slope_and_visibility():
    baseline = DEFAULT_MOVEMENT_PARAMS.effective_speed(
        panic_level=0.0, slope_pct=0.0, degraded_visibility=False,
        carrying_load=False, cumulative_distance_m=0.0,
    )
    penalized = DEFAULT_MOVEMENT_PARAMS.effective_speed(
        panic_level=0.0, slope_pct=20.0, degraded_visibility=True,
        carrying_load=True, cumulative_distance_m=500.0,
    )
    assert penalized < baseline


def test_effective_speed_increases_with_panic():
    calm = DEFAULT_MOVEMENT_PARAMS.effective_speed(
        panic_level=0.0, slope_pct=0.0, degraded_visibility=False,
        carrying_load=False, cumulative_distance_m=0.0,
    )
    panicked = DEFAULT_MOVEMENT_PARAMS.effective_speed(
        panic_level=0.9, slope_pct=0.0, degraded_visibility=False,
        carrying_load=False, cumulative_distance_m=0.0,
    )
    assert panicked > calm


def test_hazard_propagation_blocks_edges_over_time(layout):
    state = DigitalTwinState(layout)
    risk_node = next(n.node_id for n in layout.nodes if n.node_type == NodeType.RISK_ZONE)
    state.spawn_hazard(HazardType.FIRE, origin_node_id=risk_node, intensity=0.9)
    engine = HazardPropagationEngine(state)

    for _ in range(60):
        engine.step()
        state.advance_step()

    # Importante: se verifica sobre `state.layout` (la copia interna que
    # DigitalTwinState mantiene y muta), no sobre el `layout` original que
    # se le pasó al constructor — éste permanece intacto por diseño.
    statuses = {e.status for e in state.layout.edges}
    assert EdgeStatus.BLOCKED in statuses or EdgeStatus.DEGRADED in statuses


def test_model_without_hazard_evacuates_all_agents(layout):
    state = DigitalTwinState(layout)
    model = MineEvacuationModel(state, n_agents=15, random_seed=1)
    metrics = model.run_headless(max_steps=1500)

    assert metrics.evacuated == metrics.total_agents
    assert metrics.lost == 0
    assert metrics.timed_out == 0
    assert metrics.evacuation_rate == pytest.approx(1.0)


def test_model_with_severe_fire_reduces_evacuation_rate(layout):
    state_baseline = DigitalTwinState(layout)
    model_baseline = MineEvacuationModel(state_baseline, n_agents=15, random_seed=1)
    metrics_baseline = model_baseline.run_headless(max_steps=1500)

    config = LayoutGeneratorConfig(
        n_levels=3, galleries_per_level=8, n_refuge_chambers=3,
        n_exits=2, n_risk_zones=4, random_seed=42,
    )
    layout2 = LayoutGenerator(config).generate("pytest_layout_2")
    state_fire = DigitalTwinState(layout2)
    risk_node = next(n.node_id for n in layout2.nodes if n.node_type == NodeType.RISK_ZONE)
    state_fire.spawn_hazard(HazardType.FIRE, origin_node_id=risk_node, intensity=0.9)
    model_fire = MineEvacuationModel(state_fire, n_agents=15, random_seed=1)
    metrics_fire = model_fire.run_headless(max_steps=1500)

    assert metrics_fire.evacuation_rate < metrics_baseline.evacuation_rate


def test_all_agents_reach_a_terminal_status(layout):
    state = DigitalTwinState(layout)
    risk_node = next(n.node_id for n in layout.nodes if n.node_type == NodeType.RISK_ZONE)
    state.spawn_hazard(HazardType.FIRE, origin_node_id=risk_node, intensity=0.9)
    model = MineEvacuationModel(state, n_agents=15, random_seed=2)
    metrics = model.run_headless(max_steps=1500)

    accounted = metrics.evacuated + metrics.sheltered + metrics.lost + metrics.timed_out
    assert accounted == metrics.total_agents
