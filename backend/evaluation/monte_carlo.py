"""Monte Carlo de escenarios de evacuación: N corridas por escenario.

Regla del proyecto: toda métrica de simulación de evacuación (tiempo total,
tiempo por agente, tasa de no evacuados, longitud de cola en cuellos de
botella) se calcula por Monte Carlo (N corridas), nunca por una corrida
única — el ABM tiene componentes estocásticos (posición inicial, pánico,
elección de rutas empatadas) que hacen que una sola corrida no sea
representativa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.digital_twin.graph_models import MineLayout, NodeType
from backend.digital_twin.state import DigitalTwinState, HazardType
from backend.simulation.model import EvacuationRunMetrics, MineEvacuationModel
from backend.simulation.routing import Router
from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MonteCarloResult:
    scenario_name: str
    router_name: str
    n_runs: int
    run_metrics: list[EvacuationRunMetrics]

    def metric_array(self, metric_name: str) -> np.ndarray:
        return np.array([getattr(m, metric_name) for m in self.run_metrics])

    def summary(self) -> dict[str, float]:
        evac_rates = self.metric_array("evacuation_rate")
        not_evac_rates = self.metric_array("not_evacuated_rate")
        bottlenecks = np.array([m.bottleneck_max_waiting for m in self.run_metrics])
        mean_evac_times = np.array(
            [
                np.mean(m.evacuation_times_steps) if m.evacuation_times_steps else np.nan
                for m in self.run_metrics
            ]
        )
        return {
            "n_runs": self.n_runs,
            "mean_evacuation_rate": float(np.mean(evac_rates)),
            "std_evacuation_rate": float(np.std(evac_rates)),
            "mean_not_evacuated_rate": float(np.mean(not_evac_rates)),
            "mean_evacuation_time_steps": float(np.nanmean(mean_evac_times)),
            "mean_bottleneck_max_waiting": float(np.mean(bottlenecks)),
        }


def run_monte_carlo(
    layout: MineLayout,
    scenario_name: str,
    router: Router,
    router_name: str,
    n_agents: int | None = None,
    n_runs: int | None = None,
    max_steps: int | None = None,
    hazard_type: HazardType | None = HazardType.FIRE,
    hazard_origin_node_id: str | None = None,
    hazard_intensity: float = 0.9,
    base_seed: int = 1000,
) -> MonteCarloResult:
    """Ejecuta N corridas independientes del mismo escenario (mismo layout,
    mismo evento de emergencia) variando solo la semilla aleatoria del ABM,
    para obtener una distribución de métricas, no un punto único.
    """
    settings = get_settings()
    n_runs = n_runs or settings.ABM_MONTE_CARLO_RUNS
    n_agents = n_agents or settings.ABM_DEFAULT_N_AGENTS
    max_steps = max_steps or settings.ABM_MAX_STEPS

    if hazard_type is not None and hazard_origin_node_id is None:
        risk_nodes = [n.node_id for n in layout.nodes if n.node_type == NodeType.RISK_ZONE]
        if not risk_nodes:
            raise ValueError("El layout no tiene zonas de riesgo para originar el evento.")
        hazard_origin_node_id = risk_nodes[0]

    run_metrics: list[EvacuationRunMetrics] = []
    for run_idx in range(n_runs):
        state = DigitalTwinState(layout)  # copia profunda e independiente por corrida
        if hazard_type is not None:
            state.spawn_hazard(hazard_type, origin_node_id=hazard_origin_node_id, intensity=hazard_intensity)

        model = MineEvacuationModel(
            state,
            n_agents=n_agents,
            path_provider=router,
            random_seed=base_seed + run_idx,
        )
        metrics = model.run_headless(max_steps=max_steps)
        run_metrics.append(metrics)

    logger.info(
        "Monte Carlo finalizado",
        extra={"scenario": scenario_name, "router": router_name, "n_runs": n_runs},
    )

    return MonteCarloResult(
        scenario_name=scenario_name, router_name=router_name, n_runs=n_runs, run_metrics=run_metrics
    )
