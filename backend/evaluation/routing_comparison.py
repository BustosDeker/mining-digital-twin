"""Comparación cuantitativa de estrategias de enrutamiento (Componente 2).

Ejecuta Monte Carlo para cada estrategia sobre el MISMO layout y el MISMO
evento de emergencia (misma semilla base por corrida, para que las
corridas estén pareadas: la corrida i de cada estrategia enfrenta
exactamente el mismo escenario aleatorio), y aplica Wilcoxon/Friedman para
decidir cuál es mejor, no solo cuál tiene la media más alta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.digital_twin.graph_models import MineLayout
from backend.digital_twin.state import HazardType
from backend.evaluation.monte_carlo import MonteCarloResult, run_monte_carlo
from backend.simulation.routing import Router
from backend.statistics.tests import select_best
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RoutingComparisonReport:
    metric_name: str
    per_router_summary: dict[str, dict[str, float]]
    statistical_decision: dict[str, Any]
    monte_carlo_results: dict[str, MonteCarloResult]


def compare_routing_strategies(
    layout: MineLayout,
    routers: dict[str, Router],
    scenario_name: str = "default_scenario",
    n_runs: int | None = None,
    n_agents: int | None = None,
    hazard_type: HazardType | None = HazardType.FIRE,
    hazard_origin_node_id: str | None = None,
    metric_name: str = "evacuation_rate",
    base_seed: int = 1000,
) -> RoutingComparisonReport:
    mc_results: dict[str, MonteCarloResult] = {}
    for router_name, router in routers.items():
        mc_results[router_name] = run_monte_carlo(
            layout,
            scenario_name=scenario_name,
            router=router,
            router_name=router_name,
            n_agents=n_agents,
            n_runs=n_runs,
            hazard_type=hazard_type,
            hazard_origin_node_id=hazard_origin_node_id,
            base_seed=base_seed,  # misma semilla base -> corridas pareadas entre estrategias
        )

    samples = {name: result.metric_array(metric_name) for name, result in mc_results.items()}
    decision = select_best(samples)

    per_router_summary = {name: result.summary() for name, result in mc_results.items()}

    logger.info(
        "Comparación de estrategias de enrutamiento finalizada",
        extra={"metric": metric_name, "decision": decision["statistically_justified_best"]},
    )

    return RoutingComparisonReport(
        metric_name=metric_name,
        per_router_summary=per_router_summary,
        statistical_decision=decision,
        monte_carlo_results=mc_results,
    )
