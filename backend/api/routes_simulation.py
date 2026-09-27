"""Rutas REST de simulación: ciclo de vida de sesiones del gemelo digital
y lanzamiento de comparaciones Monte Carlo bajo demanda.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas import CreateSessionRequest, MonteCarloRequest, SessionSummary
from backend.digital_twin.graph_models import NodeType
from backend.digital_twin.layout_generator import LayoutGeneratorConfig
from backend.digital_twin.state import HazardType
from backend.evaluation.routing_comparison import compare_routing_strategies
from backend.services.session_manager import session_manager
from backend.simulation.routing import AdaptiveShortestPathRouter, QLearningRouter
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/simulations", tags=["simulation"])


def _session_summary(session) -> SessionSummary:
    return SessionSummary(
        session_id=session.session_id,
        scenario_name=session.scenario_name,
        router_name=session.router_name,
        status=session.status.value,
        step=session.state.current_step,
    )


@router.post("", response_model=SessionSummary)
def create_session(payload: CreateSessionRequest) -> SessionSummary:
    layout_config = LayoutGeneratorConfig(**payload.layout_config.model_dump())
    hazard_type = HazardType(payload.hazard_type) if payload.hazard_type else None

    session = session_manager.create_session(
        scenario_name=payload.scenario_name,
        n_agents=payload.n_agents,
        router_name=payload.router_name,
        hazard_type=hazard_type,
        hazard_intensity=payload.hazard_intensity,
        layout_config=layout_config,
    )
    return _session_summary(session)


@router.get("", response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    return [_session_summary(s) for s in session_manager.list_sessions()]


@router.get("/{session_id}")
def get_session_state(session_id: str) -> dict:
    try:
        session = session_manager.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return session.snapshot()


@router.post("/{session_id}/start", response_model=SessionSummary)
async def start_session(session_id: str) -> SessionSummary:
    try:
        session = await session_manager.start(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _session_summary(session)


@router.post("/{session_id}/pause", response_model=SessionSummary)
def pause_session(session_id: str) -> SessionSummary:
    try:
        session = session_manager.pause(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _session_summary(session)


@router.post("/{session_id}/stop", response_model=SessionSummary)
def stop_session(session_id: str) -> SessionSummary:
    try:
        session = session_manager.stop(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _session_summary(session)


@router.post("/{session_id}/reset", response_model=SessionSummary)
def reset_session(session_id: str) -> SessionSummary:
    try:
        session = session_manager.reset(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _session_summary(session)


@router.post("/{session_id}/step", response_model=SessionSummary)
async def step_session(session_id: str) -> SessionSummary:
    try:
        session = await session_manager.step_once(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _session_summary(session)


# --------------------------------------------------------------------
# Monte Carlo bajo demanda (Fase 7, expuesto ahora vía API)
# --------------------------------------------------------------------
mc_router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

_ROUTER_BUILDERS = {
    "adaptive_astar": lambda layout, risk_nodes, episodes: AdaptiveShortestPathRouter(),
    "q_learning": None,  # se construye/entrena bajo demanda, ver abajo
}


@mc_router.post("/routing-comparison")
def routing_comparison_endpoint(payload: MonteCarloRequest) -> dict:
    from backend.digital_twin.layout_generator import LayoutGenerator

    layout_config = LayoutGeneratorConfig(**payload.layout_config.model_dump())
    layout = LayoutGenerator(layout_config).generate(payload.scenario_name)
    risk_nodes = [n.node_id for n in layout.nodes if n.node_type == NodeType.RISK_ZONE]
    if not risk_nodes:
        raise HTTPException(status_code=400, detail="El layout generado no tiene zonas de riesgo.")

    routers = {}
    for name in payload.router_names:
        if name == "adaptive_astar":
            routers[name] = AdaptiveShortestPathRouter()
        elif name == "q_learning":
            ql = QLearningRouter()
            ql.train_offline(
                layout,
                hazard_origin_node_ids=risk_nodes,
                n_episodes_per_origin=payload.q_learning_training_episodes,
                max_steps_per_episode=40,
            )
            routers[name] = ql
        else:
            raise HTTPException(status_code=400, detail=f"Router desconocido: {name}")

    hazard_type = HazardType(payload.hazard_type) if payload.hazard_type else None
    report = compare_routing_strategies(
        layout,
        routers=routers,
        scenario_name=payload.scenario_name,
        n_runs=payload.n_runs,
        n_agents=payload.n_agents,
        hazard_type=hazard_type,
        hazard_origin_node_id=risk_nodes[0],
        metric_name=payload.metric_name,
    )

    return {
        "scenario_name": payload.scenario_name,
        "metric_name": report.metric_name,
        "per_router_summary": report.per_router_summary,
        "statistical_decision": report.statistical_decision,
    }
