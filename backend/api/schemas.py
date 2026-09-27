"""Esquemas Pydantic de request/response de la API FastAPI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LayoutConfigRequest(BaseModel):
    n_levels: int = Field(default=3, ge=1, le=6)
    galleries_per_level: int = Field(default=8, ge=1, le=30)
    n_refuge_chambers: int = Field(default=3, ge=0, le=10)
    n_exits: int = Field(default=2, ge=1, le=6)
    n_risk_zones: int = Field(default=4, ge=0, le=15)
    random_seed: int = Field(default=42)


class CreateSessionRequest(BaseModel):
    scenario_name: str = "default"
    n_agents: int | None = Field(default=None, ge=1, le=200)
    router_name: Literal["adaptive_astar", "q_learning"] = "adaptive_astar"
    hazard_type: Literal["fire", "collapse", "gas_leak"] | None = "fire"
    hazard_intensity: float = Field(default=0.9, ge=0.0, le=1.0)
    layout_config: LayoutConfigRequest = Field(default_factory=LayoutConfigRequest)


class SessionSummary(BaseModel):
    session_id: str
    scenario_name: str
    router_name: str
    status: str
    step: int


class MonteCarloRequest(BaseModel):
    scenario_name: str = "mc_scenario"
    router_names: list[Literal["adaptive_astar", "q_learning"]] = Field(
        default_factory=lambda: ["adaptive_astar", "q_learning"]
    )
    n_runs: int | None = Field(default=None, ge=2, le=1000)
    n_agents: int | None = Field(default=None, ge=1, le=200)
    hazard_type: Literal["fire", "collapse", "gas_leak"] | None = "fire"
    hazard_intensity: float = Field(default=0.9, ge=0.0, le=1.0)
    metric_name: str = "evacuation_rate"
    layout_config: LayoutConfigRequest = Field(default_factory=LayoutConfigRequest)
    q_learning_training_episodes: int = Field(
        default=100, ge=10, le=2000, description="Episodios de entrenamiento offline si se incluye 'q_learning'."
    )


class RunEDARequest(BaseModel):
    subject_limit: int | None = Field(default=None, ge=1)


class ActivateModelResponse(BaseModel):
    architecture_name: str
    version_id: str
    activated_at: str


class PredictStressRequest(BaseModel):
    """Vector de features HRV/EDA (mismo esquema que `preprocessing/features.py`)
    para inferencia con el modelo `features_mlp` activo."""

    features: dict[str, float]


class PredictStressResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    model_version_id: str


class GenericMessage(BaseModel):
    detail: str
    data: dict[str, Any] | None = None
