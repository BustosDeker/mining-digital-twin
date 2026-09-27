"""Configuración centralizada del sistema, cargada desde variables de entorno.

Regla fija del proyecto: ningún valor de negocio queda hardcodeado en el
código. Todo parámetro configurable (datasets activos, hiperparámetros por
defecto, rutas, umbrales de intervención, CORS, etc.) se define aquí y se
sobreescribe mediante un archivo `.env` o variables de entorno reales.

Uso:
    from backend.utils.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del proyecto = carpeta que contiene `backend/` y `frontend/`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"


class Settings(BaseSettings):
    """Configuración global del backend (Motor IA + FastAPI + ABM)."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------
    # Metadatos de la aplicación
    # ---------------------------------------------------------------
    APP_NAME: str = "mining-digital-twin"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_JSON: bool = False  # True en producción para logs estructurados

    # ---------------------------------------------------------------
    # Rutas de datos y artefactos (todas relativas a PROJECT_ROOT,
    # sobreescribibles si se despliega con otra disposición de discos)
    # ---------------------------------------------------------------
    DATA_RAW_DIR: Path = BACKEND_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BACKEND_ROOT / "data" / "processed"
    DATA_SESSIONS_DIR: Path = BACKEND_ROOT / "data" / "sessions"
    MODELS_REGISTRY_DIR: Path = BACKEND_ROOT / "models_registry"
    REPORTS_DIR: Path = BACKEND_ROOT / "data" / "reports"
    ARTIFACTS_DIR: Path = BACKEND_ROOT / "data" / "artifacts"  # EDA, curvas, matrices

    # ---------------------------------------------------------------
    # Motor IA — selección de datasets (regla fija: por configuración,
    # nunca hardcoded en el código de entrenamiento)
    # ---------------------------------------------------------------
    DATASETS: list[str] = Field(default=["wesad"])
    WESAD_URL: str = (
        "https://uni-siegen.sciebo.de/s/HGdUkoNlTOJOtwZ/download"  # espejo oficial WESAD
    )
    WESAD_EXPECTED_SHA256: str | None = None  # se fija tras la primera descarga verificada

    # ---------------------------------------------------------------
    # Motor IA — preprocesamiento / ventaneo (compartido con inferencia
    # y con el esquema de canales usado en sesiones propias)
    # ---------------------------------------------------------------
    SIGNAL_SAMPLE_RATE_HZ: int = 700  # frecuencia base del chest device WESAD (RespiBAN)
    WINDOW_SECONDS: float = 60.0
    WINDOW_OVERLAP: float = 0.5  # 50% overlap
    STRESS_CLASSES: list[str] = Field(default=["baseline", "stress", "amusement"])

    # ---------------------------------------------------------------
    # Motor IA — entrenamiento
    # ---------------------------------------------------------------
    CV_STRATEGY: Literal["kfold", "loso"] = "loso"
    CV_N_FOLDS: int = 5
    RANDOM_SEED: int = 42
    OPTUNA_N_TRIALS: int = 30
    MODEL_ARCHITECTURES: list[str] = Field(default=["cnn_lstm", "features_mlp"])

    # ---------------------------------------------------------------
    # Simulación ABM — parámetros no biométricos (velocidades, etc. se
    # documentan y citan en simulation/parameters.py, no aquí)
    # ---------------------------------------------------------------
    ABM_DEFAULT_N_AGENTS: int = 30
    ABM_MONTE_CARLO_RUNS: int = 100
    ABM_MAX_STEPS: int = 2000
    ABM_STEP_SECONDS: float = 1.0

    # ---------------------------------------------------------------
    # Módulo de intervención adaptativa (reglas + umbrales auditable)
    # ---------------------------------------------------------------
    STRESS_THRESHOLD_MODERATE: float = 0.4
    STRESS_THRESHOLD_HIGH: float = 0.75
    INTERVENTION_AUTO_TAKEOVER_STEPS: int = 5  # pasos consecutivos en estrés alto

    # ---------------------------------------------------------------
    # API / WebSocket / CORS
    # ---------------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = Field(default=["http://localhost:3000"])
    WEBSOCKET_BROADCAST_HZ: float = 10.0  # frecuencia de emisión del estado del gemelo

    @field_validator(
        "DATA_RAW_DIR",
        "DATA_PROCESSED_DIR",
        "DATA_SESSIONS_DIR",
        "MODELS_REGISTRY_DIR",
        "REPORTS_DIR",
        "ARTIFACTS_DIR",
        mode="after",
    )
    @classmethod
    def _ensure_dir_exists(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia cacheada (singleton) de la configuración."""
    return Settings()
