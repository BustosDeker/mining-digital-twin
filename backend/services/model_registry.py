"""Model Registry: versionado, metadatos y activación explícita de modelos.

Cada modelo entrenado (de cualquier arquitectura) se registra con:
    - Pesos (`model.h5`, formato exigido por la regla fija del proyecto).
    - Scaler de normalización, si aplica (`scaler.pkl`, arquitectura
      features_mlp).
    - Metadatos completos (`metadata.json`): dataset, hiperparámetros,
      métricas de CV, fecha, versión, nombres de clase.

Solo el modelo marcado explícitamente como "activo" (`activate_model`) es
el que la API de producción (Fase 9) carga para inferencia — nunca el más
reciente por defecto, para evitar promociones accidentales sin revisión.
"""

from __future__ import annotations

import json
import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

_ACTIVE_POINTER_FILENAME = "active_model.json"


@dataclass
class ModelMetadata:
    architecture_name: str
    version_id: str
    dataset_name: str
    hyperparams: dict[str, Any]
    cv_metrics: dict[str, float]
    class_names: list[str]
    created_at: str
    has_scaler: bool
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_name": self.architecture_name,
            "version_id": self.version_id,
            "dataset_name": self.dataset_name,
            "hyperparams": self.hyperparams,
            "cv_metrics": self.cv_metrics,
            "class_names": self.class_names,
            "created_at": self.created_at,
            "has_scaler": self.has_scaler,
            "extra": self.extra or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        return cls(**data)


def _registry_root() -> Path:
    return get_settings().MODELS_REGISTRY_DIR


def _version_dir(architecture_name: str, version_id: str) -> Path:
    return _registry_root() / architecture_name / version_id


def register_model(
    model: tf.keras.Model,
    architecture_name: str,
    dataset_name: str,
    hyperparams: dict[str, Any],
    cv_metrics: dict[str, float],
    class_names: list[str],
    scaler: StandardScaler | None = None,
    extra: dict[str, Any] | None = None,
) -> ModelMetadata:
    version_id = f"v_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    out_dir = _version_dir(architecture_name, version_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(out_dir / "model.h5")

    if scaler is not None:
        with open(out_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    metadata = ModelMetadata(
        architecture_name=architecture_name,
        version_id=version_id,
        dataset_name=dataset_name,
        hyperparams=hyperparams,
        cv_metrics=cv_metrics,
        class_names=class_names,
        created_at=datetime.now(timezone.utc).isoformat(),
        has_scaler=scaler is not None,
        extra=extra,
    )
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(
        "Modelo registrado",
        extra={"architecture": architecture_name, "version_id": version_id, "dataset": dataset_name},
    )
    return metadata


def list_models(architecture_name: str | None = None) -> list[ModelMetadata]:
    root = _registry_root()
    if not root.exists():
        return []

    architectures = [architecture_name] if architecture_name else [p.name for p in root.iterdir() if p.is_dir()]
    results: list[ModelMetadata] = []
    for arch in architectures:
        arch_dir = root / arch
        if not arch_dir.is_dir():
            continue
        for version_dir in sorted(arch_dir.iterdir()):
            metadata_path = version_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    results.append(ModelMetadata.from_dict(json.load(f)))
    return sorted(results, key=lambda m: m.created_at, reverse=True)


def get_model_metadata(architecture_name: str, version_id: str) -> ModelMetadata:
    metadata_path = _version_dir(architecture_name, version_id) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata para {architecture_name}/{version_id}")
    with open(metadata_path, encoding="utf-8") as f:
        return ModelMetadata.from_dict(json.load(f))


def load_model(architecture_name: str, version_id: str) -> tuple[tf.keras.Model, StandardScaler | None, ModelMetadata]:
    version_dir = _version_dir(architecture_name, version_id)
    model = tf.keras.models.load_model(version_dir / "model.h5")

    scaler = None
    scaler_path = version_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    metadata = get_model_metadata(architecture_name, version_id)
    return model, scaler, metadata


# ------------------------------------------------------------------
# Activación explícita del modelo de producción
# ------------------------------------------------------------------
def activate_model(architecture_name: str, version_id: str) -> None:
    """Marca un modelo como el que sirve la API de producción.

    Regla del proyecto: la activación es siempre explícita, nunca
    automática al terminar un entrenamiento — un humano (o un pipeline de
    CI con aprobación) decide qué modelo pasa a producción.
    """
    get_model_metadata(architecture_name, version_id)  # valida que exista

    pointer = {
        "architecture_name": architecture_name,
        "version_id": version_id,
        "activated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(_registry_root() / _ACTIVE_POINTER_FILENAME, "w", encoding="utf-8") as f:
        json.dump(pointer, f, indent=2, ensure_ascii=False)

    logger.warning(
        "Modelo activado para producción",
        extra={"architecture": architecture_name, "version_id": version_id},
    )


def get_active_pointer() -> dict[str, str] | None:
    pointer_path = _registry_root() / _ACTIVE_POINTER_FILENAME
    if not pointer_path.exists():
        return None
    with open(pointer_path, encoding="utf-8") as f:
        return json.load(f)


def load_active_model() -> tuple[tf.keras.Model, StandardScaler | None, ModelMetadata] | None:
    pointer = get_active_pointer()
    if pointer is None:
        return None
    return load_model(pointer["architecture_name"], pointer["version_id"])
