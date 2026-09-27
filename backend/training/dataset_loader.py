"""Interfaz `DatasetLoader` del Motor IA.

Regla fija del proyecto: el Motor IA debe soportar uno o varios datasets
públicos simultáneamente, seleccionados por configuración
(`settings.DATASETS`), nunca por código acoplado a un dataset concreto.
Cada dataset implementa esta interfaz; añadir un dataset nuevo (SWELL-KW,
AffectiveROAD, etc.) NO requiere modificar el código de entrenamiento, solo
registrar una nueva implementación en `DATASET_REGISTRY`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SubjectRecording:
    """Una grabación cruda de un sujeto: señales fisiológicas + etiquetas.

    Todas las señales se almacenan re-muestreadas a `sample_rate_hz` (la
    frecuencia base de referencia, ver `settings.SIGNAL_SAMPLE_RATE_HZ`),
    para que distintos datasets con distintas frecuencias nativas puedan
    combinarse en un único pipeline de ventaneo (Fase 5, `preprocessing/`).
    """

    subject_id: str
    dataset_name: str
    sample_rate_hz: int
    channels: dict[str, np.ndarray]  # p.ej. {"ecg": [...], "eda": [...], ...}
    labels: np.ndarray  # una etiqueta de clase por muestra, misma longitud que las señales
    label_names: dict[int, str]  # mapeo id_entero -> nombre de clase (p.ej. {1: "baseline"})
    metadata: dict[str, str | float | int] | None = None

    def n_samples(self) -> int:
        any_channel = next(iter(self.channels.values()))
        return len(any_channel)


class DatasetLoader(ABC):
    """Contrato que debe implementar cada dataset soportado por el Motor IA."""

    #: Nombre corto usado en `settings.DATASETS` (p.ej. "wesad", "swell").
    dataset_name: str

    @abstractmethod
    def is_available_locally(self) -> bool:
        """True si los datos crudos ya están descargados y listos para leer."""

    @abstractmethod
    def list_subjects(self) -> list[str]:
        """IDs de todos los sujetos disponibles localmente."""

    @abstractmethod
    def load_subject(self, subject_id: str) -> SubjectRecording:
        """Carga la grabación cruda completa de un sujeto."""

    def load_all(self) -> list[SubjectRecording]:
        return [self.load_subject(sid) for sid in self.list_subjects()]


# ---------------------------------------------------------------------
# Registro de datasets soportados. Añadir un dataset nuevo = registrar su
# loader aquí; el resto del Motor IA (preprocesamiento, entrenamiento, EDA)
# no cambia.
# ---------------------------------------------------------------------
DATASET_REGISTRY: dict[str, type[DatasetLoader]] = {}


def register_dataset(name: str):
    def _decorator(cls: type[DatasetLoader]) -> type[DatasetLoader]:
        DATASET_REGISTRY[name] = cls
        return cls

    return _decorator


def get_loader(name: str) -> DatasetLoader:
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' no está registrado. Disponibles: {list(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name]()


def get_configured_loaders() -> list[DatasetLoader]:
    """Devuelve los loaders de todos los datasets activos en `settings.DATASETS`."""
    from backend.utils.config import get_settings

    settings = get_settings()
    return [get_loader(name) for name in settings.DATASETS]
