"""Wrapper de inferencia del clasificador de estrés para producción.

Encapsula el ciclo completo de inferencia en tiempo real sobre una ventana
de señal ya recibida (de una sesión VR/frontend futura, o de datos de
prueba): extracción de features (reutilizando `preprocessing/features.py`,
el MISMO código usado en entrenamiento — regla de no duplicar lógica),
normalización con el scaler del modelo activo, y predicción.

Este módulo es el punto único que `api/routes_ml.py` debería usar para
inferencia (mantiene la lógica de negocio fuera de la capa HTTP).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.preprocessing.features import extract_features
from backend.preprocessing.windowing import SignalWindow
from backend.services import model_registry as mr
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StressInferenceResult:
    predicted_class: str
    probabilities: dict[str, float]
    model_version_id: str
    architecture_name: str


class StressInferenceEngine:
    """Carga (y cachea en memoria) el modelo activo de producción para
    clasificar ventanas de señal o vectores de features ya extraídos.
    """

    def __init__(self) -> None:
        self._cache_key: str | None = None
        self._model = None
        self._scaler = None
        self._metadata = None

    def _ensure_active_model_loaded(self) -> None:
        pointer = mr.get_active_pointer()
        if pointer is None:
            raise RuntimeError("No hay ningún modelo activo en producción.")

        cache_key = f"{pointer['architecture_name']}:{pointer['version_id']}"
        if cache_key == self._cache_key:
            return

        model, scaler, metadata = mr.load_model(
            pointer["architecture_name"], pointer["version_id"]
        )
        self._model, self._scaler, self._metadata = model, scaler, metadata
        self._cache_key = cache_key
        logger.info("Modelo de inferencia (re)cargado", extra={"cache_key": cache_key})

    def predict_from_features(self, features: dict[str, float]) -> StressInferenceResult:
        self._ensure_active_model_loaded()
        assert self._metadata is not None

        if self._metadata.architecture_name != "features_mlp":
            raise NotImplementedError(
                "La inferencia por vector de features solo está implementada "
                "para la arquitectura 'features_mlp'."
            )

        expected = (self._metadata.extra or {}).get("feature_names")
        if expected:
            missing = set(expected) - set(features)
            if missing:
                raise ValueError(f"Faltan features requeridas: {sorted(missing)}")
            vector = np.array([[features[name] for name in expected]])
        else:
            vector = np.array([[v for v in features.values()]])

        if self._scaler is not None:
            vector = self._scaler.transform(vector)

        return self._predict_vector(vector)

    def predict_from_window(self, window: SignalWindow) -> StressInferenceResult:
        """Extrae features de una ventana cruda (mismo código que
        entrenamiento) y clasifica. Punto de entrada natural para una
        futura integración con ingesta biométrica en vivo.
        """
        features = extract_features(window)
        return self.predict_from_features(features)

    def _predict_vector(self, vector: np.ndarray) -> StressInferenceResult:
        assert self._model is not None and self._metadata is not None
        probabilities = self._model.predict(vector, verbose=0)[0]
        predicted_idx = int(np.argmax(probabilities))

        return StressInferenceResult(
            predicted_class=self._metadata.class_names[predicted_idx],
            probabilities={
                name: float(p) for name, p in zip(self._metadata.class_names, probabilities)
            },
            model_version_id=self._metadata.version_id,
            architecture_name=self._metadata.architecture_name,
        )


# Instancia única del proceso: evita recargar el modelo en cada request.
stress_inference_engine = StressInferenceEngine()
