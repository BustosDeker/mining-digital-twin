"""Manejo de desbalance de clases: pesos de clase y SMOTE (solo tabular).

WESAD está desbalanceado (más muestras de 'baseline' que de 'amusement',
por ejemplo). Se ofrecen dos estrategias, seleccionables por configuración:

- `class weights`: aplicable a CUALQUIER arquitectura (ventanas crudas o
  features tabulares), se usa directamente en la función de pérdida.
- SMOTE: solo aplicable a vectores de features tabulares de longitud fija
  (arquitectura features+MLP); no tiene sentido geométrico sobre ventanas
  de señal cruda multicanal de alta dimensionalidad, por lo que NO se
  ofrece para la rama CNN-1D+LSTM (se documenta esta limitación
  explícitamente en vez de aplicar SMOTE de forma inválida).
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Pesos de clase inversamente proporcionales a la frecuencia, para usar
    en `class_weight` de Keras (soporta ambas arquitecturas).
    """
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def apply_smote_to_features(
    features: np.ndarray, labels: np.ndarray, random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Sobremuestreo SMOTE sobre vectores de features tabulares (2D:
    n_muestras x n_features). Requiere `imbalanced-learn`.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "SMOTE requiere el paquete 'imbalanced-learn'. Instálelo con: "
            "pip install imbalanced-learn --break-system-packages"
        ) from exc

    if features.ndim != 2:
        raise ValueError(
            "SMOTE solo aplica a vectores de features tabulares (2D). "
            "Para ventanas de señal cruda (CNN-1D+LSTM), use class weights."
        )

    smote = SMOTE(random_state=random_seed)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    return features_resampled, labels_resampled
