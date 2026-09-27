"""Resampleo de señal para unificar frecuencias entre datasets distintos.

Regla del proyecto: el Motor IA debe soportar múltiples datasets con
frecuencias nativas potencialmente distintas bajo un único pipeline de
ventaneo. Este módulo remuestrea cualquier canal a la frecuencia de
referencia (`settings.SIGNAL_SAMPLE_RATE_HZ`).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def resample_channel(values: np.ndarray, original_hz: int, target_hz: int) -> np.ndarray:
    if original_hz == target_hz:
        return values
    from math import gcd

    g = gcd(original_hz, target_hz)
    up, down = target_hz // g, original_hz // g
    return resample_poly(values, up, down)


def resample_all_channels(
    channels: dict[str, np.ndarray], original_hz: int, target_hz: int
) -> dict[str, np.ndarray]:
    if original_hz == target_hz:
        return channels
    return {name: resample_channel(values, original_hz, target_hz) for name, values in channels.items()}


def resample_labels(labels: np.ndarray, original_hz: int, target_hz: int) -> np.ndarray:
    """Las etiquetas son categóricas: se remuestrean por selección del
    vecino más cercano (nunca por interpolación numérica).
    """
    if original_hz == target_hz:
        return labels
    original_n = len(labels)
    target_n = int(round(original_n * target_hz / original_hz))
    indices = np.clip(
        np.round(np.linspace(0, original_n - 1, target_n)).astype(int), 0, original_n - 1
    )
    return labels[indices]
