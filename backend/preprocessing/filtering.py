"""Filtrado de señal fisiológica cruda, por tipo de canal.

Frecuencias de corte estándar de la literatura de procesamiento de señales
biométricas (no específicas de WESAD, aplicables a cualquier dataset con
canales homólogos, cumpliendo la regla de reutilización multi-dataset):
    - ECG: pasa-banda 0.5-40 Hz (elimina deriva de línea base y ruido EMG).
    - EMG: pasa-banda 20-450 Hz (o hasta Nyquist si la frecuencia de
      muestreo es menor).
    - EDA: pasa-bajos 5 Hz (la actividad electrodérmica es de variación lenta).
    - Resp/Temp: pasa-bajos 1 Hz.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

_FILTER_SPECS: dict[str, tuple[str, float | tuple[float, float]]] = {
    "ecg": ("bandpass", (0.5, 40.0)),
    "emg": ("bandpass", (20.0, 200.0)),
    "eda": ("lowpass", 5.0),
    "resp": ("lowpass", 1.0),
    "temp": ("lowpass", 1.0),
}


def _design_filter(kind: str, cutoff, sample_rate_hz: int, order: int = 4):
    nyquist = sample_rate_hz / 2.0
    if kind == "bandpass":
        low, high = cutoff
        high = min(high, nyquist * 0.99)
        return butter(order, [low / nyquist, high / nyquist], btype="bandpass")
    if kind == "lowpass":
        cutoff = min(cutoff, nyquist * 0.99)
        return butter(order, cutoff / nyquist, btype="lowpass")
    raise ValueError(f"Tipo de filtro no soportado: {kind}")


def filter_channel(channel_name: str, values: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Aplica el filtro apropiado según el tipo de canal (por nombre base,
    p.ej. 'ecg', 'eda'); canales no reconocidos (acelerómetro, etc.) se
    devuelven sin cambios, ya que no está definido un filtro estándar único
    para ellos en este pipeline.
    """
    base_name = channel_name.lower().split("_")[0]
    if base_name not in _FILTER_SPECS:
        return values
    if len(values) < 15:  # filtfilt requiere un mínimo de muestras
        return values

    kind, cutoff = _FILTER_SPECS[base_name]
    b, a = _design_filter(kind, cutoff, sample_rate_hz)
    return filtfilt(b, a, values)


def filter_all_channels(channels: dict[str, np.ndarray], sample_rate_hz: int) -> dict[str, np.ndarray]:
    return {name: filter_channel(name, values, sample_rate_hz) for name, values in channels.items()}
