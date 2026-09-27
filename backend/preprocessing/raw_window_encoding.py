"""Codificación de ventanas de señal cruda a un tensor de longitud fija.

La arquitectura CNN-1D+LSTM (Componente 3) consume ventanas multicanal de
longitud fija. A 700 Hz, una ventana de 60s tiene 42,000 muestras: entrenar
sobre esa longitud es viable en GPU de producción, pero deliberadamente
excesivo para CI/desarrollo en CPU. Se remuestrea cada canal a
`RAW_WINDOW_TARGET_LENGTH` (configurable), preservando la forma general de
la señal (relevante para el clasificador) a un costo computacional
tratable. Esta es una decisión de ingeniería explícita, no un recorte de
información oculto: en un despliegue de producción con GPU, basta con subir
`RAW_WINDOW_TARGET_LENGTH` en `.env` sin tocar el código.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample

from backend.preprocessing.windowing import SignalWindow

DEFAULT_RAW_CHANNELS = ("ecg", "eda", "emg", "resp", "temp", "acc_magnitude")
DEFAULT_TARGET_LENGTH = 256


def encode_window(
    window: SignalWindow,
    channel_names: tuple[str, ...] = DEFAULT_RAW_CHANNELS,
    target_length: int = DEFAULT_TARGET_LENGTH,
) -> np.ndarray:
    """Devuelve un array (target_length, n_channels)."""
    encoded = np.zeros((target_length, len(channel_names)), dtype=np.float32)
    for i, channel_name in enumerate(channel_names):
        if channel_name not in window.channels:
            continue  # canal ausente en este dataset: se deja en cero, documentado en metadatos
        encoded[:, i] = resample(window.channels[channel_name], target_length)
    return encoded


def encode_windows_batch(
    windows: list[SignalWindow],
    channel_names: tuple[str, ...] = DEFAULT_RAW_CHANNELS,
    target_length: int = DEFAULT_TARGET_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (X, y_label_names, subject_ids).

    X tiene forma (n_windows, target_length, n_channels).
    """
    X = np.stack([encode_window(w, channel_names, target_length) for w in windows])
    y = np.array([w.label_name for w in windows])
    subject_ids = np.array([w.subject_id for w in windows])
    return X, y, subject_ids
