"""Ventaneo de señal con overlap, para construir el dataset de entrenamiento
del clasificador de estrés a partir de señales continuas etiquetadas.

Cada ventana conserva la etiqueta mayoritaria si su pureza (fracción de
muestras con esa etiqueta dentro de la ventana) supera `purity_threshold`;
ventanas de transición entre condiciones (mezcla de clases) se descartan,
ya que introducirían ruido de etiquetado en el entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.training.dataset_loader import SubjectRecording
from backend.utils.config import get_settings


@dataclass
class SignalWindow:
    """Una ventana de señal multicanal con una única etiqueta de clase."""

    subject_id: str
    dataset_name: str
    channels: dict[str, np.ndarray]
    label_name: str
    start_sample: int
    sample_rate_hz: int


def create_windows(
    recording: SubjectRecording,
    window_seconds: float | None = None,
    overlap: float | None = None,
    valid_class_names: list[str] | None = None,
    purity_threshold: float = 0.9,
) -> list[SignalWindow]:
    settings = get_settings()
    window_seconds = window_seconds if window_seconds is not None else settings.WINDOW_SECONDS
    overlap = overlap if overlap is not None else settings.WINDOW_OVERLAP
    valid_class_names = valid_class_names or settings.STRESS_CLASSES

    window_length = int(round(window_seconds * recording.sample_rate_hz))
    step = max(1, int(round(window_length * (1.0 - overlap))))

    n_samples = recording.n_samples()
    windows: list[SignalWindow] = []

    start = 0
    while start + window_length <= n_samples:
        end = start + window_length
        label_segment = recording.labels[start:end]

        values, counts = np.unique(label_segment, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_label_id = int(values[majority_idx])
        purity = counts[majority_idx] / len(label_segment)

        majority_label_name = recording.label_names.get(majority_label_id, "unknown")

        if purity >= purity_threshold and majority_label_name in valid_class_names:
            window_channels = {
                name: signal[start:end] for name, signal in recording.channels.items()
            }
            windows.append(
                SignalWindow(
                    subject_id=recording.subject_id,
                    dataset_name=recording.dataset_name,
                    channels=window_channels,
                    label_name=majority_label_name,
                    start_sample=start,
                    sample_rate_hz=recording.sample_rate_hz,
                )
            )

        start += step

    return windows


def create_windows_for_all_subjects(
    recordings: list[SubjectRecording],
    window_seconds: float | None = None,
    overlap: float | None = None,
    valid_class_names: list[str] | None = None,
    purity_threshold: float = 0.9,
) -> list[SignalWindow]:
    all_windows: list[SignalWindow] = []
    for recording in recordings:
        all_windows.extend(
            create_windows(recording, window_seconds, overlap, valid_class_names, purity_threshold)
        )
    return all_windows
