"""Normalización por sujeto (z-score).

La variabilidad inter-sujeto en señales biométricas es alta (línea base de
EDA, frecuencia cardiaca en reposo, etc. difieren mucho entre personas), por
lo que se normaliza cada canal usando la media/desviación estándar de ESE
MISMO sujeto (no la del dataset completo), evitando que el modelo aprenda
simplemente a distinguir sujetos en vez de estados de estrés.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from backend.training.dataset_loader import SubjectRecording

_EPSILON = 1e-8


def compute_subject_stats(recording: SubjectRecording) -> dict[str, tuple[float, float]]:
    """Media y desviación estándar por canal, calculadas sobre toda la
    grabación del sujeto (todas las condiciones), para no sesgar la
    normalización hacia ninguna clase en particular.
    """
    return {
        name: (float(np.mean(values)), float(np.std(values)))
        for name, values in recording.channels.items()
    }


def normalize_recording(recording: SubjectRecording) -> SubjectRecording:
    stats = compute_subject_stats(recording)
    normalized_channels = {
        name: (values - stats[name][0]) / (stats[name][1] + _EPSILON)
        for name, values in recording.channels.items()
    }
    return replace(recording, channels=normalized_channels)


def normalize_all_subjects(recordings: list[SubjectRecording]) -> list[SubjectRecording]:
    return [normalize_recording(r) for r in recordings]
