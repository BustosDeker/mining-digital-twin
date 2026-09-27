"""Generador SINTÉTICO con el esquema exacto de canales de WESAD.

ADVERTENCIA (léase antes de usar): este dataset es 100% sintético. Existe
únicamente para poder ejecutar y validar el pipeline completo del Motor IA
(EDA, preprocesamiento, entrenamiento, CV) en este entorno de desarrollo,
que no tiene acceso de red al servidor de descarga real de WESAD.

Se registra bajo el nombre `"wesad_synthetic_demo"`, deliberadamente
distinto de `"wesad"`, para que sea imposible activarlo por accidente vía
`settings.DATASETS=["wesad"]` y para que ninguna cifra generada con él
pueda confundirse con un resultado real. Regla fija del proyecto: ninguna
cifra cuantitativa del artículo puede provenir de este dataset.

Las señales se generan con propiedades estadísticas *plausibles* (mayor
tono simpático — EDA y frecuencia cardiaca más altas — bajo la condición
"stress" que bajo "baseline"/"amusement"), pero no representan sujetos
reales ni deben usarse para ninguna conclusión científica.
"""

from __future__ import annotations

import numpy as np

from backend.training.dataset_loader import DatasetLoader, SubjectRecording, register_dataset
from backend.training.wesad_loader import WESAD_LABEL_NAMES

_SAMPLE_RATE_HZ = 700
_SEGMENT_SECONDS = {"baseline": 300, "stress": 300, "amusement": 200, "undefined_transient": 30}
_SEGMENT_ORDER = ["undefined_transient", "baseline", "undefined_transient", "stress", "undefined_transient", "amusement"]
_LABEL_ID_BY_NAME = {v: k for k, v in WESAD_LABEL_NAMES.items()}


def _simulate_ecg_like(n_samples: int, rng: np.random.Generator, heart_rate_bpm: float) -> np.ndarray:
    """Señal cuasi-periódica simple que imita la morfología general de un
    ECG (no es un simulador electrofisiológico real; solo aporta
    periodicidad y ruido realistas para el pipeline de EDA/ventaneo).
    """
    t = np.arange(n_samples) / _SAMPLE_RATE_HZ
    beat_freq_hz = heart_rate_bpm / 60.0
    signal = np.sin(2 * np.pi * beat_freq_hz * t)
    signal += 0.3 * np.sin(2 * np.pi * beat_freq_hz * 2 * t)  # armónico
    signal += rng.normal(0, 0.05, size=n_samples)
    return signal


def _simulate_eda_like(n_samples: int, rng: np.random.Generator, tonic_level: float) -> np.ndarray:
    drift = np.cumsum(rng.normal(0, 0.001, size=n_samples))
    return tonic_level + drift + rng.normal(0, 0.02, size=n_samples)


@register_dataset("wesad_synthetic_demo")
class SyntheticWESADLoader(DatasetLoader):
    dataset_name = "wesad_synthetic_demo"

    def __init__(self, n_subjects: int = 8, random_seed: int = 42) -> None:
        self.n_subjects = n_subjects
        self._random_seed = random_seed

    def is_available_locally(self) -> bool:
        return True  # se genera en memoria bajo demanda, no requiere descarga

    def list_subjects(self) -> list[str]:
        return [f"SYN{i}" for i in range(1, self.n_subjects + 1)]

    def load_subject(self, subject_id: str) -> SubjectRecording:
        subject_index = int(subject_id.replace("SYN", ""))
        rng = np.random.default_rng(self._random_seed + subject_index)

        base_hr = rng.uniform(60, 75)  # frecuencia cardiaca base del sujeto (bpm)
        base_eda = rng.uniform(2.0, 4.0)  # nivel tónico EDA base del sujeto (µS)

        segments_signals: dict[str, list[np.ndarray]] = {
            "acc_x": [], "acc_y": [], "acc_z": [],
            "ecg": [], "emg": [], "eda": [], "temp": [], "resp": [],
        }
        label_segments: list[np.ndarray] = []

        for condition in _SEGMENT_ORDER:
            n_samples = _SEGMENT_SECONDS[condition] * _SAMPLE_RATE_HZ
            label_id = _LABEL_ID_BY_NAME[condition]

            if condition == "stress":
                hr, eda_level, emg_amp = base_hr * 1.35, base_eda * 1.8, 0.15
            elif condition == "amusement":
                hr, eda_level, emg_amp = base_hr * 1.15, base_eda * 1.3, 0.08
            elif condition == "baseline":
                hr, eda_level, emg_amp = base_hr, base_eda, 0.03
            else:  # undefined_transient
                hr, eda_level, emg_amp = base_hr * 1.05, base_eda * 1.05, 0.05

            segments_signals["ecg"].append(_simulate_ecg_like(n_samples, rng, hr))
            segments_signals["eda"].append(_simulate_eda_like(n_samples, rng, eda_level))
            segments_signals["emg"].append(rng.normal(0, emg_amp, size=n_samples))
            segments_signals["temp"].append(31.0 + rng.normal(0, 0.1, size=n_samples).cumsum() * 0.0001)
            segments_signals["resp"].append(np.sin(2 * np.pi * 0.25 * np.arange(n_samples) / _SAMPLE_RATE_HZ) + rng.normal(0, 0.05, n_samples))
            acc_noise_scale = 0.02 if condition != "stress" else 0.08
            for axis in ("acc_x", "acc_y", "acc_z"):
                segments_signals[axis].append(rng.normal(0, acc_noise_scale, size=n_samples))

            label_segments.append(np.full(n_samples, label_id, dtype=np.int64))

        channels = {name: np.concatenate(chunks) for name, chunks in segments_signals.items()}
        acc_stack = np.stack([channels["acc_x"], channels["acc_y"], channels["acc_z"]], axis=1)
        channels["acc_magnitude"] = np.linalg.norm(acc_stack, axis=1)
        labels = np.concatenate(label_segments)

        return SubjectRecording(
            subject_id=subject_id,
            dataset_name=self.dataset_name,
            sample_rate_hz=_SAMPLE_RATE_HZ,
            channels=channels,
            labels=labels,
            label_names=WESAD_LABEL_NAMES,
            metadata={"synthetic": True, "base_heart_rate_bpm": base_hr, "base_eda_level": base_eda},
        )
