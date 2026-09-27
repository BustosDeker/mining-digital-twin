"""Loader del dataset WESAD (formato oficial `.pkl` por sujeto).

Cada archivo `SX.pkl` contiene, entre otros campos:
    data['signal']['chest']['ACC']   -> (N, 3) acelerómetro, 700 Hz
    data['signal']['chest']['ECG']   -> (N, 1) electrocardiograma, 700 Hz
    data['signal']['chest']['EMG']   -> (N, 1) electromiograma, 700 Hz
    data['signal']['chest']['EDA']   -> (N, 1) actividad electrodérmica, 700 Hz
    data['signal']['chest']['Temp']  -> (N, 1) temperatura, 700 Hz
    data['signal']['chest']['Resp']  -> (N, 1) respiración, 700 Hz
    data['label']                    -> (N,)   etiqueta por muestra, 700 Hz
    data['subject']                  -> str, p.ej. 'S2'

Taxonomía oficial de etiquetas WESAD:
    0 = no definido/transición, 1 = baseline, 2 = stress, 3 = amusement,
    4 = meditation, 5/6/7 = reservado (ignorar en entrenamiento).

Este loader expone TODAS las etiquetas tal cual (incluidas 0/4/5/6/7) para
que el EDA pueda reportar honestamente la composición real del dataset; el
filtrado a las 3 clases de interés (`settings.STRESS_CLASSES`) ocurre en
`preprocessing/`, no aquí.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from backend.training.dataset_loader import DatasetLoader, SubjectRecording, register_dataset
from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

WESAD_LABEL_NAMES: dict[int, str] = {
    0: "undefined_transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "reserved_5",
    6: "reserved_6",
    7: "reserved_7",
}

_CHEST_CHANNELS = ("ACC", "ECG", "EMG", "EDA", "Temp", "Resp")


@register_dataset("wesad")
class WESADLoader(DatasetLoader):
    dataset_name = "wesad"

    def __init__(self) -> None:
        settings = get_settings()
        self._root_dir: Path = settings.DATA_RAW_DIR / "WESAD"
        self._native_sample_rate_hz = 700  # frecuencia nativa del chest device (RespiBAN)

    def is_available_locally(self) -> bool:
        return self._root_dir.is_dir() and any(self._root_dir.glob("S*/S*.pkl"))

    def list_subjects(self) -> list[str]:
        if not self.is_available_locally():
            return []
        return sorted(
            p.stem for p in self._root_dir.glob("S*/S*.pkl")
        )

    def load_subject(self, subject_id: str) -> SubjectRecording:
        pkl_path = self._root_dir / subject_id / f"{subject_id}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"No se encontró {pkl_path}. ¿Ejecutó download_wesad.py?")

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        chest = raw["signal"]["chest"]
        channels: dict[str, np.ndarray] = {}
        for channel_name in _CHEST_CHANNELS:
            values = np.asarray(chest[channel_name])
            if values.ndim > 1 and values.shape[1] == 3:
                # Acelerómetro triaxial: se guarda la magnitud (norma L2) como
                # canal escalar único, y cada eje por separado para el EDA.
                channels[f"{channel_name}_x"] = values[:, 0]
                channels[f"{channel_name}_y"] = values[:, 1]
                channels[f"{channel_name}_z"] = values[:, 2]
                channels[f"{channel_name}_magnitude"] = np.linalg.norm(values, axis=1)
            else:
                channels[channel_name.lower()] = values.reshape(-1)

        labels = np.asarray(raw["label"]).reshape(-1)

        return SubjectRecording(
            subject_id=subject_id,
            dataset_name=self.dataset_name,
            sample_rate_hz=self._native_sample_rate_hz,
            channels=channels,
            labels=labels,
            label_names=WESAD_LABEL_NAMES,
            metadata={"source_file": str(pkl_path)},
        )
