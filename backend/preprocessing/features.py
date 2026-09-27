"""Extracción de features HRV (variabilidad de frecuencia cardiaca) y EDA
por ventana, usados por la arquitectura "features + MLP" (Componente 3,
segunda arquitectura a comparar frente a CNN-1D+LSTM sobre señal cruda).

Estas features son estándar en la literatura de detección de estrés
biométrico (dominio del tiempo, sin necesidad de librerías especializadas
de HRV para mantener el pipeline con dependencias mínimas y auditable).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from backend.preprocessing.windowing import SignalWindow


def _hrv_features(ecg: np.ndarray, sample_rate_hz: int) -> dict[str, float]:
    min_distance_samples = int(0.3 * sample_rate_hz)  # < 200 bpm máximo fisiológico razonable
    peaks, _ = find_peaks(ecg, distance=min_distance_samples, prominence=np.std(ecg) * 0.5)

    if len(peaks) < 3:
        return {"hrv_mean_hr_bpm": np.nan, "hrv_sdnn_ms": np.nan, "hrv_rmssd_ms": np.nan}

    rr_intervals_s = np.diff(peaks) / sample_rate_hz
    mean_hr_bpm = 60.0 / np.mean(rr_intervals_s)
    sdnn_ms = float(np.std(rr_intervals_s) * 1000)
    rmssd_ms = float(np.sqrt(np.mean(np.diff(rr_intervals_s) ** 2)) * 1000) if len(rr_intervals_s) > 1 else np.nan

    return {"hrv_mean_hr_bpm": float(mean_hr_bpm), "hrv_sdnn_ms": sdnn_ms, "hrv_rmssd_ms": rmssd_ms}


def _eda_features(eda: np.ndarray, sample_rate_hz: int) -> dict[str, float]:
    t = np.arange(len(eda))
    slope = float(np.polyfit(t, eda, deg=1)[0]) if len(eda) > 1 else 0.0
    peaks, _ = find_peaks(eda, prominence=np.std(eda) * 0.3 + 1e-8)

    return {
        "eda_mean": float(np.mean(eda)),
        "eda_std": float(np.std(eda)),
        "eda_slope": slope,
        "eda_n_scr_peaks": float(len(peaks)),
    }


def _emg_features(emg: np.ndarray) -> dict[str, float]:
    return {"emg_mean_abs": float(np.mean(np.abs(emg))), "emg_std": float(np.std(emg))}


def _resp_features(resp: np.ndarray, sample_rate_hz: int) -> dict[str, float]:
    peaks, _ = find_peaks(resp, distance=int(1.0 * sample_rate_hz))
    duration_s = len(resp) / sample_rate_hz
    resp_rate_per_min = (len(peaks) / duration_s) * 60.0 if duration_s > 0 else np.nan
    return {"resp_rate_per_min": float(resp_rate_per_min), "resp_std": float(np.std(resp))}


def _temp_features(temp: np.ndarray) -> dict[str, float]:
    return {"temp_mean": float(np.mean(temp)), "temp_std": float(np.std(temp))}


def extract_features(window: SignalWindow) -> dict[str, float]:
    """Extrae el vector de features HRV/EDA/EMG/Resp/Temp de una ventana.

    Canales ausentes (p.ej. un dataset sin EMG) se omiten con NaN en vez de
    fallar, para que el pipeline soporte datasets con distinto esquema de
    canales sin acoplarse a WESAD específicamente.
    """
    features: dict[str, float] = {}
    ch = window.channels

    if "ecg" in ch:
        features.update(_hrv_features(ch["ecg"], window.sample_rate_hz))
    if "eda" in ch:
        features.update(_eda_features(ch["eda"], window.sample_rate_hz))
    if "emg" in ch:
        features.update(_emg_features(ch["emg"]))
    if "resp" in ch:
        features.update(_resp_features(ch["resp"], window.sample_rate_hz))
    if "temp" in ch:
        features.update(_temp_features(ch["temp"]))
    if "acc_magnitude" in ch:
        features["acc_magnitude_mean"] = float(np.mean(ch["acc_magnitude"]))
        features["acc_magnitude_std"] = float(np.std(ch["acc_magnitude"]))

    return features


def extract_features_batch(windows: list[SignalWindow]) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Convierte una lista de ventanas en una matriz de features (X),
    manejando NaN por imputación de mediana por columna.

    Devuelve: (X, feature_names, y_label_names, subject_ids)
    """
    rows = [extract_features(w) for w in windows]
    feature_names = sorted({k for row in rows for k in row})

    matrix = np.array([[row.get(name, np.nan) for name in feature_names] for row in rows])
    col_medians = np.nanmedian(matrix, axis=0)
    nan_mask = np.isnan(matrix)
    matrix[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    labels = np.array([w.label_name for w in windows])
    subject_ids = np.array([w.subject_id for w in windows])

    return matrix, feature_names, labels, subject_ids
