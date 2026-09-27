"""EDA (Análisis Exploratorio de Datos) completo por dataset.

Genera, para cualquier dataset registrado en `DATASET_REGISTRY`:
    1. Distribución de clases (global y por sujeto).
    2. Estadísticas por canal (media, std, min, max) por clase.
    3. Señales de ejemplo por canal y clase.
    4. Detección de artefactos (flatline, clipping/saturación).
    5. Mapa de correlaciones entre canales (a nivel de features por ventana).
    6. Reporte de calidad (completitud, sujetos, balance).

Todos los artefactos (figuras + tablas) se guardan en
`settings.ARTIFACTS_DIR / <dataset_name> / eda /`, para ser servidos luego
vía FastAPI (Fase 9) sin volver a computarlos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend sin display, apto para servidor/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backend.preprocessing.features import extract_features_batch
from backend.preprocessing.windowing import create_windows_for_all_subjects
from backend.training.dataset_loader import DatasetLoader, SubjectRecording
from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")

_FLATLINE_STD_THRESHOLD = 1e-4
_FLATLINE_WINDOW_SAMPLES = 700  # 1 segundo a 700 Hz


def _artifacts_dir(dataset_name: str) -> Path:
    settings = get_settings()
    out_dir = settings.ARTIFACTS_DIR / dataset_name / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# --------------------------------------------------------------------
# 1. Distribución de clases
# --------------------------------------------------------------------
def _class_distribution(recordings: list[SubjectRecording], out_dir: Path) -> dict[str, Any]:
    rows = []
    for rec in recordings:
        values, counts = np.unique(rec.labels, return_counts=True)
        for v, c in zip(values, counts):
            rows.append(
                {
                    "subject_id": rec.subject_id,
                    "label_id": int(v),
                    "label_name": rec.label_names.get(int(v), "unknown"),
                    "n_samples": int(c),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "class_distribution_by_subject.csv", index=False)

    aggregate = df.groupby("label_name")["n_samples"].sum().sort_values(ascending=False)
    aggregate.to_csv(out_dir / "class_distribution_aggregate.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    aggregate.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(aggregate)))
    ax.set_ylabel("Número de muestras (a frecuencia nativa)")
    ax.set_title(f"Distribución de clases — {recordings[0].dataset_name}")
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    return {"aggregate_counts": aggregate.to_dict(), "n_subjects": len(recordings)}


# --------------------------------------------------------------------
# 2. Estadísticas por canal y por clase
# --------------------------------------------------------------------
def _per_channel_stats_by_class(
    recordings: list[SubjectRecording], valid_class_names: list[str], out_dir: Path
) -> None:
    rows = []
    for rec in recordings:
        for label_id, label_name in rec.label_names.items():
            if label_name not in valid_class_names:
                continue
            mask = rec.labels == label_id
            if not mask.any():
                continue
            for channel_name, values in rec.channels.items():
                segment = values[mask]
                rows.append(
                    {
                        "subject_id": rec.subject_id,
                        "label_name": label_name,
                        "channel": channel_name,
                        "mean": float(np.mean(segment)),
                        "std": float(np.std(segment)),
                        "min": float(np.min(segment)),
                        "max": float(np.max(segment)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_channel_stats_by_class.csv", index=False)

    summary = df.groupby(["channel", "label_name"])[["mean", "std"]].mean().reset_index()
    summary.to_csv(out_dir / "per_channel_stats_summary.csv", index=False)


# --------------------------------------------------------------------
# 3. Señales de ejemplo
# --------------------------------------------------------------------
def _example_signals(
    recordings: list[SubjectRecording], valid_class_names: list[str], out_dir: Path
) -> None:
    rec = recordings[0]
    plot_channels = [c for c in ("ecg", "eda", "emg", "resp", "temp") if c in rec.channels]
    if not plot_channels:
        plot_channels = list(rec.channels.keys())[:5]

    fig, axes = plt.subplots(len(plot_channels), 1, figsize=(11, 2.2 * len(plot_channels)), sharex=False)
    if len(plot_channels) == 1:
        axes = [axes]

    name_to_id = {v: k for k, v in rec.label_names.items()}
    window_samples = int(10 * rec.sample_rate_hz)  # 10 segundos de ejemplo

    for ax, channel_name in zip(axes, plot_channels):
        for class_name in valid_class_names:
            label_id = name_to_id.get(class_name)
            if label_id is None:
                continue
            mask = rec.labels == label_id
            idx = np.where(mask)[0]
            if len(idx) < window_samples:
                continue
            start = idx[len(idx) // 2]
            segment = rec.channels[channel_name][start : start + window_samples]
            t = np.arange(len(segment)) / rec.sample_rate_hz
            ax.plot(t, segment, label=class_name, alpha=0.85)
        ax.set_ylabel(channel_name)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle(f"Señales de ejemplo (10s) — sujeto {rec.subject_id}")
    fig.tight_layout()
    fig.savefig(out_dir / "example_signals.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------
# 4. Detección de artefactos
# --------------------------------------------------------------------
def _detect_artifacts(recordings: list[SubjectRecording], out_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for rec in recordings:
        subject_report = {}
        for channel_name, values in rec.channels.items():
            n_windows = len(values) // _FLATLINE_WINDOW_SAMPLES
            if n_windows == 0:
                continue
            trimmed = values[: n_windows * _FLATLINE_WINDOW_SAMPLES]
            reshaped = trimmed.reshape(n_windows, _FLATLINE_WINDOW_SAMPLES)
            stds = reshaped.std(axis=1)
            n_flatline = int(np.sum(stds < _FLATLINE_STD_THRESHOLD))

            value_range = np.max(values) - np.min(values)
            clipping_threshold = 0.001 * value_range if value_range > 0 else 0
            n_clipped_high = int(np.sum(values >= np.max(values) - clipping_threshold))
            n_clipped_low = int(np.sum(values <= np.min(values) + clipping_threshold))

            subject_report[channel_name] = {
                "flatline_windows": n_flatline,
                "flatline_windows_pct": round(100 * n_flatline / n_windows, 3),
                "clipped_samples_pct": round(100 * (n_clipped_high + n_clipped_low) / len(values), 4),
            }
        report[rec.subject_id] = subject_report

    with open(out_dir / "artifact_detection_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


# --------------------------------------------------------------------
# 5. Mapa de correlaciones (a nivel de features por ventana)
# --------------------------------------------------------------------
def _correlation_heatmap(
    recordings: list[SubjectRecording], valid_class_names: list[str], out_dir: Path
) -> None:
    windows = create_windows_for_all_subjects(recordings, valid_class_names=valid_class_names)
    if not windows:
        logger.warning("Sin ventanas válidas para el mapa de correlaciones; se omite.")
        return

    X, feature_names, _, _ = extract_features_batch(windows)
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlación entre features (por ventana)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    corr.to_csv(out_dir / "feature_correlation_matrix.csv")


# --------------------------------------------------------------------
# 6. Reporte de calidad
# --------------------------------------------------------------------
def _quality_report(
    recordings: list[SubjectRecording],
    valid_class_names: list[str],
    class_dist: dict[str, Any],
    artifacts: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    total_flatline_pct = np.mean(
        [
            ch_report["flatline_windows_pct"]
            for subj in artifacts.values()
            for ch_report in subj.values()
        ]
    ) if artifacts else 0.0

    valid_counts = {
        name: count
        for name, count in class_dist["aggregate_counts"].items()
        if name in valid_class_names
    }
    total_valid = sum(valid_counts.values()) or 1
    class_balance_ratio = {
        name: round(count / total_valid, 4) for name, count in valid_counts.items()
    }

    report = {
        "dataset_name": recordings[0].dataset_name,
        "n_subjects": len(recordings),
        "subjects": [r.subject_id for r in recordings],
        "channels_present": sorted(recordings[0].channels.keys()),
        "valid_class_counts": valid_counts,
        "class_balance_ratio": class_balance_ratio,
        "mean_flatline_pct_across_channels": round(float(total_flatline_pct), 4),
        "is_synthetic": bool(recordings[0].metadata and recordings[0].metadata.get("synthetic")),
    }

    with open(out_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


# --------------------------------------------------------------------
# Orquestador
# --------------------------------------------------------------------
def run_eda(loader: DatasetLoader, subject_limit: int | None = None) -> dict[str, Any]:
    settings = get_settings()
    if not loader.is_available_locally():
        raise FileNotFoundError(
            f"El dataset '{loader.dataset_name}' no está disponible localmente. "
            "Ejecute el script de descarga correspondiente primero."
        )

    subjects = loader.list_subjects()
    if subject_limit:
        subjects = subjects[:subject_limit]
    recordings = [loader.load_subject(sid) for sid in subjects]

    out_dir = _artifacts_dir(loader.dataset_name)
    valid_class_names = settings.STRESS_CLASSES

    logger.info("Iniciando EDA", extra={"dataset": loader.dataset_name, "n_subjects": len(recordings)})

    class_dist = _class_distribution(recordings, out_dir)
    _per_channel_stats_by_class(recordings, valid_class_names, out_dir)
    _example_signals(recordings, valid_class_names, out_dir)
    artifacts = _detect_artifacts(recordings, out_dir)
    _correlation_heatmap(recordings, valid_class_names, out_dir)
    quality = _quality_report(recordings, valid_class_names, class_dist, artifacts, out_dir)

    summary = {
        "dataset_name": loader.dataset_name,
        "artifacts_dir": str(out_dir),
        "class_distribution": class_dist,
        "quality_report": quality,
    }
    with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("EDA finalizado", extra={"dataset": loader.dataset_name, "artifacts_dir": str(out_dir)})
    return summary
