"""Tests de la Fase 5: loaders, filtrado, ventaneo, normalización, features y EDA."""

import numpy as np
import pytest

from backend.preprocessing.features import extract_features, extract_features_batch
from backend.preprocessing.filtering import filter_all_channels
from backend.preprocessing.imbalance import compute_class_weights
from backend.preprocessing.normalization import normalize_recording
from backend.preprocessing.resampling import resample_channel
from backend.preprocessing.windowing import create_windows, create_windows_for_all_subjects
from backend.training import synthetic_wesad  # noqa: F401 - registra el dataset
from backend.training.dataset_loader import DATASET_REGISTRY, get_loader
from backend.training.eda import run_eda


@pytest.fixture(scope="module")
def synthetic_loader():
    return get_loader("wesad_synthetic_demo")


@pytest.fixture(scope="module")
def small_recording(synthetic_loader):
    return synthetic_loader.load_subject(synthetic_loader.list_subjects()[0])


def test_datasets_are_registered_by_configuration():
    assert "wesad" in DATASET_REGISTRY
    assert "wesad_synthetic_demo" in DATASET_REGISTRY


def test_synthetic_loader_produces_valid_schema(small_recording):
    assert small_recording.sample_rate_hz == 700
    expected_channels = {"ecg", "eda", "emg", "resp", "temp", "acc_x", "acc_y", "acc_z", "acc_magnitude"}
    assert expected_channels.issubset(small_recording.channels.keys())
    assert small_recording.n_samples() == len(small_recording.labels)


def test_synthetic_stress_has_higher_eda_than_baseline(small_recording):
    mask_baseline = small_recording.labels == 1
    mask_stress = small_recording.labels == 2
    eda_baseline = small_recording.channels["eda"][mask_baseline].mean()
    eda_stress = small_recording.channels["eda"][mask_stress].mean()
    assert eda_stress > eda_baseline


def test_resample_channel_preserves_approximate_duration():
    values = np.sin(np.linspace(0, 20 * np.pi, 7000))
    resampled = resample_channel(values, original_hz=700, target_hz=100)
    expected_len = int(len(values) * 100 / 700)
    assert abs(len(resampled) - expected_len) <= 2


def test_filter_all_channels_preserves_shape(small_recording):
    filtered = filter_all_channels(small_recording.channels, small_recording.sample_rate_hz)
    for name in small_recording.channels:
        assert filtered[name].shape == small_recording.channels[name].shape


def test_windowing_only_keeps_pure_windows(small_recording):
    windows = create_windows(small_recording, window_seconds=30, overlap=0.5)
    assert len(windows) > 0
    valid_names = {"baseline", "stress", "amusement"}
    assert all(w.label_name in valid_names for w in windows)


def test_windowing_respects_overlap_step(small_recording):
    windows_no_overlap = create_windows(small_recording, window_seconds=30, overlap=0.0)
    windows_with_overlap = create_windows(small_recording, window_seconds=30, overlap=0.5)
    assert len(windows_with_overlap) > len(windows_no_overlap)


def test_normalization_yields_zero_mean_unit_std(small_recording):
    normalized = normalize_recording(small_recording)
    for values in normalized.channels.values():
        assert abs(float(np.mean(values))) < 1e-4
        assert abs(float(np.std(values)) - 1.0) < 1e-4


def test_feature_extraction_batch_has_no_nan(synthetic_loader):
    recordings = [synthetic_loader.load_subject(sid) for sid in synthetic_loader.list_subjects()[:2]]
    windows = create_windows_for_all_subjects(recordings, window_seconds=60, overlap=0.5)
    X, feature_names, y, subject_ids = extract_features_batch(windows)
    assert X.shape[0] == len(windows)
    assert X.shape[1] == len(feature_names)
    assert not np.isnan(X).any()
    assert len(y) == len(windows)
    assert len(subject_ids) == len(windows)


def test_class_weights_favor_minority_class(synthetic_loader):
    recordings = [synthetic_loader.load_subject(sid) for sid in synthetic_loader.list_subjects()[:2]]
    windows = create_windows_for_all_subjects(recordings, window_seconds=60, overlap=0.5)
    label_to_id = {"baseline": 0, "stress": 1, "amusement": 2}
    y_int = np.array([label_to_id[w.label_name] for w in windows])
    weights = compute_class_weights(y_int)
    counts = {c: int(np.sum(y_int == c)) for c in weights}
    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)
    if counts[minority_class] != counts[majority_class]:
        assert weights[minority_class] > weights[majority_class]


def test_run_eda_generates_all_expected_artifacts(synthetic_loader, tmp_path):
    summary = run_eda(synthetic_loader, subject_limit=3)
    from pathlib import Path

    artifacts_dir = Path(summary["artifacts_dir"])
    expected_files = [
        "class_distribution.png",
        "class_distribution_aggregate.csv",
        "example_signals.png",
        "artifact_detection_report.json",
        "feature_correlation_heatmap.png",
        "quality_report.json",
        "eda_summary.json",
    ]
    for filename in expected_files:
        assert (artifacts_dir / filename).exists(), f"Falta artefacto: {filename}"

    assert summary["quality_report"]["is_synthetic"] is True
    assert summary["quality_report"]["n_subjects"] == 3
