"""Tests de la Fase 6: arquitecturas Keras, CV, tuning y Model Registry."""

import numpy as np
import pytest

from backend.preprocessing.features import extract_features_batch
from backend.preprocessing.raw_window_encoding import encode_windows_batch
from backend.preprocessing.windowing import create_windows_for_all_subjects
from backend.services import model_registry as mr
from backend.training import synthetic_wesad  # noqa: F401
from backend.training.architectures import (
    CNNLSTMHyperparams,
    FeaturesMLPHyperparams,
    build_cnn_lstm_model,
    build_features_mlp_model,
)
from backend.training.cross_validation import get_cv_splits, loso_splits
from backend.training.dataset_loader import get_loader
from backend.training.hyperparameter_tuning import tune_architecture
from backend.training.train import fit_final_model, train_cv


@pytest.fixture(scope="module")
def small_windows_and_features():
    loader = get_loader("wesad_synthetic_demo")
    subjects = loader.list_subjects()[:4]
    recordings = [loader.load_subject(sid) for sid in subjects]
    windows = create_windows_for_all_subjects(recordings, window_seconds=60, overlap=0.5)
    X, feature_names, y, subject_ids = extract_features_batch(windows)
    return windows, X, feature_names, y, subject_ids


def test_build_features_mlp_model_output_shape():
    model = build_features_mlp_model(input_dim=15, n_classes=3, hp=FeaturesMLPHyperparams(hidden_layers=(8,)))
    assert model.output_shape == (None, 3)
    assert model.input_shape == (None, 15)


def test_build_cnn_lstm_model_output_shape():
    hp = CNNLSTMHyperparams(conv_filters=(8,), lstm_units=8, dense_units=8)
    model = build_cnn_lstm_model(input_shape=(64, 6), n_classes=3, hp=hp)
    assert model.output_shape == (None, 3)
    assert model.input_shape == (None, 64, 6)


def test_loso_splits_hold_out_one_subject_per_fold(small_windows_and_features):
    _, _, _, y, subject_ids = small_windows_and_features
    n_unique_subjects = len(set(subject_ids))
    folds = list(loso_splits(y, subject_ids))
    assert len(folds) == n_unique_subjects
    for train_idx, test_idx, held_out in folds:
        assert set(subject_ids[test_idx]) == {held_out}
        assert held_out not in set(subject_ids[train_idx])


def test_get_cv_splits_dispatches_by_strategy(small_windows_and_features):
    _, _, _, y, subject_ids = small_windows_and_features
    loso = get_cv_splits(y, subject_ids, strategy="loso")
    kfold = get_cv_splits(y, subject_ids, strategy="kfold")
    assert len(loso) == len(set(subject_ids))
    assert len(kfold) > 0
    with pytest.raises(ValueError):
        get_cv_splits(y, subject_ids, strategy="not_a_real_strategy")


def test_train_cv_features_mlp_returns_valid_metrics(small_windows_and_features):
    _, X, _, y, subject_ids = small_windows_and_features
    hp = FeaturesMLPHyperparams(hidden_layers=(8,), dropout_rate=0.2)
    result = train_cv("features_mlp", X, y, subject_ids, hyperparams=hp, cv_strategy="loso", epochs=5, batch_size=8)
    assert len(result.fold_results) > 0
    agg = result.aggregate_metrics()
    assert 0.0 <= agg["mean_accuracy"] <= 1.0
    assert 0.0 <= agg["mean_f1_macro"] <= 1.0


def test_train_cv_cnn_lstm_returns_valid_metrics(small_windows_and_features):
    windows, _, _, _, _ = small_windows_and_features
    X, y, subject_ids = encode_windows_batch(windows, target_length=64)
    hp = CNNLSTMHyperparams(conv_filters=(8,), lstm_units=8, dense_units=8)
    result = train_cv("cnn_lstm", X, y, subject_ids, hyperparams=hp, cv_strategy="loso", epochs=3, batch_size=8)
    assert len(result.fold_results) > 0
    agg = result.aggregate_metrics()
    assert 0.0 <= agg["mean_accuracy"] <= 1.0


def test_hyperparameter_tuning_tracks_trials(small_windows_and_features):
    _, X, _, y, subject_ids = small_windows_and_features
    tuning = tune_architecture(
        "features_mlp", X, y, subject_ids, n_trials=2, epochs_per_trial=5, cv_strategy="loso"
    )
    df = tuning.trials_dataframe()
    assert len(df) == 2
    assert 0.0 <= tuning.best_value <= 1.0
    assert tuning.best_cv_result is not None


def test_model_registry_full_cycle(small_windows_and_features, tmp_path, monkeypatch):
    from backend.utils import config as config_module

    config_module.get_settings.cache_clear()
    monkeypatch.setenv("MODELS_REGISTRY_DIR", str(tmp_path / "registry"))
    config_module.get_settings.cache_clear()

    _, X, feature_names, y, _ = small_windows_and_features
    hp = FeaturesMLPHyperparams(hidden_layers=(8,), dropout_rate=0.2)
    model, scaler, class_names = fit_final_model("features_mlp", X, y, hyperparams=hp, epochs=5, batch_size=8)

    metadata = mr.register_model(
        model,
        architecture_name="features_mlp",
        dataset_name="wesad_synthetic_demo",
        hyperparams=hp.to_dict(),
        cv_metrics={"mean_f1_macro": 0.9},
        class_names=class_names,
        scaler=scaler,
    )

    models = mr.list_models("features_mlp")
    assert any(m.version_id == metadata.version_id for m in models)

    mr.activate_model("features_mlp", metadata.version_id)
    pointer = mr.get_active_pointer()
    assert pointer["version_id"] == metadata.version_id

    loaded_model, loaded_scaler, loaded_metadata = mr.load_active_model()
    assert loaded_metadata.version_id == metadata.version_id
    assert loaded_scaler is not None

    X_scaled = scaler.transform(X[:3])
    original_pred = model.predict(X_scaled, verbose=0)
    loaded_pred = loaded_model.predict(loaded_scaler.transform(X[:3]), verbose=0)
    assert np.allclose(original_pred, loaded_pred, atol=1e-5)

    config_module.get_settings.cache_clear()
