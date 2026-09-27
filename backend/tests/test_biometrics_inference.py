"""Test del wrapper de inferencia (backend/biometrics/inference.py)."""

import pytest

from backend.biometrics.inference import StressInferenceEngine
from backend.preprocessing.features import extract_features_batch
from backend.preprocessing.windowing import create_windows_for_all_subjects
from backend.services import model_registry as mr
from backend.training import synthetic_wesad  # noqa: F401
from backend.training.architectures import FeaturesMLPHyperparams
from backend.training.dataset_loader import get_loader
from backend.training.train import fit_final_model


@pytest.fixture()
def registered_active_model(tmp_path, monkeypatch):
    from backend.utils import config as config_module

    config_module.get_settings.cache_clear()
    monkeypatch.setenv("MODELS_REGISTRY_DIR", str(tmp_path / "registry"))
    config_module.get_settings.cache_clear()

    loader = get_loader("wesad_synthetic_demo")
    recordings = [loader.load_subject(sid) for sid in loader.list_subjects()[:4]]
    windows = create_windows_for_all_subjects(recordings, window_seconds=60, overlap=0.5)
    X, feature_names, y, _ = extract_features_batch(windows)

    hp = FeaturesMLPHyperparams(hidden_layers=(8,), dropout_rate=0.2)
    model, scaler, class_names = fit_final_model("features_mlp", X, y, hyperparams=hp, epochs=5, batch_size=8)
    metadata = mr.register_model(
        model, architecture_name="features_mlp", dataset_name="wesad_synthetic_demo",
        hyperparams=hp.to_dict(), cv_metrics={"mean_f1_macro": 0.9}, class_names=class_names,
        scaler=scaler, extra={"feature_names": feature_names},
    )
    mr.activate_model("features_mlp", metadata.version_id)

    yield feature_names, class_names
    config_module.get_settings.cache_clear()


def test_predict_from_features_returns_valid_result(registered_active_model):
    feature_names, class_names = registered_active_model
    engine = StressInferenceEngine()

    features = {name: 0.5 for name in feature_names}
    result = engine.predict_from_features(features)

    assert result.predicted_class in class_names
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-3
    assert set(result.probabilities.keys()) == set(class_names)


def test_predict_from_features_raises_on_missing_features(registered_active_model):
    feature_names, _ = registered_active_model
    engine = StressInferenceEngine()

    incomplete = {name: 0.5 for name in feature_names[:-1]}  # falta una feature
    with pytest.raises(ValueError):
        engine.predict_from_features(incomplete)


def test_predict_without_any_active_model_raises(tmp_path, monkeypatch):
    from backend.utils import config as config_module

    config_module.get_settings.cache_clear()
    monkeypatch.setenv("MODELS_REGISTRY_DIR", str(tmp_path / "empty_registry"))
    config_module.get_settings.cache_clear()

    engine = StressInferenceEngine()
    with pytest.raises(RuntimeError):
        engine.predict_from_features({"eda_mean": 0.1})

    config_module.get_settings.cache_clear()
