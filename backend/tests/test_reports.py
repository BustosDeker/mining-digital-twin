"""Tests de la Fase 8: generación de reportes PDF."""

import pytest

from backend.evaluation.architecture_comparison import compare_architectures
from backend.preprocessing.features import extract_features_batch
from backend.preprocessing.windowing import create_windows_for_all_subjects
from backend.reports.pdf_generator import generate_training_report
from backend.training import synthetic_wesad  # noqa: F401
from backend.training.architectures import FeaturesMLPHyperparams
from backend.training.dataset_loader import get_loader
from backend.training.eda import run_eda
from backend.training.train import train_cv


@pytest.fixture(scope="module")
def cv_results():
    loader = get_loader("wesad_synthetic_demo")
    subjects = loader.list_subjects()[:4]
    recordings = [loader.load_subject(sid) for sid in subjects]
    windows = create_windows_for_all_subjects(recordings, window_seconds=60, overlap=0.5)
    X, _, y, subject_ids = extract_features_batch(windows)
    hp = FeaturesMLPHyperparams(hidden_layers=(8,), dropout_rate=0.2)
    result_a = train_cv("features_mlp", X, y, subject_ids, hyperparams=hp, cv_strategy="loso", epochs=5, batch_size=8)
    result_b = train_cv("features_mlp", X, y, subject_ids, hyperparams=hp, cv_strategy="loso", epochs=5, batch_size=8)
    return {"model_a": result_a, "model_b": result_b}


def test_generate_report_with_eda_only():
    loader = get_loader("wesad_synthetic_demo")
    summary = run_eda(loader, subject_limit=3)
    path = generate_training_report(
        "test_report_eda_only.pdf",
        eda_quality_reports={"wesad_synthetic_demo": summary["quality_report"]},
    )
    assert path.exists()
    assert path.stat().st_size > 500  # un PDF válido no está vacío


def test_generate_report_with_architecture_sections(cv_results):
    arch_report = compare_architectures(cv_results, metric_name="f1_macro")
    path = generate_training_report(
        "test_report_architectures.pdf",
        architecture_cv_results=cv_results,
        architecture_comparison=arch_report,
    )
    assert path.exists()
    assert path.stat().st_size > 1000


def test_generate_report_is_valid_pdf_structure():
    path = generate_training_report("test_report_minimal.pdf")
    with open(path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"
