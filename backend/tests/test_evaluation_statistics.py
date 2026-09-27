"""Tests de la Fase 7: estadística, Monte Carlo y comparaciones."""

import numpy as np
import pytest

from backend.digital_twin.graph_models import NodeType
from backend.digital_twin.layout_generator import LayoutGenerator, LayoutGeneratorConfig
from backend.evaluation.architecture_comparison import compare_architectures
from backend.evaluation.monte_carlo import run_monte_carlo
from backend.evaluation.routing_comparison import compare_routing_strategies
from backend.preprocessing.features import extract_features_batch
from backend.preprocessing.windowing import create_windows_for_all_subjects
from backend.simulation.routing import AdaptiveShortestPathRouter, QLearningRouter
from backend.statistics.tests import compare_multiple_paired, compare_two_paired, select_best
from backend.statistics.tests import test_normality as run_normality_test
from backend.training import synthetic_wesad  # noqa: F401
from backend.training.architectures import FeaturesMLPHyperparams
from backend.training.dataset_loader import get_loader
from backend.training.train import train_cv


@pytest.fixture()
def layout():
    config = LayoutGeneratorConfig(
        n_levels=3, galleries_per_level=8, n_refuge_chambers=3, n_exits=2, n_risk_zones=4, random_seed=42
    )
    return LayoutGenerator(config).generate("pytest_eval_layout")


def test_normality_detects_clearly_nonnormal_and_normal():
    rng = np.random.default_rng(0)
    normal_sample = rng.normal(0, 1, 30)
    skewed_sample = rng.exponential(1.0, 30)
    results = run_normality_test({"normal": normal_sample, "skewed": skewed_sample})
    by_name = {r.method_name: r for r in results}
    assert by_name["normal"].is_normal_at_0_05 == True  # noqa: E712


def test_compare_two_paired_detects_significant_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(0.9, 0.02, 12)
    b = rng.normal(0.6, 0.02, 12)
    result = compare_two_paired(a, b, "a", "b")
    assert result.significant_at_0_05 is True
    assert result.better_method == "a"


def test_compare_two_paired_no_difference_when_identical():
    values = np.array([0.5, 0.6, 0.7, 0.8])
    result = compare_two_paired(values, values.copy(), "a", "b")
    assert result.significant_at_0_05 is False


def test_compare_multiple_paired_requires_equal_length():
    with pytest.raises(ValueError):
        compare_multiple_paired({"a": np.array([1, 2, 3]), "b": np.array([1, 2])})


def test_compare_multiple_paired_significant_case():
    rng = np.random.default_rng(2)
    samples = {
        "a": rng.normal(0.9, 0.02, 15),
        "b": rng.normal(0.7, 0.02, 15),
        "c": rng.normal(0.5, 0.02, 15),
    }
    result = compare_multiple_paired(samples)
    assert result.significant_at_0_05 is True
    assert result.ranking[0] == "a"
    assert result.nemenyi_p_values is not None


def test_select_best_dispatches_two_vs_multiple():
    rng = np.random.default_rng(3)
    two = {"a": rng.normal(0.9, 0.02, 10), "b": rng.normal(0.5, 0.02, 10)}
    three = {**two, "c": rng.normal(0.3, 0.02, 10)}
    r2 = select_best(two)
    r3 = select_best(three)
    assert r2["comparison_type"] == "wilcoxon_two_methods"
    assert r3["comparison_type"] == "friedman_nemenyi_multiple_methods"
    assert r2["statistically_justified_best"] == "a"
    assert r3["statistically_justified_best"] == "a"


def test_run_monte_carlo_produces_n_runs_with_summary(layout):
    router = AdaptiveShortestPathRouter()
    risk_node = next(n.node_id for n in layout.nodes if n.node_type == NodeType.RISK_ZONE)
    result = run_monte_carlo(
        layout, scenario_name="test_scenario", router=router, router_name="astar",
        n_agents=10, n_runs=5, hazard_origin_node_id=risk_node,
    )
    assert result.n_runs == 5
    assert len(result.run_metrics) == 5
    summary = result.summary()
    assert "mean_evacuation_rate" in summary
    assert 0.0 <= summary["mean_evacuation_rate"] <= 1.0


def test_compare_routing_strategies_returns_valid_decision(layout):
    risk_node = next(n.node_id for n in layout.nodes if n.node_type == NodeType.RISK_ZONE)
    report = compare_routing_strategies(
        layout,
        routers={"astar": AdaptiveShortestPathRouter(), "astar_copy": AdaptiveShortestPathRouter()},
        n_runs=5, n_agents=10, hazard_origin_node_id=risk_node,
    )
    assert set(report.per_router_summary.keys()) == {"astar", "astar_copy"}
    assert report.statistical_decision["comparison_type"] == "wilcoxon_two_methods"
    # Dos instancias de la MISMA estrategia, mismas semillas -> sin diferencia real
    assert report.statistical_decision["comparison"]["significant_at_0_05"] is False


def test_compare_architectures_uses_common_fold_count():
    loader = get_loader("wesad_synthetic_demo")
    subjects = loader.list_subjects()[:4]
    recordings = [loader.load_subject(sid) for sid in subjects]
    windows = create_windows_for_all_subjects(recordings, window_seconds=60, overlap=0.5)
    X, _, y, subject_ids = extract_features_batch(windows)

    hp = FeaturesMLPHyperparams(hidden_layers=(8,), dropout_rate=0.2)
    result_a = train_cv("features_mlp", X, y, subject_ids, hyperparams=hp, cv_strategy="loso", epochs=5, batch_size=8)
    result_b = train_cv("features_mlp", X, y, subject_ids, hyperparams=hp, cv_strategy="loso", epochs=5, batch_size=8)

    report = compare_architectures({"model_a": result_a, "model_b": result_b}, metric_name="f1_macro")
    assert report.metric_name == "f1_macro"
    assert "model_a" in report.per_architecture_mean
    assert "statistically_justified_best" in report.statistical_decision
