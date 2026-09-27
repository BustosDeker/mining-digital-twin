"""Pruebas estadísticas para comparar modelos/estrategias sobre múltiples
corridas (folds de CV o corridas Monte Carlo).

Regla del proyecto: la selección del mejor modelo/estrategia se basa en
métricas + significancia estadística, nunca en una sola cifra puntual.

Flujo estándar:
    1. `test_normality` (Shapiro-Wilk) sobre las muestras pareadas de cada
       método, para decidir si procede un test paramétrico o no paramétrico.
    2. Si son 2 métodos: `compare_two_paired` (Wilcoxon signed-rank, ya que
       las corridas SIEMPRE están pareadas — mismo fold/escenario para
       ambos métodos).
    3. Si son ≥3 métodos: `compare_multiple_paired` (Friedman + post-hoc de
       Nemenyi para identificar qué pares difieren significativamente).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scikit_posthocs as sp
from scipy import stats


@dataclass
class NormalityResult:
    method_name: str
    statistic: float
    p_value: float
    is_normal_at_0_05: bool


def test_normality(samples: dict[str, np.ndarray], alpha: float = 0.05) -> list[NormalityResult]:
    """Shapiro-Wilk por método. Requiere al menos 3 muestras por método."""
    results = []
    for name, values in samples.items():
        values = np.asarray(values)
        if len(values) < 3:
            results.append(NormalityResult(name, float("nan"), float("nan"), False))
            continue
        statistic, p_value = stats.shapiro(values)
        results.append(NormalityResult(name, float(statistic), float(p_value), bool(p_value > alpha)))
    return results


@dataclass
class PairwiseComparisonResult:
    method_a: str
    method_b: str
    test_name: str
    statistic: float
    p_value: float
    significant_at_0_05: bool
    mean_a: float
    mean_b: float
    better_method: str | None


def compare_two_paired(
    values_a: np.ndarray, values_b: np.ndarray, name_a: str, name_b: str, alpha: float = 0.05
) -> PairwiseComparisonResult:
    """Wilcoxon signed-rank para dos métodos evaluados sobre las MISMAS
    corridas (folds, escenarios). No asume normalidad.
    """
    values_a, values_b = np.asarray(values_a), np.asarray(values_b)
    diffs = values_a - values_b

    if np.allclose(diffs, 0):
        statistic, p_value = float("nan"), 1.0
    else:
        statistic, p_value = stats.wilcoxon(values_a, values_b)

    significant = p_value < alpha
    better = None
    if significant:
        better = name_a if np.mean(values_a) > np.mean(values_b) else name_b

    return PairwiseComparisonResult(
        method_a=name_a,
        method_b=name_b,
        test_name="wilcoxon_signed_rank",
        statistic=float(statistic),
        p_value=float(p_value),
        significant_at_0_05=bool(significant),
        mean_a=float(np.mean(values_a)),
        mean_b=float(np.mean(values_b)),
        better_method=better,
    )


@dataclass
class MultipleComparisonResult:
    method_names: list[str]
    friedman_statistic: float
    friedman_p_value: float
    significant_at_0_05: bool
    nemenyi_p_values: np.ndarray | None  # matriz NxN, None si Friedman no fue significativo
    mean_by_method: dict[str, float]
    ranking: list[str]  # de mejor a peor, por media


def compare_multiple_paired(
    samples: dict[str, np.ndarray], alpha: float = 0.05
) -> MultipleComparisonResult:
    """Friedman (ómnibus) + Nemenyi post-hoc si es significativo.

    `samples` debe tener la MISMA cantidad de corridas por método (mismos
    folds/escenarios para todos), condición requerida por Friedman.
    """
    method_names = list(samples.keys())
    arrays = [np.asarray(samples[name]) for name in method_names]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"Friedman requiere el mismo número de corridas por método; recibido: "
            f"{ {name: len(a) for name, a in zip(method_names, arrays)} }"
        )

    statistic, p_value = stats.friedmanchisquare(*arrays)
    significant = p_value < alpha

    nemenyi_matrix = None
    if significant:
        data_matrix = np.column_stack(arrays)  # (n_corridas, n_metodos)
        nemenyi_df = sp.posthoc_nemenyi_friedman(data_matrix)
        nemenyi_df.columns = method_names
        nemenyi_df.index = method_names
        nemenyi_matrix = nemenyi_df.values

    mean_by_method = {name: float(np.mean(arr)) for name, arr in zip(method_names, arrays)}
    ranking = sorted(mean_by_method, key=lambda n: mean_by_method[n], reverse=True)

    return MultipleComparisonResult(
        method_names=method_names,
        friedman_statistic=float(statistic),
        friedman_p_value=float(p_value),
        significant_at_0_05=bool(significant),
        nemenyi_p_values=nemenyi_matrix,
        mean_by_method=mean_by_method,
        ranking=ranking,
    )


def select_best(
    samples: dict[str, np.ndarray], alpha: float = 0.05
) -> dict[str, Any]:
    """Punto de entrada único: decide automáticamente entre comparación de
    2 o ≥3 métodos y devuelve una decisión razonada (no solo la media más
    alta), tal como exige la regla del proyecto.
    """
    normality = test_normality(samples, alpha=alpha)
    method_names = list(samples.keys())
    means = {name: float(np.mean(values)) for name, values in samples.items()}
    naive_best = max(means, key=means.get)

    if len(method_names) == 2:
        a, b = method_names
        comparison = compare_two_paired(samples[a], samples[b], a, b, alpha=alpha)
        decision = comparison.better_method or "empate_no_significativo"
        return {
            "comparison_type": "wilcoxon_two_methods",
            "normality": [n.__dict__ for n in normality],
            "comparison": comparison.__dict__,
            "naive_best_by_mean": naive_best,
            "statistically_justified_best": decision,
        }

    comparison = compare_multiple_paired(samples, alpha=alpha)
    decision = comparison.ranking[0] if comparison.significant_at_0_05 else "empate_no_significativo"
    return {
        "comparison_type": "friedman_nemenyi_multiple_methods",
        "normality": [n.__dict__ for n in normality],
        "friedman_statistic": comparison.friedman_statistic,
        "friedman_p_value": comparison.friedman_p_value,
        "significant": comparison.significant_at_0_05,
        "nemenyi_p_values": (comparison.nemenyi_p_values.tolist() if comparison.nemenyi_p_values is not None else None),
        "ranking": comparison.ranking,
        "naive_best_by_mean": naive_best,
        "statistically_justified_best": decision,
    }
