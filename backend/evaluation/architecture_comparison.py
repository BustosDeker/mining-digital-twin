"""Comparación cuantitativa de arquitecturas del clasificador de estrés.

Toma los resultados de CV (`CVTrainingResult`, Fase 6) de dos o más
arquitecturas entrenadas y evaluadas sobre EXACTAMENTE los mismos folds
(mismo esquema de CV, mismos sujetos retenidos en el mismo orden — LOSO
garantiza esto por construcción, ya que ambas arquitecturas se entrenan
sobre las mismas ventanas/sujetos), y aplica la selección estadística de
la Fase 7 (`statistics.tests.select_best`) sobre la métrica elegida
(F1-macro por defecto, por el desbalance esperado de clases).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.statistics.tests import select_best
from backend.training.train import CVTrainingResult
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ArchitectureComparisonReport:
    metric_name: str
    per_architecture_metrics: dict[str, list[float]]
    per_architecture_mean: dict[str, float]
    statistical_decision: dict[str, Any]


def compare_architectures(
    results: dict[str, CVTrainingResult], metric_name: str = "f1_macro"
) -> ArchitectureComparisonReport:
    fold_counts = {name: len(r.fold_results) for name, r in results.items()}
    if len(set(fold_counts.values())) != 1:
        logger.warning(
            "Las arquitecturas tienen distinto número de folds; la comparación "
            "pareada (Wilcoxon/Friedman) requiere folds alineados. Se truncará "
            "al mínimo común para poder comparar de forma válida.",
            extra=fold_counts,
        )

    min_folds = min(fold_counts.values())
    samples = {
        name: np.array(result.per_fold_metric(metric_name)[:min_folds])
        for name, result in results.items()
    }

    decision = select_best(samples)
    per_architecture_mean = {name: float(np.mean(values)) for name, values in samples.items()}

    logger.info(
        "Comparación de arquitecturas finalizada",
        extra={"metric": metric_name, "decision": decision["statistically_justified_best"]},
    )

    return ArchitectureComparisonReport(
        metric_name=metric_name,
        per_architecture_metrics={name: values.tolist() for name, values in samples.items()},
        per_architecture_mean=per_architecture_mean,
        statistical_decision=decision,
    )
