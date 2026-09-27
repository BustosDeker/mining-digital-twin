"""Entrenamiento con validación cruzada para ambas arquitecturas.

Cada fold entrena un modelo desde cero (sin fuga de información entre
folds), evalúa sobre el sujeto/fold retenido y registra métricas
homogéneas, listas para la comparación estadística de la Fase 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from backend.training.architectures import (
    ARCHITECTURE_BUILDERS,
    CNNLSTMHyperparams,
    FeaturesMLPHyperparams,
)
from backend.training.cross_validation import get_cv_splits
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CVFoldResult:
    fold_id: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    class_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class CVTrainingResult:
    architecture_name: str
    fold_results: list[CVFoldResult]
    hyperparams: dict[str, Any]
    class_names: list[str]

    def aggregate_metrics(self) -> dict[str, float]:
        keys = self.fold_results[0].metrics.keys()
        return {
            f"mean_{k}": float(np.mean([fr.metrics[k] for fr in self.fold_results])) for k in keys
        } | {
            f"std_{k}": float(np.std([fr.metrics[k] for fr in self.fold_results])) for k in keys
        }

    def per_fold_metric(self, metric_name: str) -> list[float]:
        return [fr.metrics[metric_name] for fr in self.fold_results]


def _compute_fold_metrics(
    y_true_int: np.ndarray, y_pred_int: np.ndarray, y_proba: np.ndarray, n_classes: int
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true_int, y_pred_int),
        "precision_macro": precision_score(y_true_int, y_pred_int, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_int, y_pred_int, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_int, y_pred_int, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true_int, y_pred_int),
    }
    try:
        if n_classes == 2:
            metrics["auc"] = roc_auc_score(y_true_int, y_proba[:, 1])
        else:
            metrics["auc"] = roc_auc_score(y_true_int, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        metrics["auc"] = float("nan")  # ocurre si un fold no contiene todas las clases
    return metrics


def train_cv(
    architecture_name: str,
    X: np.ndarray,
    y_names: np.ndarray,
    subject_ids: np.ndarray,
    hyperparams: CNNLSTMHyperparams | FeaturesMLPHyperparams | None = None,
    cv_strategy: str | None = None,
    epochs: int = 25,
    batch_size: int = 16,
    verbose: int = 0,
) -> CVTrainingResult:
    if architecture_name not in ARCHITECTURE_BUILDERS:
        raise ValueError(f"Arquitectura desconocida: {architecture_name}")

    class_names = sorted(set(y_names))
    label_to_int = {name: i for i, name in enumerate(class_names)}
    y_int = np.array([label_to_int[name] for name in y_names])
    n_classes = len(class_names)

    splits = get_cv_splits(y_names, subject_ids, strategy=cv_strategy)
    fold_results: list[CVFoldResult] = []

    for train_idx, test_idx, fold_id in splits:
        if len(np.unique(y_int[train_idx])) < 2:
            logger.warning("Fold omitido: menos de 2 clases en entrenamiento", extra={"fold_id": str(fold_id)})
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_int[train_idx], y_int[test_idx]

        if architecture_name == "features_mlp":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = ARCHITECTURE_BUILDERS[architecture_name](X_train.shape[1], n_classes, hyperparams)
        else:  # cnn_lstm
            model = ARCHITECTURE_BUILDERS[architecture_name](X_train.shape[1:], n_classes, hyperparams)

        class_weight = {
            c: len(y_train) / (n_classes * max(1, np.sum(y_train == c))) for c in range(n_classes)
        }

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=verbose,
        )

        y_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = _compute_fold_metrics(y_test, y_pred, y_proba, n_classes)
        fold_results.append(
            CVFoldResult(
                fold_id=str(fold_id),
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                class_names=class_names,
                metrics=metrics,
            )
        )
        tf.keras.backend.clear_session()  # evita acumulación de memoria entre folds

    logger.info(
        "Entrenamiento CV finalizado",
        extra={"architecture": architecture_name, "n_folds": len(fold_results)},
    )

    return CVTrainingResult(
        architecture_name=architecture_name,
        fold_results=fold_results,
        hyperparams=(hyperparams.to_dict() if hyperparams else {}),
        class_names=class_names,
    )


def confusion_matrix_aggregate(result: CVTrainingResult) -> np.ndarray:
    n_classes = len(result.class_names)
    aggregate = np.zeros((n_classes, n_classes), dtype=int)
    for fr in result.fold_results:
        aggregate += confusion_matrix(fr.y_true, fr.y_pred, labels=list(range(n_classes)))
    return aggregate


def fit_final_model(
    architecture_name: str,
    X: np.ndarray,
    y_names: np.ndarray,
    hyperparams: CNNLSTMHyperparams | FeaturesMLPHyperparams | None = None,
    epochs: int = 30,
    batch_size: int = 16,
    verbose: int = 0,
) -> tuple[tf.keras.Model, StandardScaler | None, list[str]]:
    """Ajusta el modelo FINAL de producción sobre TODOS los datos disponibles
    (tras validar la arquitectura/hiperparámetros mediante CV). Este modelo,
    no los modelos de cada fold, es el que se registra en el Model Registry.
    """
    class_names = sorted(set(y_names))
    label_to_int = {name: i for i, name in enumerate(class_names)}
    y_int = np.array([label_to_int[name] for name in y_names])
    n_classes = len(class_names)

    scaler: StandardScaler | None = None
    X_fit = X
    if architecture_name == "features_mlp":
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
        model = ARCHITECTURE_BUILDERS[architecture_name](X_fit.shape[1], n_classes, hyperparams)
    else:
        model = ARCHITECTURE_BUILDERS[architecture_name](X_fit.shape[1:], n_classes, hyperparams)

    class_weight = {
        c: len(y_int) / (n_classes * max(1, np.sum(y_int == c))) for c in range(n_classes)
    }
    model.fit(X_fit, y_int, epochs=epochs, batch_size=batch_size, class_weight=class_weight, verbose=verbose)

    return model, scaler, class_names
