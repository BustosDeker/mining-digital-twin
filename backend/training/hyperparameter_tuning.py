"""Búsqueda de hiperparámetros con Optuna, con tracking completo de trials.

El objetivo a maximizar es el F1-macro medio de la validación cruzada
(preferido sobre accuracy por el desbalance de clases esperado, según la
regla del proyecto). Cada trial ejecuta una CV completa; el número de
trials y de épocas por trial son configurables para poder ajustar el costo
computacional según el entorno (CPU de desarrollo vs. GPU de producción).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from backend.training.architectures import CNNLSTMHyperparams, FeaturesMLPHyperparams
from backend.training.train import CVTrainingResult, train_cv
from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)  # evita miles de líneas de log por trial


@dataclass
class TuningResult:
    architecture_name: str
    best_hyperparams: dict[str, Any]
    best_value: float
    study: optuna.Study
    best_cv_result: CVTrainingResult

    def trials_dataframe(self) -> pd.DataFrame:
        return self.study.trials_dataframe()


def _sample_features_mlp_hp(trial: optuna.Trial) -> FeaturesMLPHyperparams:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layers = tuple(
        trial.suggest_categorical(f"units_layer_{i}", [16, 32, 64, 128]) for i in range(n_layers)
    )
    return FeaturesMLPHyperparams(
        hidden_layers=hidden_layers,
        dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 1e-2, log=True),
    )


def _sample_cnn_lstm_hp(trial: optuna.Trial) -> CNNLSTMHyperparams:
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 2)
    conv_filters = tuple(
        trial.suggest_categorical(f"filters_layer_{i}", [16, 32, 64]) for i in range(n_conv_layers)
    )
    return CNNLSTMHyperparams(
        conv_filters=conv_filters,
        conv_kernel_size=trial.suggest_categorical("conv_kernel_size", [3, 5, 7]),
        lstm_units=trial.suggest_categorical("lstm_units", [16, 32, 64]),
        dense_units=trial.suggest_categorical("dense_units", [16, 32, 64]),
        dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    )


_SAMPLERS = {"features_mlp": _sample_features_mlp_hp, "cnn_lstm": _sample_cnn_lstm_hp}


def tune_architecture(
    architecture_name: str,
    X,
    y_names,
    subject_ids,
    n_trials: int | None = None,
    epochs_per_trial: int = 15,
    batch_size: int = 16,
    cv_strategy: str | None = None,
    random_seed: int | None = None,
) -> TuningResult:
    if architecture_name not in _SAMPLERS:
        raise ValueError(f"Arquitectura sin sampler de hiperparámetros: {architecture_name}")

    settings = get_settings()
    n_trials = n_trials or settings.OPTUNA_N_TRIALS
    random_seed = random_seed if random_seed is not None else settings.RANDOM_SEED

    best_holder: dict[str, Any] = {"result": None}

    def objective(trial: optuna.Trial) -> float:
        hp = _SAMPLERS[architecture_name](trial)
        cv_result = train_cv(
            architecture_name,
            X,
            y_names,
            subject_ids,
            hyperparams=hp,
            cv_strategy=cv_strategy,
            epochs=epochs_per_trial,
            batch_size=batch_size,
            verbose=0,
        )
        mean_f1 = cv_result.aggregate_metrics()["mean_f1_macro"]

        if best_holder["result"] is None or mean_f1 > best_holder["result"].aggregate_metrics()["mean_f1_macro"]:
            best_holder["result"] = cv_result

        return mean_f1

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Tuning de hiperparámetros finalizado",
        extra={
            "architecture": architecture_name,
            "n_trials": n_trials,
            "best_value_f1_macro": study.best_value,
        },
    )

    return TuningResult(
        architecture_name=architecture_name,
        best_hyperparams=study.best_params,
        best_value=study.best_value,
        study=study,
        best_cv_result=best_holder["result"],
    )
