"""Validación cruzada con split estratificado por sujeto.

Regla del proyecto: split estratificado por sujeto y validación cruzada
K-Fold (o LOSO — Leave-One-Subject-Out, preferida por ser biométrico). Un
split aleatorio por muestra (en vez de por sujeto) sobreestimaría el
rendimiento real, ya que ventanas del mismo sujeto en train y test filtran
información fisiológica individual (fuga de datos).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from backend.utils.config import get_settings


def loso_splits(
    labels: np.ndarray, subject_ids: np.ndarray
) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Leave-One-Subject-Out: en cada fold, un sujeto completo queda fuera
    como test. Devuelve (train_idx, test_idx, held_out_subject_id).
    """
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(np.zeros(len(labels)), labels, groups=subject_ids):
        held_out_subject = subject_ids[test_idx[0]]
        yield train_idx, test_idx, held_out_subject


def stratified_group_kfold_splits(
    labels: np.ndarray, subject_ids: np.ndarray, n_splits: int | None = None, random_seed: int | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """K-Fold estratificado por clase, agrupado por sujeto (ningún sujeto
    aparece simultáneamente en train y test de un mismo fold).
    """
    settings = get_settings()
    # Only use settings default if n_splits is explicitly None, not if it's 0
    if n_splits is None:
        n_splits = settings.CV_N_FOLDS
    random_seed = random_seed if random_seed is not None else settings.RANDOM_SEED

    # Limit n_splits to number of unique subjects
    n_unique_subjects = len(set(subject_ids))
    if n_splits > n_unique_subjects:
        n_splits = n_unique_subjects

    label_to_int = {name: i for i, name in enumerate(sorted(set(labels)))}
    y_int = np.array([label_to_int[label] for label in labels])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(y_int, y_int, groups=subject_ids)):
        yield train_idx, test_idx, fold_idx


def get_cv_splits(
    labels: np.ndarray, subject_ids: np.ndarray, strategy: str | None = None, n_folds: int | None = None
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Punto de entrada único: usa `settings.CV_STRATEGY` salvo que se
    indique lo contrario explícitamente.
    """
    settings = get_settings()
    strategy = strategy or settings.CV_STRATEGY

    if strategy == "loso":
        return list(loso_splits(labels, subject_ids))
    if strategy == "kfold":
        return [
            (train_idx, test_idx, f"fold_{fold_idx}")
            for train_idx, test_idx, fold_idx in stratified_group_kfold_splits(labels, subject_ids, n_splits=n_folds)
        ]
    raise ValueError(f"Estrategia de CV desconocida: {strategy}")
