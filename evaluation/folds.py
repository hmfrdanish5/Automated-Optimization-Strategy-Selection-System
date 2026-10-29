"""Grouped cross-validation.

Folds are cut on the eight-feature signature. A signature cannot sit in
both the training fold and the test fold. This does not remove the limits
of a synthetic dataset.
"""

from __future__ import annotations

from dataset.diversity import feature_signature
from selection.config import SPLIT_RANDOM_STATE
from selection.splitting import LabeledExample


def grouped_folds(
    examples: list[LabeledExample],
    n_splits: int = 5,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, list[tuple[list[int], list[int]]]]:
    """Return the splitter name and train/test index pairs.

    StratifiedGroupKFold is preferred. If it cannot build the folds, the
    fallback is GroupKFold. That fallback does not promise class balance,
    and it is still a group split rather than a row-level split.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    groups = [feature_signature(example.features, feature_columns) for example in examples]
    labels = [example.label for example in examples]
    group_ids = _group_ids(groups)
    matrix = np.zeros((len(examples), 1))
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SPLIT_RANDOM_STATE,
        )
        folds = [
            (train.tolist(), test.tolist())
            for train, test in splitter.split(matrix, labels, group_ids)
        ]
        _require_no_group_overlap(groups, folds)
        return "StratifiedGroupKFold", folds
    except ValueError as error:
        splitter = GroupKFold(n_splits=n_splits)
        folds = [
            (train.tolist(), test.tolist())
            for train, test in splitter.split(matrix, labels, group_ids)
        ]
        _require_no_group_overlap(groups, folds)
        return (
            "GroupKFold because StratifiedGroupKFold could not build the folds "
            f"({error}). Class proportions are not guaranteed.",
            folds,
        )


def _group_ids(groups: list[tuple]) -> list[int]:
    indexes = {group: index for index, group in enumerate(dict.fromkeys(groups))}
    return [indexes[group] for group in groups]


def _require_no_group_overlap(groups: list[tuple], folds: list[tuple[list[int], list[int]]]) -> None:
    for train, test in folds:
        train_groups = {groups[index] for index in train}
        overlap = [index for index in test if groups[index] in train_groups]
        if overlap:
            raise RuntimeError(f"a feature group crossed a fold boundary ({len(overlap)} rows)")
