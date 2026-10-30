"""Row-level and feature-grouped train/test splits.

The model sees only the selected feature columns. Two rows with identical inputs
are the same input, so a strict split keeps that tuple entirely in train
or entirely in test.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from dataset.diversity import feature_signature
from selection.config import STRATIFIED_SPLIT, SPLIT_RANDOM_STATE, STRICT_SPLIT, TEST_SIZE
from tac.program import TACProgram


@dataclass(frozen=True)
class LabeledExample:
    """One TAC program, its selected feature vector, and its measured label."""

    program_id: str
    program: TACProgram
    features: dict[str, float]
    label: str


@dataclass(frozen=True)
class SplitInfo:
    """How the examples were divided, including a failed stratification."""

    stratified: bool
    explanation: str
    mode: str = STRATIFIED_SPLIT
    overlap_rows: int = 0
    train_groups: int | None = None
    test_groups: int | None = None


def split_examples(
    examples: list[LabeledExample],
    mode: str = STRATIFIED_SPLIT,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[LabeledExample], list[LabeledExample], SplitInfo]:
    """Hold out about TEST_SIZE of the examples.

    `stratified` splits rows and may place the same feature vector on both sides.
    `strict` splits groups of identical feature vectors. If that grouped
    split cannot be stratified, the fallback is still a group split. It is
    not a row-level shuffle.
    """
    if mode == STRATIFIED_SPLIT:
        return _split_rows(examples)
    if mode == STRICT_SPLIT:
        return _split_groups(examples, feature_columns)
    raise ValueError(f"unknown split mode {mode}")


def _split_rows(
    examples: list[LabeledExample],
) -> tuple[list[LabeledExample], list[LabeledExample], SplitInfo]:
    labels = [example.label for example in examples]
    try:
        train, test = train_test_split(
            examples,
            test_size=TEST_SIZE,
            random_state=SPLIT_RANDOM_STATE,
            stratify=labels,
        )
        info = SplitInfo(True, "Stratified row-level split succeeded", STRATIFIED_SPLIT)
    except ValueError as error:
        train, test = train_test_split(
            examples,
            test_size=TEST_SIZE,
            random_state=SPLIT_RANDOM_STATE,
            stratify=None,
        )
        info = SplitInfo(
            False,
            "Stratified row-level split could not be stratified "
            f"({error}). The fallback is an ordinary row shuffle with "
            f"random_state={SPLIT_RANDOM_STATE}.",
            STRATIFIED_SPLIT,
        )
    train_list = list(train)
    test_list = list(test)
    overlap = _overlap_rows(train_list, test_list)
    return train_list, test_list, SplitInfo(
        info.stratified,
        info.explanation,
        STRATIFIED_SPLIT,
        overlap,
    )


def _split_groups(
    examples: list[LabeledExample],
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[LabeledExample], list[LabeledExample], SplitInfo]:
    columns = list(feature_columns) if feature_columns else None
    grouped: dict[tuple, list[LabeledExample]] = defaultdict(list)
    order: list[tuple] = []
    for example in examples:
        signature = feature_signature(example.features, columns)
        if signature not in grouped:
            order.append(signature)
        grouped[signature].append(example)
    groups = [grouped[signature] for signature in order]
    group_labels = [group[0].label for group in groups]
    indices = list(range(len(groups)))
    try:
        train_index, test_index = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=SPLIT_RANDOM_STATE,
            stratify=group_labels,
        )
        stratified = True
        explanation = (
            "MODE B - strict feature-grouped split, stratified by the first "
            "label in each feature group. "
            "The strict feature-grouped split prevents identical feature "
            "signatures from crossing the train/test boundary."
        )
    except ValueError as error:
        train_index, test_index = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=SPLIT_RANDOM_STATE,
            stratify=None,
        )
        stratified = False
        explanation = (
            "MODE B - strict feature-grouped split could not be stratified "
            f"({error}). Groups were still held out together with "
            f"random_state={SPLIT_RANDOM_STATE}. This is not a row-level split. "
            "The strict feature-grouped split prevents identical feature "
            "signatures from crossing the train/test boundary."
        )
    train = [example for index in train_index for example in groups[int(index)]]
    test = [example for index in test_index for example in groups[int(index)]]
    overlap = _overlap_rows(train, test, columns)
    if overlap != 0:
        raise RuntimeError(f"grouped split left {overlap} overlapping test rows")
    return train, test, SplitInfo(
        stratified,
        explanation,
        STRICT_SPLIT,
        overlap,
        len(list(train_index)),
        len(list(test_index)),
    )


def _overlap_rows(
    train: list[LabeledExample],
    test: list[LabeledExample],
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> int:
    columns = list(feature_columns) if feature_columns else None
    train_signatures = {feature_signature(example.features, columns) for example in train}
    return sum(
        1 for example in test if feature_signature(example.features, columns) in train_signatures
    )
