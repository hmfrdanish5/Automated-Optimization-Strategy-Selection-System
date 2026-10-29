"""Ground-truth labels from six independent optimizer runs.

For each strategy the original program is copied, that one optimizer is
applied, and the gains are measured. Optimizers are never chained.
The label comes from `choose_best_strategy`, not from a template name.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.comparison import StrategyScore, choose_best_strategy
from evaluation.label_policy import NO_IMPROVEMENT
from evaluation.metrics import (
    code_size_gain,
    instruction_gain,
    operation_gain,
    temporary_gain,
    weighted_cost_gain,
)
from features.extractor import FEATURE_COLUMNS, extract_features
from optimizers.registry import STRATEGY_NAMES, apply_strategy
from tac.program import TACProgram


# Short prefixes for the diagnostic gain columns. These are not model inputs.
GAIN_PREFIX = {
    "constant_folding": "cf",
    "constant_propagation": "cp",
    "algebraic_simplification": "alg",
    "common_subexpression_elimination": "cse",
    "dead_code_elimination": "dce",
    "strength_reduction": "sr",
}

_GAIN_SUFFIXES = (
    ("instruction_gain", "instruction_gain"),
    ("weighted_cost_gain", "weighted_cost_gain"),
    ("operation_gain", "operation_gain"),
    ("temporary_gain", "temp_gain"),
    ("code_size_gain", "code_size_gain"),
)


def diagnostic_gain_columns() -> list[str]:
    """Column names such as `cf_instruction_gain`, in strategy order."""
    columns: list[str] = []
    for strategy in STRATEGY_NAMES:
        prefix = GAIN_PREFIX[strategy]
        for _attribute, suffix in _GAIN_SUFFIXES:
            columns.append(f"{prefix}_{suffix}")
    return columns


DATASET_COLUMNS = [
    "program_id",
    *FEATURE_COLUMNS,
    *diagnostic_gain_columns(),
    "best_strategy",
    "best_primary_gain",
    "best_secondary_gain",
]


@dataclass(frozen=True)
class LabelResult:
    """The six scores and the strategy selected from them."""

    best_strategy: str
    scores: tuple[StrategyScore, ...]

    def winning_score(self) -> StrategyScore | None:
        """The score of the chosen optimizer, or None for `no_improvement`."""
        if self.best_strategy == NO_IMPROVEMENT:
            return None
        for score in self.scores:
            if score.strategy == self.best_strategy:
                return score
        return None


def score_strategy(strategy: str, original: TACProgram, optimized: TACProgram) -> StrategyScore:
    """Measure one optimized program against the original."""
    return StrategyScore(
        strategy=strategy,
        instruction_gain=instruction_gain(original, optimized),
        weighted_cost_gain=weighted_cost_gain(original, optimized),
        operation_gain=operation_gain(original, optimized),
        temporary_gain=temporary_gain(original, optimized),
        code_size_gain=code_size_gain(original, optimized),
    )


def evaluate_strategies(program: TACProgram) -> tuple[StrategyScore, ...]:
    """Run each optimizer on its own copy of `program`.

    The program text is checked after every call. A pass that edited the
    input would make the later gains meaningless.
    """
    original_text = program.to_tac()
    scores: list[StrategyScore] = []
    for strategy in STRATEGY_NAMES:
        optimized = apply_strategy(program.copy(), strategy)
        if program.to_tac() != original_text:
            raise RuntimeError(f"{strategy} changed the program it was given")
        scores.append(score_strategy(strategy, program, optimized))
    return tuple(scores)


def label_program(program: TACProgram) -> LabelResult:
    """Choose `best_strategy` by measuring the six optimizers separately."""
    scores = evaluate_strategies(program)
    return LabelResult(
        best_strategy=choose_best_strategy(list(scores)),
        scores=scores,
    )


def training_features(row: dict[str, object]) -> dict[str, object]:
    """Return only the eight model inputs.

    Gain columns and `best_strategy` are ground truth. Training on them
    would leak the answer into the features.
    """
    return {column: row[column] for column in FEATURE_COLUMNS}


def label_row(program_id: str, program: TACProgram) -> dict[str, object]:
    """Build one dataset row: eight features, diagnostic gains, and the label."""
    features = extract_features(program)
    labeled = label_program(program)
    row: dict[str, object] = {"program_id": program_id}
    row.update(features)
    for score in labeled.scores:
        prefix = GAIN_PREFIX[score.strategy]
        for attribute, suffix in _GAIN_SUFFIXES:
            row[f"{prefix}_{suffix}"] = getattr(score, attribute)
    winner = labeled.winning_score()
    row["best_strategy"] = labeled.best_strategy
    row["best_primary_gain"] = 0 if winner is None else winner.instruction_gain
    row["best_secondary_gain"] = 0 if winner is None else winner.weighted_cost_gain
    return row
