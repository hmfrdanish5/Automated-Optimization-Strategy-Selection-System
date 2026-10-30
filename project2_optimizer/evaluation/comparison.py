"""Choose the best independent optimizer result.

`compare_strategy_results` is the only ordering used for labels. It follows
LABEL_CRITERIA, then STRATEGY_PRIORITY.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.label_policy import LABEL_CRITERIA, NO_IMPROVEMENT, STRATEGY_PRIORITY


@dataclass(frozen=True)
class StrategyScore:
    """Gains from applying one optimizer to the original program."""

    strategy: str
    instruction_gain: int
    weighted_cost_gain: int
    operation_gain: int
    temporary_gain: int
    code_size_gain: int

    def metric_tuple(self) -> tuple[int, ...]:
        """Gains in label-policy order, without the priority tie-break."""
        return tuple(getattr(self, criterion) for criterion in LABEL_CRITERIA)


def compare_strategy_results(left: StrategyScore, right: StrategyScore) -> int:
    """Return 1 if `left` is the better label, -1 if `right` is, and 0 if equal.

    Higher instruction-count gain wins. Weighted operation-cost gain is next,
    then the number of arithmetic operations, then temporary variables, then
    rendered code size. Weighted cost never overrides a larger instruction
    reduction. If every gain matches, the earlier name in STRATEGY_PRIORITY
    wins.
    """
    for criterion in LABEL_CRITERIA:
        left_value = getattr(left, criterion)
        right_value = getattr(right, criterion)
        if left_value != right_value:
            return 1 if left_value > right_value else -1
    left_priority = STRATEGY_PRIORITY.index(left.strategy)
    right_priority = STRATEGY_PRIORITY.index(right.strategy)
    if left_priority != right_priority:
        return 1 if left_priority < right_priority else -1
    return 0


def improves_on_original(score: StrategyScore) -> bool:
    """Return True when `score` is lexicographically better than all zeros."""
    return score.metric_tuple() > tuple(0 for _ in LABEL_CRITERIA)


def choose_best_strategy(scores: list[StrategyScore]) -> str:
    """Return the winning strategy name, or `no_improvement` when none helps.

    A negative gain does not count as an opportunity. If the best measured
    result is no better than leaving the program unchanged, the label is
    `no_improvement` rather than an arbitrary optimizer.
    """
    if not scores:
        return NO_IMPROVEMENT
    winner = scores[0]
    for candidate in scores[1:]:
        if compare_strategy_results(candidate, winner) > 0:
            winner = candidate
    if not improves_on_original(winner):
        return NO_IMPROVEMENT
    return winner.strategy
