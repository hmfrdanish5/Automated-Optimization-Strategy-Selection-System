"""Measurements, ground-truth labels, and classification metrics.

Classification metrics compare a predicted strategy name with best_strategy.
They are separate from instruction-count gain and weighted-cost gain.
"""

from evaluation.comparison import StrategyScore, choose_best_strategy, compare_strategy_results
from evaluation.label_policy import LABEL_CRITERIA, NO_IMPROVEMENT, STRATEGY_PRIORITY
from evaluation.metrics import (
    arithmetic_operation_count,
    code_size,
    code_size_gain,
    instruction_count,
    instruction_gain,
    operation_gain,
    reduction_percentage,
    temporary_gain,
    temporary_variable_count,
    weighted_cost_gain,
    weighted_operation_cost,
)

__all__ = [
    "LABEL_CRITERIA",
    "NO_IMPROVEMENT",
    "STRATEGY_PRIORITY",
    "StrategyScore",
    "arithmetic_operation_count",
    "choose_best_strategy",
    "code_size",
    "code_size_gain",
    "compare_strategy_results",
    "instruction_count",
    "instruction_gain",
    "operation_gain",
    "reduction_percentage",
    "temporary_gain",
    "temporary_variable_count",
    "weighted_cost_gain",
    "weighted_operation_cost",
]
