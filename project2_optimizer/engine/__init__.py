"""Iterative optimization loop.

The selector ranks strategies. The optimizers change copies of the TAC.
Measurement decides whether a change is kept.
"""

from engine.loop import MAX_ITERATIONS, OptimizationRun, optimize_program
from engine.selectors import ModelSelector, RankingSelector, RuleSelector, saved_selectors

__all__ = [
    "MAX_ITERATIONS",
    "ModelSelector",
    "OptimizationRun",
    "RankingSelector",
    "RuleSelector",
    "optimize_program",
    "saved_selectors",
]
