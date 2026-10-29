"""The six optimization strategies, each callable on its own.

Applying one name always starts from the program it is given. This module
does not chain passes and does not choose a winner.
"""

from __future__ import annotations

from collections.abc import Callable

from optimizers import algebraic_simplification
from optimizers import common_subexpression_elimination
from optimizers import constant_folding
from optimizers import constant_propagation
from optimizers import dead_code_elimination
from optimizers import strength_reduction
from tac.program import TACProgram


Optimize = Callable[[TACProgram], TACProgram]

STRATEGY_NAMES = (
    "constant_folding",
    "constant_propagation",
    "algebraic_simplification",
    "common_subexpression_elimination",
    "dead_code_elimination",
    "strength_reduction",
)

OPTIMIZERS: dict[str, Optimize] = {
    "constant_folding": constant_folding.optimize,
    "constant_propagation": constant_propagation.optimize,
    "algebraic_simplification": algebraic_simplification.optimize,
    "common_subexpression_elimination": common_subexpression_elimination.optimize,
    "dead_code_elimination": dead_code_elimination.optimize,
    "strength_reduction": strength_reduction.optimize,
}


def apply_strategy(program: TACProgram, strategy_name: str) -> TACProgram:
    """Run one named optimizer. The input program is not modified."""
    try:
        optimize = OPTIMIZERS[strategy_name]
    except KeyError as error:
        known = ", ".join(STRATEGY_NAMES)
        raise KeyError(f"unknown strategy {strategy_name!r}; expected one of: {known}") from error
    return optimize(program)
