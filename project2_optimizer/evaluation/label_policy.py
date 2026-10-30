"""Ground-truth label policy.

Instruction-count reduction is the first and most important comparison.
The later criteria are used only when every earlier criterion ties.
STRATEGY_PRIORITY is used only when every gain in LABEL_CRITERIA is equal.

To change the policy later, edit LABEL_CRITERIA. Dataset generation reads
this tuple; it does not contain its own ordering.

A result that is not strictly better than doing nothing is labeled
`no_improvement`. That label is not one of the six optimizers.
"""

from __future__ import annotations


# Earlier entries outrank later entries.
LABEL_CRITERIA = (
    "instruction_gain",
    "weighted_cost_gain",
    "operation_gain",
    "temporary_gain",
    "code_size_gain",
)

# Earlier names win when every gain in LABEL_CRITERIA is equal.
STRATEGY_PRIORITY = (
    "constant_folding",
    "constant_propagation",
    "algebraic_simplification",
    "common_subexpression_elimination",
    "dead_code_elimination",
    "strength_reduction",
)

NO_IMPROVEMENT = "no_improvement"
