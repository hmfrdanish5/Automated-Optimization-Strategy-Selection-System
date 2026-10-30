"""Settings for the search controls, the fair rule baseline and the search experiments."""

from __future__ import annotations

from selection.config import OUTPUT_DIRECTORY

SEARCH_OUTPUT = OUTPUT_DIRECTORY / "search"

# Seed for the random-order baseline used in the search experiments.
RANDOM_ORDER_SEED = 7

# Confidence thresholds tried for stopping early on a `no_improvement` prediction.
EARLY_STOP_THRESHOLDS = (0.5, 0.7, 0.9)
# Per-round candidate budgets tried in the limited-tries experiment.
ATTEMPT_BUDGETS = (1, 2)
# Threshold behind the reported early-stop results, the search-effort plot, the
# unseen benchmark and the demo's "stop early" switch. At 0.7 the tree models and
# the eleven-feature rule keep all of the gain on the held-out programs.
DEFAULT_EARLY_STOP_THRESHOLD = 0.7

# Rule-based scores from the eleven-feature schema (the fair rule baseline).
#
# The label policy compares instruction-count reduction first, and only dead-code
# elimination removes instructions, so any removable dead statement decides the
# label. Otherwise the passes compete on weighted operation cost: folding a
# constant, reusing an expression, removing an identity and turning x * 2 into
# x + x each save about one operation per site. Constant propagation rewrites
# operands without removing operations, so it only wins when nothing else can.
RULE_WEIGHTS_V2 = {
    "constant_folding": {"num_constants": 1.0},
    "constant_propagation": {"num_constant_propagation_uses": 0.01},
    "algebraic_simplification": {"num_algebraic_ops": 1.0},
    "common_subexpression_elimination": {"num_safe_repeated_expr": 1.0},
    "dead_code_elimination": {"num_dead_stmts": 100.0},
    "strength_reduction": {"num_mul_by_two_in_loops": 1.0},
}
