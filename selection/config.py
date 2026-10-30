"""Fixed model and split settings.

Seeds are shared so a later run can reproduce this split and these models.
The dataset is small. Test metrics from this split are a pipeline check,
not a stable estimate of how the models would do on new programs.
"""

from __future__ import annotations

from pathlib import Path


DATASET_SEED = 42
SPLIT_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 42
TEST_SIZE = 0.20
STRATIFIED_SPLIT = "stratified"
STRICT_SPLIT = "strict"

TREE_PARAMETERS = {
    "random_state": MODEL_RANDOM_STATE,
}

FOREST_PARAMETERS = {
    "n_estimators": 100,
    "random_state": MODEL_RANDOM_STATE,
}

# Count features are divided by num_statements before these weights are applied.
# const_ratio is already a fraction, so it is not divided again.
# Constant propagation has no feature of its own. Its weight is intentionally
# smaller than constant folding's use of the same constant information.
RULE_WEIGHTS = {
    "constant_folding": {"num_constants": 1.0, "const_ratio": 0.5},
    "constant_propagation": {"const_ratio": 0.2},
    "algebraic_simplification": {"num_algebraic_ops": 1.0},
    "common_subexpression_elimination": {"num_repeated_expr": 1.0},
    "dead_code_elimination": {"num_dead_stmts": 1.0},
    "strength_reduction": {"num_loop_ops": 1.0},
}

MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "models"
TREE_PATH = MODEL_DIRECTORY / "decision_tree.pkl"
FOREST_PATH = MODEL_DIRECTORY / "random_forest.pkl"
HOLDOUT_TREE_PATH = MODEL_DIRECTORY / "holdout_decision_tree.pkl"
HOLDOUT_FOREST_PATH = MODEL_DIRECTORY / "holdout_random_forest.pkl"
PRODUCTION_TREE_PATH = MODEL_DIRECTORY / "production_decision_tree.pkl"
PRODUCTION_FOREST_PATH = MODEL_DIRECTORY / "production_random_forest.pkl"
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs"
