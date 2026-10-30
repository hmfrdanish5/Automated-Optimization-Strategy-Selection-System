"""Review-2 paths and hyperparameters (separate from Review-1)."""

from __future__ import annotations

from pathlib import Path

from selection.config import DATASET_SEED, MODEL_RANDOM_STATE, MODEL_DIRECTORY, OUTPUT_DIRECTORY

V2_DATASET_SEED = 43
V2_EXTENSION_COUNT = 200
V2_DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "dataset_v2.csv"
REVIEW2_OUTPUT = OUTPUT_DIRECTORY / "review2"

LOGISTIC_PARAMETERS = {
    "solver": "lbfgs",
    "C": 1.0,
    "max_iter": 3000,
    "random_state": MODEL_RANDOM_STATE,
    "class_weight": None,
}

MLP_PARAMETERS = {
    "hidden_layer_sizes": (32, 16),
    "max_iter": 2000,
    "alpha": 0.001,
    "random_state": MODEL_RANDOM_STATE,
}

REVIEW2_HOLDOUT_TREE_V1 = MODEL_DIRECTORY / "review2_holdout_decision_tree_v1.pkl"
REVIEW2_HOLDOUT_FOREST_V1 = MODEL_DIRECTORY / "review2_holdout_random_forest_v1.pkl"
REVIEW2_HOLDOUT_LOGISTIC_V1 = MODEL_DIRECTORY / "review2_holdout_logistic_regression_v1.pkl"
REVIEW2_HOLDOUT_MLP_V1 = MODEL_DIRECTORY / "review2_holdout_mlp_v1.pkl"

REVIEW2_HOLDOUT_TREE_V2 = MODEL_DIRECTORY / "review2_holdout_decision_tree_v2.pkl"
REVIEW2_HOLDOUT_FOREST_V2 = MODEL_DIRECTORY / "review2_holdout_random_forest_v2.pkl"
REVIEW2_HOLDOUT_LOGISTIC_V2 = MODEL_DIRECTORY / "review2_holdout_logistic_regression_v2.pkl"
REVIEW2_HOLDOUT_MLP_V2 = MODEL_DIRECTORY / "review2_holdout_mlp_v2.pkl"

REVIEW2_PRODUCTION_TREE_V1 = MODEL_DIRECTORY / "review2_production_decision_tree_v1.pkl"
REVIEW2_PRODUCTION_FOREST_V1 = MODEL_DIRECTORY / "review2_production_random_forest_v1.pkl"
REVIEW2_PRODUCTION_LOGISTIC_V1 = MODEL_DIRECTORY / "review2_production_logistic_regression_v1.pkl"
REVIEW2_PRODUCTION_MLP_V1 = MODEL_DIRECTORY / "review2_production_mlp_v1.pkl"

REVIEW2_PRODUCTION_TREE_V2 = MODEL_DIRECTORY / "review2_production_decision_tree_v2.pkl"
REVIEW2_PRODUCTION_FOREST_V2 = MODEL_DIRECTORY / "review2_production_random_forest_v2.pkl"
REVIEW2_PRODUCTION_LOGISTIC_V2 = MODEL_DIRECTORY / "review2_production_logistic_regression_v2.pkl"
REVIEW2_PRODUCTION_MLP_V2 = MODEL_DIRECTORY / "review2_production_mlp_v2.pkl"

DATASET_VERSION_V2 = "dataset_v2.csv"
DATASET_SEED_V2_BUILD = DATASET_SEED  # same 200 programs as final use GENERATION_SEED
