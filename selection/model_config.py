"""Paths and hyperparameters for the trained selector models."""

from __future__ import annotations

from pathlib import Path

from selection.config import MODEL_RANDOM_STATE, MODEL_DIRECTORY, OUTPUT_DIRECTORY

EXTENSION_DATASET_SEED = 43
EXTENSION_PROGRAM_COUNT = 200
OPTIMIZATION_DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "optimization_dataset.csv"
EVALUATION_OUTPUT = OUTPUT_DIRECTORY / "evaluation"

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

PRODUCTION_TREE_EXTENDED = MODEL_DIRECTORY / "production_decision_tree_11_features.pkl"
PRODUCTION_FOREST_EXTENDED = MODEL_DIRECTORY / "production_random_forest_11_features.pkl"
PRODUCTION_LOGISTIC_EXTENDED = MODEL_DIRECTORY / "production_logistic_regression_11_features.pkl"
PRODUCTION_MLP_EXTENDED = MODEL_DIRECTORY / "production_mlp_11_features.pkl"

DATASET_VERSION = "optimization_dataset.csv"
