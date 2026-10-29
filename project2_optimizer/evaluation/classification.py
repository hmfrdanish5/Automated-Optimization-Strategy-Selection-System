"""Classification metrics for strategy predictions.

These numbers answer whether the predicted name matches best_strategy.
They do not measure how many instructions the chosen optimizer removed.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from evaluation.label_policy import NO_IMPROVEMENT
from optimizers.registry import STRATEGY_NAMES


CLASS_ORDER = (*STRATEGY_NAMES, NO_IMPROVEMENT)

SHORT_CLASS_NAMES = {
    "constant_folding": "CF",
    "constant_propagation": "CP",
    "algebraic_simplification": "ALG",
    "common_subexpression_elimination": "CSE",
    "dead_code_elimination": "DCE",
    "strength_reduction": "SR",
    "no_improvement": "NONE",
}


def accuracy(actual: list[str], predicted: list[str]) -> float:
    """Fraction of predictions that match the measured label."""
    return float(accuracy_score(actual, predicted))


def macro_f1(actual: list[str], predicted: list[str]) -> float:
    """Unweighted mean of per-class F1 scores. A rare class counts equally."""
    return float(
        f1_score(
            actual,
            predicted,
            labels=list(CLASS_ORDER),
            average="macro",
            zero_division=0,
        )
    )


def metric_bundle(actual: list[str], predicted: list[str]) -> dict[str, object]:
    """Accuracy, macro and weighted averages, and one row per class."""
    labels = list(CLASS_ORDER)
    return {
        "accuracy": float(accuracy_score(actual, predicted)) if actual else 0.0,
        "macro_precision": float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
        "per_class": classification_report(
            actual,
            predicted,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
    }


def report_text(actual: list[str], predicted: list[str]) -> str:
    """Precision, recall, F1, and support for every class, plus averages."""
    return classification_report(
        actual,
        predicted,
        labels=list(CLASS_ORDER),
        zero_division=0,
        digits=3,
    )


def confusion(actual: list[str], predicted: list[str]):
    """Rows are the measured label and columns are the prediction, in CLASS_ORDER."""
    return confusion_matrix(actual, predicted, labels=list(CLASS_ORDER))
