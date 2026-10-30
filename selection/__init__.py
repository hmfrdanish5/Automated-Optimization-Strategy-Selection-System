"""Model factories and strategy prediction helpers.

Training is imported lazily so dataset configuration can import the selection
package without creating a dataset/selection import cycle.
"""

from selection.classifiers import decision_tree, load_model, random_forest, save_model
from selection.rule_based import predict_rule


def __getattr__(name: str):
    if name == "train_classifiers":
        from selection.pipeline import train_classifiers

        return train_classifiers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "decision_tree",
    "load_model",
    "predict_rule",
    "random_forest",
    "save_model",
    "train_classifiers",
]
