"""Decision tree and random forest wrappers.

`fit` trains. `predict` returns a strategy name. `predict_proba` returns a
probability for every class the model saw. The probability of the predicted
class is a confidence value, not a promise that the optimizer will help.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from features.extractor import FEATURE_COLUMNS
from selection.config import FOREST_PARAMETERS, TREE_PARAMETERS
from selection.model_config import LOGISTIC_PARAMETERS, MLP_PARAMETERS


@dataclass(frozen=True)
class StrategyPrediction:
    """One classification of a feature vector."""

    strategy: str
    confidence: float
    probabilities: dict[str, float]


class StrategyClassifier:
    """A fitted scikit-learn classifier restricted to one feature schema.

    The default schema is the original eight FEATURE_COLUMNS. Extended-schema models
    pass FEATURE_COLUMNS_EXTENDED. A frame with a different column list is rejected.
    """

    def __init__(self, name: str, estimator, feature_columns: list[str] | None = None):
        self.name = name
        self.estimator = estimator
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "StrategyClassifier":
        matrix = _feature_matrix(features, self.feature_columns)
        self.estimator.fit(matrix, labels)
        return self

    def predict(self, features: pd.DataFrame) -> list[str]:
        matrix = _feature_matrix(features, self.feature_columns)
        return [str(label) for label in self.estimator.predict(matrix)]

    def predict_proba(self, features: pd.DataFrame) -> list[dict[str, float]]:
        """Probability of each class, in the estimator's class order."""
        matrix = _feature_matrix(features, self.feature_columns)
        matrix = self.estimator.predict_proba(matrix)
        class_names = [str(name) for name in self.estimator.classes_]
        return [
            {class_names[index]: float(probability) for index, probability in enumerate(row)}
            for row in matrix
        ]

    def predict_one(self, features: dict[str, float]) -> StrategyPrediction:
        """Classify one feature dictionary."""
        frame = pd.DataFrame([{column: features[column] for column in self.feature_columns}])
        strategy = self.predict(frame)[0]
        probabilities = self.predict_proba(frame)[0]
        return StrategyPrediction(
            strategy=strategy,
            confidence=probabilities[strategy],
            probabilities=probabilities,
        )

    def export_tree_text(self) -> str:
        """Render a decision tree. Random forest has no single tree to print."""
        tree = _inner_estimator(self.estimator)
        if not isinstance(tree, DecisionTreeClassifier):
            raise TypeError(f"{self.name} is not a decision tree")
        return export_text(tree, feature_names=list(self.feature_columns))

    def feature_importances(self) -> dict[str, float]:
        """Importance of each feature column when the estimator exposes it."""
        estimator = _inner_estimator(self.estimator)
        values = getattr(estimator, "feature_importances_", None)
        if values is None:
            raise TypeError(f"{self.name} does not expose feature_importances_")
        return {name: float(value) for name, value in zip(self.feature_columns, values, strict=True)}

    def standardized_coefficients(self) -> list[dict[str, object]]:
        """Logistic coefficients after StandardScaler, one row per class and feature.

        These numbers change the model's class scores. They are not causal effects.
        """
        estimator = _inner_estimator(self.estimator)
        coef = getattr(estimator, "coef_", None)
        if coef is None:
            raise TypeError(f"{self.name} does not expose coefficients")
        classes = [str(name) for name in estimator.classes_]
        rows = []
        for class_index, class_name in enumerate(classes):
            for feature_index, feature_name in enumerate(self.feature_columns):
                rows.append(
                    {
                        "class": class_name,
                        "feature": feature_name,
                        "coefficient": float(coef[class_index, feature_index]),
                    }
                )
        return rows


def decision_tree(feature_columns: list[str] | None = None) -> StrategyClassifier:
    """An interpretable tree with the project's fixed random seed."""
    return StrategyClassifier(
        "decision_tree",
        DecisionTreeClassifier(**TREE_PARAMETERS),
        feature_columns,
    )


def random_forest(feature_columns: list[str] | None = None) -> StrategyClassifier:
    """The Random Forest: 100 trees, fixed seed, library defaults otherwise."""
    return StrategyClassifier(
        "random_forest",
        RandomForestClassifier(**FOREST_PARAMETERS),
        feature_columns,
    )


def logistic_regression(feature_columns: list[str] | None = None) -> StrategyClassifier:
    """Multinomial logistic regression. The scaler is fit only on training rows."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**LOGISTIC_PARAMETERS)),
        ]
    )
    return StrategyClassifier("logistic_regression", pipeline, feature_columns)


def mlp_classifier(feature_columns: list[str] | None = None) -> StrategyClassifier:
    """A small MLP. The scaler is fit only on training rows."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(**MLP_PARAMETERS)),
        ]
    )
    return StrategyClassifier("mlp", pipeline, feature_columns)


def save_model(model: StrategyClassifier, path: Path, metadata: dict | None = None) -> None:
    """Write the fitted estimator, its feature schema, and the scikit-learn version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": model.name,
        "estimator": model.estimator,
        "feature_columns": list(model.feature_columns),
        "sklearn_version": sklearn.__version__,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    joblib.dump(payload, path)


# One warning per file that was saved without a scikit-learn version.
_MISSING_VERSION_WARNED: set[str] = set()


def load_model(path: Path) -> StrategyClassifier:
    """Reload a model saved by `save_model`.

    A missing scikit-learn version is warned about once per file. A recorded
    version that differs from this environment is warned about on every load.
    Neither warning is filtered here, and neither warning refuses to load.
    """
    payload = joblib.load(path)
    saved = payload.get("sklearn_version") if isinstance(payload, dict) else None
    current = sklearn.__version__
    key = str(path.resolve())
    if saved is None:
        if key not in _MISSING_VERSION_WARNED:
            _MISSING_VERSION_WARNED.add(key)
            warnings.warn(
                f"{path.name} has no recorded scikit-learn version. "
                f"This environment is scikit-learn {current}.",
                UserWarning,
                stacklevel=2,
            )
    elif saved != current:
        warnings.warn(
            f"{path.name} was saved with scikit-learn {saved}, "
            f"but this environment is scikit-learn {current}.",
            UserWarning,
            stacklevel=2,
        )
    feature_columns = payload.get("feature_columns") if isinstance(payload, dict) else None
    if feature_columns is None:
        feature_columns = list(FEATURE_COLUMNS)
    return StrategyClassifier(payload["name"], payload["estimator"], feature_columns)


def _inner_estimator(estimator):
    if isinstance(estimator, Pipeline):
        return estimator.named_steps["classifier"]
    return estimator


def _feature_matrix(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    columns = list(features.columns)
    missing = [column for column in feature_columns if column not in features.columns]
    extra = [column for column in columns if column not in feature_columns]
    if missing or extra:
        raise ValueError(
            f"feature schema mismatch for {len(feature_columns)} columns: "
            f"missing {missing}, extra {extra}"
        )
    if columns != feature_columns:
        raise ValueError(
            f"feature columns must match schema order {feature_columns}, not {columns}"
        )
    return features.loc[:, feature_columns]
