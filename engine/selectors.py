"""Strategy rankings for the iterative engine.

A selector orders the six optimizers. It does not change TAC. A probability
or a rule score is not an instruction-count gain.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.label_policy import NO_IMPROVEMENT, STRATEGY_PRIORITY
from features.extended_extractor import FEATURE_COLUMNS_V2, extract_features_v2
from features.extractor import FEATURE_COLUMNS, extract_features
from optimizers.registry import STRATEGY_NAMES
from selection.classifiers import StrategyClassifier, load_model
from selection.config import FOREST_PATH, TREE_PATH
from selection.rule_based import predict_rule, rule_scores


@dataclass(frozen=True)
class SelectorAdvice:
    """One ranking of the current program.

    `selected_strategy` is the selector's top class, which may be
    `no_improvement`. `try_order` is only the six optimizers. The engine
    still runs those optimizers when the top class is `no_improvement`.
    """

    selector_type: str
    selected_strategy: str
    confidence: float
    probabilities: dict[str, float]
    ranking: tuple[str, ...]
    try_order: tuple[str, ...]


class RuleSelector:
    """Rank the six optimizers by the hand-written feature scores."""

    selector_type = "rule_based"

    def advise(self, features: dict[str, float]) -> SelectorAdvice:
        scores = rule_scores(features)
        nothing = 0.0 if max(scores.values()) > 0 else 1.0
        probabilities = {**scores, NO_IMPROVEMENT: nothing}
        return _advice(self.selector_type, probabilities, predict_rule(features))


class ModelSelector:
    """Rank optimizers by a fitted model's class probabilities.

    `predict` here is inference. This class does not call `fit`.
    """

    def __init__(self, selector_type: str, model: StrategyClassifier):
        self.selector_type = selector_type
        self.model = model

    def prepare_features(self, program) -> dict[str, float]:
        if list(self.model.feature_columns) == list(FEATURE_COLUMNS_V2):
            features = extract_features_v2(program)
        else:
            features = extract_features(program)
        require_feature_keys(features, self.model.feature_columns)
        return features

    def advise(self, features: dict[str, float]) -> SelectorAdvice:
        prediction = self.model.predict_one(features)
        probabilities = {
            name: float(prediction.probabilities.get(name, 0.0))
            for name in (*STRATEGY_NAMES, NO_IMPROVEMENT)
        }
        return _advice(self.selector_type, probabilities, prediction.strategy)


class RankingSelector:
    """Use a caller-supplied score map.

    The fallback demonstration uses this so the top-ranked optimizer can be
    one that does not help, while a later one does.
    """

    def __init__(self, selector_type: str, probabilities: dict[str, float]):
        self.selector_type = selector_type
        self._probabilities = {
            name: float(probabilities.get(name, 0.0))
            for name in (*STRATEGY_NAMES, NO_IMPROVEMENT)
        }

    def advise(self, features: dict[str, float]) -> SelectorAdvice:
        del features
        selected = _rank(self._probabilities)[0]
        return _advice(self.selector_type, self._probabilities, selected)


class FixedOrderSelector:
    """Always try the six optimizers in STRATEGY_PRIORITY order.

    The first pass that measures as an improvement is kept, then the program
    is re-analyzed and the same order is tried again. This is the baseline
    that selection is meant to avoid: a fixed sequence, not a learned ranking.
    """

    selector_type = "fixed_order"

    def advise(self, features: dict[str, float]) -> SelectorAdvice:
        del features
        probabilities = {
            name: float(len(STRATEGY_PRIORITY) - index)
            for index, name in enumerate(STRATEGY_PRIORITY)
        }
        probabilities[NO_IMPROVEMENT] = 0.0
        return _advice(self.selector_type, probabilities, STRATEGY_PRIORITY[0])


def saved_selectors() -> dict[str, RuleSelector | ModelSelector]:
    """Load the expanded-dataset models. Training stays outside this function."""
    return {
        "rule_based": RuleSelector(),
        "decision_tree": ModelSelector("decision_tree", load_model(TREE_PATH)),
        "random_forest": ModelSelector("random_forest", load_model(FOREST_PATH)),
    }


def runtime_selectors() -> dict[str, RuleSelector | ModelSelector]:
    """Load Review-2 production models when present, else Review-1 production."""
    from selection.config import PRODUCTION_FOREST_PATH, PRODUCTION_TREE_PATH
    from selection.review2_config import (
        REVIEW2_PRODUCTION_FOREST_V2,
        REVIEW2_PRODUCTION_LOGISTIC_V2,
        REVIEW2_PRODUCTION_MLP_V2,
        REVIEW2_PRODUCTION_TREE_V2,
    )

    review2_ready = all(
        path.exists()
        for path in (
            REVIEW2_PRODUCTION_TREE_V2,
            REVIEW2_PRODUCTION_FOREST_V2,
            REVIEW2_PRODUCTION_LOGISTIC_V2,
            REVIEW2_PRODUCTION_MLP_V2,
        )
    )
    if review2_ready:
        return {
            "rule_based": RuleSelector(),
            "decision_tree": ModelSelector("decision_tree", load_model(REVIEW2_PRODUCTION_TREE_V2)),
            "random_forest": ModelSelector("random_forest", load_model(REVIEW2_PRODUCTION_FOREST_V2)),
            "logistic_regression": ModelSelector(
                "logistic_regression",
                load_model(REVIEW2_PRODUCTION_LOGISTIC_V2),
            ),
            "mlp": ModelSelector("mlp", load_model(REVIEW2_PRODUCTION_MLP_V2)),
        }
    if PRODUCTION_TREE_PATH.exists() and PRODUCTION_FOREST_PATH.exists():
        return {
            "rule_based": RuleSelector(),
            "decision_tree": ModelSelector("decision_tree", load_model(PRODUCTION_TREE_PATH)),
            "random_forest": ModelSelector("random_forest", load_model(PRODUCTION_FOREST_PATH)),
        }
    return saved_selectors()


def _advice(selector_type: str, probabilities: dict[str, float], selected: str) -> SelectorAdvice:
    ranking = _rank(probabilities)
    try_order = tuple(name for name in ranking if name in STRATEGY_NAMES)
    return SelectorAdvice(
        selector_type=selector_type,
        selected_strategy=selected,
        confidence=float(probabilities.get(selected, 0.0)),
        probabilities=dict(probabilities),
        ranking=ranking,
        try_order=try_order,
    )


def _rank(probabilities: dict[str, float]) -> tuple[str, ...]:
    """Highest score first. Equal scores use STRATEGY_PRIORITY, then no_improvement."""
    names = [name for name in (*STRATEGY_NAMES, NO_IMPROVEMENT) if name in probabilities]
    return tuple(sorted(names, key=lambda name: (-probabilities[name], _tie(name))))


def _tie(name: str) -> tuple[int, int]:
    if name == NO_IMPROVEMENT:
        return (1, len(STRATEGY_PRIORITY))
    return (0, STRATEGY_PRIORITY.index(name))


def require_feature_keys(
    features: dict[str, float],
    feature_columns: list[str] | None = None,
) -> None:
    columns = list(feature_columns or FEATURE_COLUMNS)
    missing = [column for column in columns if column not in features]
    if missing:
        raise KeyError(f"selector features are missing {missing}")
