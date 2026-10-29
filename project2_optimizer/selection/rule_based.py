"""Deterministic strategy scores from the eight TAC features.

This selector does not train and does not run the optimizers. A high score
means the features look like that strategy's opportunity. It does not mean
the choice is correct. Correctness is whether the name matches the measured
best_strategy label.
"""

from __future__ import annotations

from evaluation.label_policy import NO_IMPROVEMENT, STRATEGY_PRIORITY
from features.extractor import FEATURE_COLUMNS
from selection.config import RULE_WEIGHTS


def rule_scores(features: dict[str, float]) -> dict[str, float]:
    """Return a score for each of the six optimizers.

    Count features are divided by the number of statements so a long program
    does not win only because it has more lines. `const_ratio` is already on
    that scale.
    """
    statements = float(features.get("num_statements", 0))
    scale = statements if statements else 1.0
    scores: dict[str, float] = {}
    for strategy in STRATEGY_PRIORITY:
        total = 0.0
        for feature_name, weight in RULE_WEIGHTS[strategy].items():
            value = float(features[feature_name])
            if feature_name != "const_ratio":
                value = value / scale
            total += weight * value
        scores[strategy] = total
    return scores


def predict_rule(features: dict[str, float]) -> str:
    """Choose the highest-scoring strategy.

    Every score is zero when the eight features show none of the counted
    opportunities. That prediction is `no_improvement`, not an arbitrary
    optimizer. Equal positive scores use STRATEGY_PRIORITY, earlier first.
    """
    if any(column not in features for column in FEATURE_COLUMNS):
        missing = [column for column in FEATURE_COLUMNS if column not in features]
        raise KeyError(f"rule selector needs the feature columns {missing}")
    scores = rule_scores(features)
    best_score = max(scores.values())
    if best_score <= 0:
        return NO_IMPROVEMENT
    for strategy in STRATEGY_PRIORITY:
        if scores[strategy] == best_score:
            return strategy
    return NO_IMPROVEMENT
