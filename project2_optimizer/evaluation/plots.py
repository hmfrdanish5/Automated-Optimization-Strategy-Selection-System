"""Plots for the final evaluation.

Each function writes one PNG. The numbers in the plot are the caller's
measurements. Feature importance is not a causal effect, and a confidence
value is not a promise of instruction-count gain.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.classification import SHORT_CLASS_NAMES
from optimizers.registry import STRATEGY_NAMES


STRATEGY_COLORS = {
    "constant_folding": "#4C78A8",
    "constant_propagation": "#F58518",
    "algebraic_simplification": "#54A24B",
    "common_subexpression_elimination": "#E45756",
    "dead_code_elimination": "#72B7B2",
    "strength_reduction": "#B279A2",
    "no_improvement": "#9D755D",
}


def save_confusion_heatmap(matrix, path: Path, title: str) -> None:
    """Rows are the measured label and columns are the prediction."""
    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    labels = [SHORT_CLASS_NAMES[name] for name in (
        "constant_folding",
        "constant_propagation",
        "algebraic_simplification",
        "common_subexpression_elimination",
        "dead_code_elimination",
        "strength_reduction",
        "no_improvement",
    )]
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted strategy")
    axis.set_ylabel("Measured strategy")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center", color="black")
    fig.tight_layout()
    _write(fig, path)


def save_feature_importance(importances: dict[str, float], path: Path) -> None:
    """Bar chart sorted from the largest importance to the smallest."""
    ordered = sorted(importances, key=importances.get, reverse=True)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh(
        list(reversed(ordered)),
        [importances[name] for name in reversed(ordered)],
        color="#4C78A8",
    )
    axis.set_xlabel("Mean decrease in impurity")
    axis.set_ylabel("Feature")
    axis.set_title("Random forest feature importance (not a causal effect)")
    fig.tight_layout()
    _write(fig, path)


def save_confidence_scatter(points: list[dict[str, float]], path: Path) -> None:
    """Confidence of the attempted class against the measured gain of that attempt."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _scatter(axes[0], points, "instruction_gain", "Instruction-count gain")
    _scatter(axes[1], points, "weighted_cost_gain", "Weighted-cost gain")
    fig.suptitle("Model confidence does not guarantee optimization gain")
    fig.tight_layout()
    _write(fig, path)


def save_strategy_sequences(sequences: list[tuple[str, tuple[str, ...]]], path: Path) -> None:
    """One stacked bar per program. Each block is one accepted iteration."""
    fig, axis = plt.subplots(figsize=(11, 5.2))
    names = [name.replace("_", "\n") for name, _steps in sequences]
    bottoms = [0.0] * len(sequences)
    longest = max((len(steps) for _name, steps in sequences), default=0)
    for step in range(longest):
        heights = []
        colors = []
        for _name, steps in sequences:
            if step < len(steps):
                heights.append(1.0)
                colors.append(STRATEGY_COLORS[steps[step]])
            else:
                heights.append(0.0)
                colors.append("#ffffff")
        axis.bar(names, heights, bottom=bottoms, color=colors, edgecolor="white")
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
    for strategy in STRATEGY_NAMES:
        axis.bar([], [], color=STRATEGY_COLORS[strategy], label=strategy)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    axis.set_ylabel("Accepted iteration")
    axis.set_xlabel("Program")
    axis.set_title("Optimization sequence by program")
    fig.tight_layout()
    _write(fig, path)


def save_gain_boxplot(gains: dict[str, list[int]], path: Path) -> None:
    """Instruction-count gain of each optimizer on the final dataset."""
    fig, axis = plt.subplots(figsize=(9, 4.8))
    labels = [SHORT_CLASS_NAMES[name] for name in STRATEGY_NAMES]
    series = [gains.get(name, [0]) or [0] for name in STRATEGY_NAMES]
    axis.boxplot(series, tick_labels=labels)
    axis.axhline(0, color="#888888", linewidth=0.8)
    axis.set_xlabel("Strategy")
    axis.set_ylabel("Instruction-count gain")
    axis.set_title("Instruction-count gain by strategy")
    fig.tight_layout()
    _write(fig, path)


def save_constants_scatter(constants: list[float], gains: list[float], path: Path, ylabel: str) -> None:
    """num_constants against the measured constant-folding gain."""
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.scatter(constants, gains, alpha=0.75, color="#4C78A8")
    axis.set_xlabel("num_constants")
    axis.set_ylabel(ylabel)
    axis.set_title("Constants versus constant-folding gain")
    fig.tight_layout()
    _write(fig, path)


def instruction_gains_by_strategy(rows: list[dict[str, object]]) -> dict[str, list[int]]:
    """Read the diagnostic instruction-gain columns."""
    prefixes = {
        "constant_folding": "cf",
        "constant_propagation": "cp",
        "algebraic_simplification": "alg",
        "common_subexpression_elimination": "cse",
        "dead_code_elimination": "dce",
        "strength_reduction": "sr",
    }
    return {
        strategy: [int(row[f"{prefix}_instruction_gain"]) for row in rows]
        for strategy, prefix in prefixes.items()
    }


def _scatter(axis, points: list[dict[str, float]], key: str, ylabel: str) -> None:
    if points:
        axis.scatter(
            [float(point["confidence"]) for point in points],
            [float(point[key]) for point in points],
            alpha=0.7,
            color="#4C78A8",
        )
    axis.set_xlabel("Predicted probability of the attempted strategy")
    axis.set_ylabel(ylabel)
    axis.set_title(ylabel)


def _write(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
