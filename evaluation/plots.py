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
from matplotlib.patches import Patch

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
    # Explicit patches: empty bar containers do not carry their colour into
    # the legend, which made every entry the same default blue.
    used = [name for name in STRATEGY_NAMES if any(name in steps for _name, steps in sequences)]
    handles = [Patch(facecolor=STRATEGY_COLORS[name], label=name.replace("_", " ")) for name in used]
    axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    axis.set_yticks(range(0, longest + 1))
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


# --- Final-stage plots ------------------------------------------------------------


def save_confidence_line(points: list[dict[str, float]], path: Path, selector_label: str) -> None:
    """Line chart of mean gain per confidence bin, with the raw attempts behind it."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    edges = [index / 10 for index in range(11)]
    for axis, key, ylabel in (
        (axes[0], "instruction_gain", "Instruction-count gain"),
        (axes[1], "weighted_cost_gain", "Weighted-cost gain"),
    ):
        xs = [float(point["confidence"]) for point in points]
        ys = [float(point[key]) for point in points]
        axis.scatter(xs, ys, alpha=0.18, color="#9AA9B8", s=14, label="individual attempts")
        centers, means = [], []
        for low, high in zip(edges, edges[1:]):
            values = [y for x, y in zip(xs, ys) if (low <= x < high) or (high == 1.0 and x == 1.0)]
            if values:
                centers.append((low + high) / 2)
                means.append(sum(values) / len(values))
        axis.plot(centers, means, marker="o", color="#2F5D8C", linewidth=2.2, label="mean gain per 0.1 bin")
        axis.set_xlabel("Predicted probability of the attempted strategy")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.02, 1.02)
        axis.set_title(ylabel)
        axis.legend(frameon=False, loc="upper left")
    fig.suptitle(f"Decision confidence vs optimization gain ({selector_label})")
    fig.tight_layout()
    _write(fig, path)


def save_gain_boxplots(
    instruction_gains: dict[str, list[int]],
    weighted_gains: dict[str, list[int]],
    path: Path,
) -> None:
    """Instruction and weighted-cost reduction of each strategy where it is the label."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = [SHORT_CLASS_NAMES[name] for name in STRATEGY_NAMES]
    for axis, gains, ylabel in (
        (axes[0], instruction_gains, "Instruction-count reduction"),
        (axes[1], weighted_gains, "Weighted-cost reduction"),
    ):
        series = [gains.get(name) or [0] for name in STRATEGY_NAMES]
        box = axis.boxplot(series, tick_labels=labels, patch_artist=True)
        for patch, name in zip(box["boxes"], STRATEGY_NAMES):
            patch.set_facecolor(STRATEGY_COLORS[name])
            patch.set_alpha(0.55)
        axis.axhline(0, color="#888888", linewidth=0.8)
        axis.set_xlabel("Strategy (programs where it is the best strategy)")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
    fig.suptitle(
        "Reduction by strategy type\n"
        "Only dead-code elimination removes instructions; CF, ALG, CSE and SR save operation cost; "
        "CP wins on smaller code",
        fontsize=11,
    )
    fig.tight_layout()
    _write(fig, path)


def save_constants_vs_score(
    constants: list[float],
    rule_scores: list[float],
    model_scores: list[float],
    is_folding_label: list[bool],
    path: Path,
    model_label: str,
) -> None:
    """num_constants against the constant-folding score (rule score and model probability)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, scores, ylabel in (
        (axes[0], rule_scores, "Rule-based constant-folding score"),
        (axes[1], model_scores, f"{model_label} probability of constant folding"),
    ):
        for flag, color, label in ((False, "#B8C2CC", "other label"), (True, "#4C78A8", "label is constant folding")):
            xs = [x for x, f in zip(constants, is_folding_label) if f is flag]
            ys = [y for y, f in zip(scores, is_folding_label) if f is flag]
            axis.scatter(xs, ys, alpha=0.7, s=22, color=color, label=label)
        axis.set_xlabel("num_constants")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        axis.legend(frameon=False, loc="lower right")
    fig.suptitle("num_constants vs constant-folding score")
    fig.tight_layout()
    _write(fig, path)


SELECTOR_LABELS = {
    "fixed_order": "Fixed\norder",
    "random_order": "Random\norder",
    "rule_based": "Rule-Based\n(8 features)",
    "rule_based_v2": "Rule-Based v2\n(11 features)",
    "decision_tree": "Decision\nTree",
    "random_forest": "Random\nForest",
    "logistic_regression": "Logistic\nRegression",
    "mlp": "MLP",
}


def save_search_efficiency(
    rows: list[dict[str, object]],
    path: Path,
    early_policy: str,
    threshold: float,
) -> None:
    """Mean optimizer runs per program: full search, and early stop where the selector supports it."""
    full: dict[str, float] = {}
    early: dict[str, float] = {}
    for row in rows:
        if row["policy"] == "full":
            full[str(row["selector"])] = float(row["mean_optimizer_runs"])
        elif row["policy"] == early_policy:
            early[str(row["selector"])] = float(row["mean_optimizer_runs"])
    selectors = list(full)
    fig, axis = plt.subplots(figsize=(11, 4.8))
    width = 0.38
    bars: list[tuple[float, float]] = []
    for index, name in enumerate(selectors):
        if name in early:
            axis.bar(index - width / 2, full[name], width, color="#4C78A8")
            axis.bar(index + width / 2, early[name], width, color="#9ECAE9")
            bars.extend([(index - width / 2, full[name]), (index + width / 2, early[name])])
        else:
            axis.bar(index, full[name], width, color="#9D755D")
            bars.append((index, full[name]))
    for x, value in bars:
        axis.text(x, value + 0.15, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(range(len(selectors)), [SELECTOR_LABELS.get(name, name) for name in selectors], fontsize=9)
    axis.set_ylabel("Mean optimizer runs per program")
    axis.set_ylim(0, max(full.values()) * 1.15)
    axis.set_title("Search effort on held-out programs (lower is better)")
    axis.legend(
        handles=[
            Patch(color="#9D755D", label="Baseline order, full search"),
            Patch(color="#4C78A8", label="Selector, full search"),
            Patch(color="#9ECAE9", label=f"Selector, early stop (confidence >= {threshold})"),
        ],
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()
    _write(fig, path)
