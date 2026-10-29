"""Final-dataset evaluation.

Holdout scores come from models trained only on the strict training groups.
Production models are fit afterwards on every final row and stored separately.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.dataset_builder import (
    DATASET_PATH,
    EXPANDED_DATASET_PATH,
    FINAL_DATASET_PATH,
    write_final_dataset,
)
from dataset.diversity import diversity_report, format_diversity_report
from dataset.examples import load_sample_programs
from dataset.generator import FINAL_GENERATED_COUNT, GENERATION_SEED, generate_final_programs
from dataset.labeler import GAIN_PREFIX
from engine.loop import optimize_program
from engine.selectors import FixedOrderSelector, ModelSelector, RankingSelector, RuleSelector
from evaluation.classification import CLASS_ORDER, confusion, metric_bundle
from evaluation.folds import grouped_folds
from evaluation.metrics import reduction_percentage, weighted_operation_cost
from evaluation.plots import (
    instruction_gains_by_strategy,
    save_confidence_scatter,
    save_constants_scatter,
    save_confusion_heatmap,
    save_feature_importance,
    save_gain_boxplot,
    save_strategy_sequences,
)
from features.extractor import FEATURE_COLUMNS
from selection.classifiers import decision_tree, random_forest, save_model
from selection.config import (
    DATASET_SEED,
    HOLDOUT_FOREST_PATH,
    HOLDOUT_TREE_PATH,
    MODEL_RANDOM_STATE,
    OUTPUT_DIRECTORY,
    PRODUCTION_FOREST_PATH,
    PRODUCTION_TREE_PATH,
    SPLIT_RANDOM_STATE,
    STRICT_SPLIT,
)
from selection.rule_based import predict_rule
from selection.splitting import LabeledExample, split_examples
from tac.parser import parse_tac

LIMITATIONS = (
    "The dataset is synthetic. The TAC language is simplified. "
    "Optimization semantics use project-specific assumptions. "
    "Weighted operation costs are relative educational weights. "
    "The feature representation is intentionally small, so data-flow context "
    "is not fully represented and some identical feature vectors have different labels. "
    "These results are not production compiler performance. "
    "Grouped cross-validation does not remove the limits of synthetic data. "
    "Classification accuracy, optimization gain, and confidence are separate measurements."
)

SEQUENCE_IDS = (
    "constant_heavy",
    "algebraic_heavy",
    "cse_heavy",
    "dead_code_heavy",
    "loop_heavy",
)


def run_final_evaluation(output_directory: Path | None = None) -> str:
    """Write the final dataset, evaluate the selectors, and save the plots."""
    destination = output_directory or OUTPUT_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    rows = write_final_dataset()
    examples = pair_final_examples(rows)
    report = diversity_report(rows)
    _write_dataset_summary(destination, report)
    print(format_diversity_report(report))
    print(f"Dataset seed: {DATASET_SEED}")
    print(f"Split random_state: {SPLIT_RANDOM_STATE}")
    print(f"Model random_state: {MODEL_RANDOM_STATE}")

    train, test, split = split_examples(examples, STRICT_SPLIT)
    print(split.explanation)
    print(f"Feature-vector overlap between train/test: {split.overlap_rows}")
    holdout = fit_holdout(train, test)
    cv_name, cv_rows = cross_validate(examples)
    practical, points = practical_runs(test, holdout["tree"], holdout["forest"])
    gain_lookup = _gain_lookup(rows)
    write_metric_tables(destination, holdout, cv_rows, practical, gain_lookup)
    write_plots(destination, rows, holdout, points)
    fit_production(examples)
    summary = format_summary(report, split, holdout, cv_name, cv_rows, practical, points, destination)
    (destination / "experiment_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return summary


def pair_final_examples(rows: list[dict[str, object]]) -> list[LabeledExample]:
    """Pair final CSV rows with the TAC programs that produced them."""
    programs = [(program_id, program) for program_id, program in load_sample_programs()]
    programs.extend(
        (program_id, program)
        for program_id, program, _category in generate_final_programs(
            FINAL_GENERATED_COUNT, GENERATION_SEED
        )
    )
    identifiers = [str(row["program_id"]) for row in rows]
    if [program_id for program_id, _program in programs] != identifiers:
        raise ValueError("final dataset rows are not in the same order as the seeded programs")
    return [
        LabeledExample(
            program_id=str(row["program_id"]),
            program=program,
            features={column: row[column] for column in FEATURE_COLUMNS},
            label=str(row["best_strategy"]),
        )
        for (_program_id, program), row in zip(programs, rows)
    ]


def fit_holdout(train: list[LabeledExample], test: list[LabeledExample]) -> dict[str, object]:
    """Train on the strict training groups and score the held-out groups."""
    x_train, y_train = frame_of(train)
    x_test, _y_test = frame_of(test)
    tree = decision_tree().fit(x_train, y_train)
    forest = random_forest().fit(x_train, y_train)
    save_model(tree, HOLDOUT_TREE_PATH)
    save_model(forest, HOLDOUT_FOREST_PATH)
    actual = [example.label for example in test]
    predictions = {
        "rule_based": [predict_rule(example.features) for example in test],
        "decision_tree": tree.predict(x_test),
        "random_forest": forest.predict(x_test),
    }
    return {
        "tree": tree,
        "forest": forest,
        "actual": actual,
        "predictions": predictions,
        "metrics": {name: metric_bundle(actual, predicted) for name, predicted in predictions.items()},
        "test": test,
        "probabilities": forest.predict_proba(x_test),
        "importances": forest.feature_importances(),
    }


def cross_validate(examples: list[LabeledExample]) -> tuple[str, list[dict[str, object]]]:
    """Score each selector on the same grouped folds."""
    method, folds = grouped_folds(examples, 5)
    rows: list[dict[str, object]] = []
    for fold_index, (train_index, test_index) in enumerate(folds, start=1):
        train = [examples[index] for index in train_index]
        test = [examples[index] for index in test_index]
        x_train, y_train = frame_of(train)
        x_test, _y_test = frame_of(test)
        fitted = {
            "decision_tree": decision_tree().fit(x_train, y_train),
            "random_forest": random_forest().fit(x_train, y_train),
        }
        actual = [example.label for example in test]
        predictions = {
            "rule_based": [predict_rule(example.features) for example in test],
            "decision_tree": fitted["decision_tree"].predict(x_test),
            "random_forest": fitted["random_forest"].predict(x_test),
        }
        for name, predicted in predictions.items():
            bundle = metric_bundle(actual, predicted)
            rows.append(
                {
                    "selector": name,
                    "fold": fold_index,
                    "accuracy": bundle["accuracy"],
                    "macro_f1": bundle["macro_f1"],
                }
            )
    return method, rows


def practical_runs(test, tree, forest):
    """Run the engine on the holdout programs with each selector."""
    selectors = {
        "rule_based": RuleSelector(),
        "decision_tree": ModelSelector("decision_tree", tree),
        "random_forest": ModelSelector("random_forest", forest),
        "fixed_order": FixedOrderSelector(),
    }
    rows = []
    points = []
    for example in test:
        original_cost = weighted_operation_cost(example.program)
        for name, selector in selectors.items():
            run = optimize_program(example.program, selector, example.program_id)
            rows.append(
                {
                    "program_id": example.program_id,
                    "selector": name,
                    "initial_instruction_count": run.initial_instruction_count,
                    "final_instruction_count": run.final_instruction_count,
                    "instruction_gain": run.total_instruction_gain,
                    "instruction_reduction_percentage": reduction_percentage(
                        example.program, run.final_program
                    ),
                    "initial_weighted_cost": original_cost,
                    "final_weighted_cost": weighted_operation_cost(run.final_program),
                    "weighted_cost_gain": run.total_weighted_cost_gain,
                    "successful_iterations": run.successful_iterations,
                    "strategy_sequence": " -> ".join(run.strategy_sequence),
                }
            )
            if name in {"decision_tree", "random_forest"}:
                points.extend(confidence_points(name, run))
    return rows, points


def confidence_points(selector_name: str, run) -> list[dict[str, object]]:
    """Pair each attempted strategy's probability with that attempt's gains."""
    points = []
    for record in run.history:
        for attempt in record.attempts:
            points.append(
                {
                    "selector": selector_name,
                    "program_id": run.program_id,
                    "strategy": attempt.strategy,
                    "confidence": float(record.candidate_probabilities.get(attempt.strategy, 0.0)),
                    "instruction_gain": attempt.instruction_gain,
                    "weighted_cost_gain": attempt.weighted_cost_gain,
                    "accepted": attempt.accepted,
                }
            )
    return points


def mean_gains(practical: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in practical:
        grouped[str(row["selector"])].append(row)
    return {
        name: {
            "instruction_gain": sum(int(row["instruction_gain"]) for row in items) / len(items),
            "weighted_cost_gain": sum(int(row["weighted_cost_gain"]) for row in items) / len(items),
        }
        for name, items in grouped.items()
    }


def cv_statistics(cv_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Append mean and sample standard deviation rows to the per-fold scores."""
    output = list(cv_rows)
    by_selector: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in cv_rows:
        by_selector[str(row["selector"])].append(row)
    for name, items in by_selector.items():
        for label, function in (("mean", np.mean), ("std", lambda values: np.std(values, ddof=1))):
            output.append(
                {
                    "selector": name,
                    "fold": label,
                    "accuracy": float(function([float(row["accuracy"]) for row in items])),
                    "macro_f1": float(function([float(row["macro_f1"]) for row in items])),
                }
            )
    return output


def frame_of(examples: list[LabeledExample]) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame([example.features for example in examples], columns=FEATURE_COLUMNS)
    labels = pd.Series([example.label for example in examples], name="best_strategy")
    return features, labels


def fit_production(examples: list[LabeledExample]) -> None:
    """Fit models on every final row after the holdout scores have been stored."""
    features, labels = frame_of(examples)
    save_model(decision_tree().fit(features, labels), PRODUCTION_TREE_PATH)
    save_model(random_forest().fit(features, labels), PRODUCTION_FOREST_PATH)


def write_metric_tables(destination, holdout, cv_rows, practical, gain_lookup) -> None:
    write_dicts(destination / "model_metrics.csv", metric_rows(holdout, practical))
    write_dicts(destination / "cross_validation_metrics.csv", cv_statistics(cv_rows))
    matrix = confusion(holdout["actual"], holdout["predictions"]["random_forest"])
    write_matrix(destination / "confusion_matrix.csv", matrix)
    (destination / "confusion_matrix.json").write_text(
        json.dumps(
            {
                "labels": list(CLASS_ORDER),
                "matrix": matrix.tolist(),
                "model": "holdout random forest",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_dicts(destination / "optimization_results.csv", practical)
    write_dicts(
        destination / "holdout_probabilities.csv",
        probability_rows(holdout, gain_lookup),
    )


def write_plots(destination, rows, holdout, points) -> None:
    matrix = confusion(holdout["actual"], holdout["predictions"]["random_forest"])
    save_confusion_heatmap(
        matrix,
        destination / "confusion_matrix.png",
        "Strict holdout confusion matrix (random forest)",
    )
    save_feature_importance(holdout["importances"], destination / "random_forest_feature_importance.png")
    save_confidence_scatter(points, destination / "confidence_vs_gain.png")
    save_strategy_sequences(sequence_demo(holdout["forest"]), destination / "strategy_sequences.png")
    save_gain_boxplot(
        instruction_gains_by_strategy(rows),
        destination / "instruction_reduction_by_strategy.png",
    )
    save_constants_scatter(
        [float(row["num_constants"]) for row in rows],
        [float(row["cf_weighted_cost_gain"]) for row in rows],
        destination / "constants_vs_constant_folding.png",
        "Constant-folding weighted-cost gain",
    )


def sequence_demo(forest) -> list[tuple[str, tuple[str, ...]]]:
    programs = dict(load_sample_programs())
    selector = ModelSelector("random_forest", forest)
    sequences = []
    for program_id in SEQUENCE_IDS:
        run = optimize_program(programs[program_id], selector, program_id)
        sequences.append((program_id, run.strategy_sequence))
    fallback = RankingSelector(
        "fallback",
        {
            "constant_folding": 0.70,
            "dead_code_elimination": 0.20,
            "common_subexpression_elimination": 0.04,
            "algebraic_simplification": 0.02,
            "constant_propagation": 0.02,
            "strength_reduction": 0.01,
            "no_improvement": 0.01,
        },
    )
    fallback_run = optimize_program(
        parse_tac("t1 = left + right\nspare = 4\nprint t1\n"),
        fallback,
        "fallback",
    )
    sequences.append(("fallback", fallback_run.strategy_sequence))
    return sequences


def metric_rows(holdout, practical) -> list[dict[str, object]]:
    means = mean_gains(practical)
    rows = []
    for name, bundle in holdout["metrics"].items():
        rows.append(_overall_metric_row(name, bundle, means, len(holdout["actual"])))
        for class_name in CLASS_ORDER:
            cell = bundle["per_class"][class_name]
            rows.append(
                {
                    "selector": name,
                    "scope": "class",
                    "class": class_name,
                    "accuracy": "",
                    "precision": cell["precision"],
                    "recall": cell["recall"],
                    "f1": cell["f1-score"],
                    "support": cell["support"],
                    "macro_f1": "",
                    "weighted_f1": "",
                    "mean_instruction_gain": "",
                    "mean_weighted_cost_gain": "",
                    "model": "strict holdout training split",
                }
            )
    rows.append(
        {
            "selector": "fixed_order",
            "scope": "practical",
            "class": "all",
            "accuracy": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "support": "",
            "macro_f1": "",
            "weighted_f1": "",
            "mean_instruction_gain": means["fixed_order"]["instruction_gain"],
            "mean_weighted_cost_gain": means["fixed_order"]["weighted_cost_gain"],
            "model": "fixed STRATEGY_PRIORITY order, not a classifier",
        }
    )
    return rows


def _overall_metric_row(name, bundle, means, support) -> dict[str, object]:
    return {
        "selector": name,
        "scope": "overall",
        "class": "all",
        "accuracy": bundle["accuracy"],
        "precision": bundle["macro_precision"],
        "recall": bundle["macro_recall"],
        "f1": bundle["macro_f1"],
        "support": support,
        "macro_f1": bundle["macro_f1"],
        "weighted_f1": bundle["weighted_f1"],
        "mean_instruction_gain": means[name]["instruction_gain"],
        "mean_weighted_cost_gain": means[name]["weighted_cost_gain"],
        "model": "strict holdout training split",
    }


def probability_rows(holdout, gain_lookup: dict[tuple[str, str], tuple[int, int]]) -> list[dict[str, object]]:
    rows = []
    for example, probabilities, predicted in zip(
        holdout["test"],
        holdout["probabilities"],
        holdout["predictions"]["random_forest"],
    ):
        instruction_gain, weighted_gain = gain_lookup.get((example.program_id, predicted), (0, 0))
        rows.append(
            {
                "program_id": example.program_id,
                "actual_class": example.label,
                "predicted_class": predicted,
                "confidence": probabilities.get(predicted, 0.0),
                **{f"probability_{name}": probabilities.get(name, 0.0) for name in CLASS_ORDER},
                "selected_strategy_instruction_gain": instruction_gain,
                "selected_strategy_weighted_cost_gain": weighted_gain,
                "model": "holdout random forest",
            }
        )
    return rows


def _gain_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str], tuple[int, int]]:
    lookup = {}
    for row in rows:
        for strategy, prefix in GAIN_PREFIX.items():
            lookup[(str(row["program_id"]), strategy)] = (
                int(row[f"{prefix}_instruction_gain"]),
                int(row[f"{prefix}_weighted_cost_gain"]),
            )
    return lookup


def format_summary(report, split, holdout, cv_name, cv_rows, practical, points, destination: Path) -> str:
    means = mean_gains(practical)
    lines = [
        "========================================",
        "FINAL PROJECT 2 EVALUATION",
        "========================================",
        "Dataset:",
        f"    samples = {report.row_count}",
        f"    unique feature vectors = {report.unique_feature_vectors}",
        f"    duplicate rows = {report.duplicate_rows}",
        f"    conflicting signatures = {len(report.inconsistent_signatures)}",
        "Class distribution:",
    ]
    for name in CLASS_ORDER:
        lines.append(f"    {name}: {report.class_counts.get(name, 0)}")
    lines.extend(
        [
            "",
            "STRICT HOLDOUT",
            f"    {split.explanation}",
            f"    overlap = {split.overlap_rows}",
            "    Models were trained only on the training groups.",
        ]
    )
    for name in ("rule_based", "decision_tree", "random_forest"):
        bundle = holdout["metrics"][name]
        lines.extend(_selector_block(name, bundle))
    lines.extend(["", "5-FOLD GROUPED CROSS-VALIDATION", f"    method: {cv_name}"])
    for name in ("rule_based", "decision_tree", "random_forest"):
        mean_row = next(row for row in stats_rows(cv_rows) if row["selector"] == name and row["fold"] == "mean")
        std_row = next(row for row in stats_rows(cv_rows) if row["selector"] == name and row["fold"] == "std")
        lines.append(f"    {name}:")
        lines.append(f"        accuracy mean +/- std = {mean_row['accuracy']:.3f} +/- {std_row['accuracy']:.3f}")
        lines.append(f"        macro F1 mean +/- std = {mean_row['macro_f1']:.3f} +/- {std_row['macro_f1']:.3f}")
    lines.extend(["", "PRACTICAL OPTIMIZATION", "    Same strict holdout programs for every selector."])
    for name in ("rule_based", "decision_tree", "random_forest", "fixed_order"):
        lines.append(f"    {name}:")
        lines.append(f"        mean instruction gain = {means[name]['instruction_gain']:.3f}")
        lines.append(f"        mean weighted-cost gain = {means[name]['weighted_cost_gain']:.3f}")
    lines.extend(["", "Most important RF features (holdout model, not causal):"])
    for name in sorted(holdout["importances"], key=holdout["importances"].get, reverse=True):
        lines.append(f"    {name}: {holdout['importances'][name]:.4f}")
    lines.extend(["", "Confidence versus gain (attempted strategy probability):"])
    for gain_name in ("instruction_gain", "weighted_cost_gain"):
        coefficient = pearson(
            [float(point["confidence"]) for point in points],
            [float(point[gain_name]) for point in points],
        )
        shown = "not computed (too little variation)" if coefficient is None else f"{coefficient:.3f}"
        lines.append(f"    Pearson correlation with {gain_name}: {shown}")
    lines.extend(
        [
            "",
            "Limitations:",
            f"    {LIMITATIONS}",
            "",
            "Saved plots:",
            "    outputs/confusion_matrix.png",
            "    outputs/random_forest_feature_importance.png",
            "    outputs/confidence_vs_gain.png",
            "    outputs/strategy_sequences.png",
            "    outputs/instruction_reduction_by_strategy.png",
            "    outputs/constants_vs_constant_folding.png",
            "",
            f"Seeds: dataset {DATASET_SEED}, split {SPLIT_RANDOM_STATE}, model {MODEL_RANDOM_STATE}",
            f"Preserved: {DATASET_PATH.name}, {EXPANDED_DATASET_PATH.name}",
            f"Final dataset: {FINAL_DATASET_PATH.name}",
            f"Holdout models: {HOLDOUT_TREE_PATH.name}, {HOLDOUT_FOREST_PATH.name}",
            f"Production models: {PRODUCTION_TREE_PATH.name}, {PRODUCTION_FOREST_PATH.name}",
            f"Output directory: {destination}",
        ]
    )
    return "\n".join(lines)


def _selector_block(name: str, bundle: dict[str, object]) -> list[str]:
    weak = [
        class_name
        for class_name in CLASS_ORDER
        if bundle["per_class"][class_name]["f1-score"] < 0.5
    ]
    lines = [
        f"    {name}:",
        f"        accuracy = {bundle['accuracy']:.3f}",
        f"        macro precision = {bundle['macro_precision']:.3f}",
        f"        macro recall = {bundle['macro_recall']:.3f}",
        f"        macro F1 = {bundle['macro_f1']:.3f}",
        f"        weighted F1 = {bundle['weighted_f1']:.3f}",
        f"        classes with F1 below 0.50: {', '.join(weak) if weak else 'none'}",
    ]
    for class_name in CLASS_ORDER:
        cell = bundle["per_class"][class_name]
        lines.append(
            f"        {class_name}: precision {cell['precision']:.3f}, "
            f"recall {cell['recall']:.3f}, F1 {cell['f1-score']:.3f}, support {int(cell['support'])}"
        )
    return lines


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Correlation of paired observations, or None when it is not meaningful."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def stats_rows(cv_rows):
    return [row for row in cv_statistics(cv_rows) if row["fold"] in {"mean", "std"}]


def _write_dataset_summary(destination: Path, report) -> None:
    text = "\n".join(
        [
            format_diversity_report(report),
            "",
            f"Dataset seed: {DATASET_SEED}",
            f"Split random_state: {SPLIT_RANDOM_STATE}",
            f"Model random_state: {MODEL_RANDOM_STATE}",
            f"Preserved: {DATASET_PATH.name}, {EXPANDED_DATASET_PATH.name}",
            f"Final dataset: {FINAL_DATASET_PATH.name}",
        ]
    )
    (destination / "dataset_summary.txt").write_text(text + "\n", encoding="utf-8")
    payload = {
        "samples": report.row_count,
        "unique_feature_vectors": report.unique_feature_vectors,
        "duplicate_rows": report.duplicate_rows,
        "conflicting_signatures": len(report.inconsistent_signatures),
        "class_counts": report.class_counts,
        "ranges": {name: [bounds[0], bounds[1]] for name, bounds in report.ranges.items()},
        "dataset_seed": DATASET_SEED,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
    }
    (destination / "dataset_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_matrix(path: Path, matrix) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual", *CLASS_ORDER])
        for label, row in zip(CLASS_ORDER, matrix):
            writer.writerow([label, *[int(value) for value in row]])


def write_dicts(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
