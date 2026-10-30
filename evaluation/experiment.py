"""Evaluate the five selectors using holdout, grouped validation, and feature ablation."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.generator import GENERATION_SEED
from dataset.extended_builder import audit_dataset
from dataset.loader import load_dataset_rows, pair_dataset_examples
from engine.loop import optimize_program
from engine.selectors import FixedOrderSelector, ModelSelector, RuleSelector
from evaluation.classification import confusion, metric_bundle
from evaluation.folds import grouped_folds
from evaluation.final_experiment import cv_statistics, write_dicts, write_matrix
from evaluation.metrics import reduction_percentage, weighted_operation_cost
from evaluation.plots import save_confusion_heatmap, save_feature_importance
from features.extended_extractor import FEATURE_COLUMNS_EXTENDED, FEATURE_SCHEMA_EXTENDED, FEATURE_SCHEMA_ORIGINAL
from features.extractor import FEATURE_COLUMNS
from selection.classifiers import (
    decision_tree,
    logistic_regression,
    mlp_classifier,
    random_forest,
    save_model,
)
from selection.config import MODEL_RANDOM_STATE, SPLIT_RANDOM_STATE, STRICT_SPLIT
from selection.model_config import (
    DATASET_VERSION,
    LOGISTIC_PARAMETERS,
    MLP_PARAMETERS,
    EVALUATION_OUTPUT,
    PRODUCTION_FOREST_EXTENDED,
    PRODUCTION_LOGISTIC_EXTENDED,
    PRODUCTION_MLP_EXTENDED,
    PRODUCTION_TREE_EXTENDED,
    EXTENSION_DATASET_SEED,
)
from selection.rule_based import predict_rule
from selection.splitting import LabeledExample, split_examples
FIVE_SELECTORS = (
    "rule_based",
    "decision_tree",
    "random_forest",
    "logistic_regression",
    "mlp",
)


def run_model_evaluation(output_directory: Path | None = None) -> str:
    destination = output_directory or EVALUATION_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_dataset_rows()
    audit = audit_dataset(rows)
    (destination / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    examples_8 = pair_dataset_examples(rows, FEATURE_COLUMNS)
    examples_11 = pair_dataset_examples(rows, FEATURE_COLUMNS_EXTENDED)

    holdout_8 = evaluate_five_selectors(examples_8, list(FEATURE_COLUMNS), destination, "8_features")
    holdout_11 = evaluate_five_selectors(examples_11, list(FEATURE_COLUMNS_EXTENDED), destination, "11_features")
    cv_8, cv_method_8 = cross_validate_five(examples_8, list(FEATURE_COLUMNS), "8_features")
    cv_11, cv_method_11 = cross_validate_five(examples_11, list(FEATURE_COLUMNS_EXTENDED), "11_features")
    ablation = feature_ablation(examples_8, examples_11, destination)
    ablation_cv, ablation_cv_method = feature_ablation_cv(examples_8, examples_11, destination)
    (destination / "cv_methods.json").write_text(
        json.dumps(
            {
                "five_selector_cv_original_8_features": cv_method_8,
                "five_selector_cv_extended_11_features": cv_method_11,
                "feature_ablation_cv_8_feature_groups": ablation_cv_method,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    practical = practical_five(holdout_11["test"], holdout_11["fitted"])
    write_dicts(destination / "optimization_results.csv", practical)
    write_dicts(destination / "holdout_metrics.csv", holdout_8["metric_rows"] + holdout_11["metric_rows"])
    write_dicts(destination / "cross_validation_metrics.csv", cv_statistics(cv_8 + cv_11))

    summary_lines = [
        "Selector evaluation summary",
        f"Dataset: {DATASET_VERSION} ({audit['total_rows']} rows)",
        f"Unique 8-feature vectors: {audit['unique_feature_vectors_8_features']}",
        f"Unique 11-feature vectors: {audit['unique_feature_vectors_11_features']}",
        f"Extension seed: {EXTENSION_DATASET_SEED}, final-200 seed: {GENERATION_SEED}",
        "",
        "Strict holdout results:",
    ]
    for name in FIVE_SELECTORS:
        metrics_8 = holdout_8["metrics"][name]
        metrics_11 = holdout_11["metrics"][name]
        summary_lines.append(
            f"  {name} (8 features): accuracy {metrics_8['accuracy']:.3f}, macro F1 {metrics_8['macro_f1']:.3f}"
        )
        summary_lines.append(
            f"  {name} (11 features): accuracy {metrics_11['accuracy']:.3f}, macro F1 {metrics_11['macro_f1']:.3f}"
        )
    summary_lines.append("")
    summary_lines.append(
        "Feature ablation (same grouped holdout programs for both schemas):"
    )
    holdout_ids = json.loads((destination / "ablation_holdout_program_ids.json").read_text(encoding="utf-8"))
    summary_lines.append(f"  train programs: {holdout_ids['train_count']}, test programs: {holdout_ids['test_count']}")
    for row in ablation:
        summary_lines.append(f"  {row['model']} ({row['schema']}): {row['macro_f1']:.3f}")
    summary_lines.append("")
    summary_lines.append("Feature ablation cross-validation (mean macro F1):")
    for row in ablation_cv:
        summary_lines.append(f"  {row['model']} ({row['schema']}): {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")
    text = "\n".join(summary_lines)
    (destination / "summary.txt").write_text(text + "\n", encoding="utf-8")
    return text


def frame_of(examples: list[LabeledExample], feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame([example.features for example in examples], columns=feature_columns)
    labels = pd.Series([example.label for example in examples], name="best_strategy")
    return features, labels


def evaluate_five_selectors(
    examples: list[LabeledExample],
    feature_columns: list[str],
    destination: Path,
    tag: str,
) -> dict[str, object]:
    train, test, split = split_examples(examples, STRICT_SPLIT, feature_columns)
    x_train, y_train = frame_of(train, feature_columns)
    x_test, _ = frame_of(test, feature_columns)
    fitted = {
        "decision_tree": decision_tree(feature_columns).fit(x_train, y_train),
        "random_forest": random_forest(feature_columns).fit(x_train, y_train),
        "logistic_regression": logistic_regression(feature_columns).fit(x_train, y_train),
        "mlp": mlp_classifier(feature_columns).fit(x_train, y_train),
    }
    actual = [example.label for example in test]
    predictions = {
        "rule_based": [predict_rule(example.features) for example in test],
        **{name: model.predict(x_test) for name, model in fitted.items()},
    }
    metrics = {name: metric_bundle(actual, predicted) for name, predicted in predictions.items()}
    metric_rows = [
        {"selector": name, "split": "strict_holdout", "feature_schema": tag, **metrics[name]}
        for name in FIVE_SELECTORS
    ]
    matrix = confusion(actual, predictions["random_forest"])
    write_matrix(destination / f"confusion_matrix_random_forest_{tag}.csv", matrix)
    save_confusion_heatmap(
        matrix,
        destination / f"confusion_matrix_random_forest_{tag}.png",
        f"Strict holdout confusion matrix (random forest, {tag})",
    )
    if tag == "11_features":
        save_feature_importance(
            fitted["random_forest"].feature_importances(),
            destination / "random_forest_feature_importance_11_features.png",
        )
        _write_logistic_coefficients(fitted["logistic_regression"], destination / "logistic_coefficients_11_features.csv")
    return {
        "split": split,
        "fitted": fitted,
        "test": test,
        "metrics": metrics,
        "metric_rows": metric_rows,
        "actual": actual,
        "predictions": predictions,
    }


def cross_validate_five(
    examples: list[LabeledExample],
    feature_columns: list[str],
    tag: str,
) -> tuple[list[dict[str, object]], str]:
    method, folds = grouped_folds(examples, 5, feature_columns)
    rows: list[dict[str, object]] = []
    for fold_index, (train_index, test_index) in enumerate(folds, start=1):
        train = [examples[index] for index in train_index]
        test = [examples[index] for index in test_index]
        x_train, y_train = frame_of(train, feature_columns)
        x_test, _ = frame_of(test, feature_columns)
        fitted = {
            "decision_tree": decision_tree(feature_columns).fit(x_train, y_train),
            "random_forest": random_forest(feature_columns).fit(x_train, y_train),
            "logistic_regression": logistic_regression(feature_columns).fit(x_train, y_train),
            "mlp": mlp_classifier(feature_columns).fit(x_train, y_train),
        }
        actual = [example.label for example in test]
        predictions = {
            "rule_based": [predict_rule(example.features) for example in test],
            **{name: model.predict(x_test) for name, model in fitted.items()},
        }
        for name, predicted in predictions.items():
            bundle = metric_bundle(actual, predicted)
            rows.append(
                {
                    "selector": name,
                    "schema": tag,
                    "fold": fold_index,
                    "cv_method": method,
                    **bundle,
                }
            )
    return rows, method


def feature_ablation(
    examples_8: list[LabeledExample],
    examples_11: list[LabeledExample],
    destination: Path,
) -> list[dict[str, object]]:
    """Holdout ablation with one split grouped by eight-feature signatures."""
    train_8, test_8, split = split_examples(examples_8, STRICT_SPLIT, list(FEATURE_COLUMNS))
    by_id_11_features = {example.program_id: example for example in examples_11}
    train_11 = [by_id_11_features[example.program_id] for example in train_8]
    test_11 = [by_id_11_features[example.program_id] for example in test_8]
    _write_ablation_holdout_ids(destination, train_8, test_8, split.explanation)

    rows: list[dict[str, object]] = []
    for schema, train, test, columns in (
        ("original_8_features", train_8, test_8, list(FEATURE_COLUMNS)),
        ("extended_11_features", train_11, test_11, list(FEATURE_COLUMNS_EXTENDED)),
    ):
        x_train, y_train = frame_of(train, columns)
        x_test, _ = frame_of(test, columns)
        for factory, name in (
            (decision_tree, "decision_tree"),
            (random_forest, "random_forest"),
            (logistic_regression, "logistic_regression"),
            (mlp_classifier, "mlp"),
        ):
            model = factory(columns).fit(x_train, y_train)
            actual = [example.label for example in test]
            predicted = model.predict(x_test)
            bundle = metric_bundle(actual, predicted)
            rows.append(
                {
                    "model": name,
                    "schema": schema,
                    "split": "shared_strict_holdout_8feature_groups",
                    **bundle,
                }
            )
    write_dicts(destination / "feature_ablation.csv", rows)
    return rows


def feature_ablation_cv(
    examples_8: list[LabeledExample],
    examples_11: list[LabeledExample],
    destination: Path,
) -> tuple[list[dict[str, object]], str]:
    """Cross-validation ablation using folds grouped by eight-feature signatures."""
    method, folds = grouped_folds(examples_8, 5, list(FEATURE_COLUMNS))
    by_id_11_features = {example.program_id: example for example in examples_11}
    fold_rows: list[dict[str, object]] = []
    for fold_index, (train_index, test_index) in enumerate(folds, start=1):
        train_8 = [examples_8[index] for index in train_index]
        test_8 = [examples_8[index] for index in test_index]
        train_11 = [by_id_11_features[example.program_id] for example in train_8]
        test_11 = [by_id_11_features[example.program_id] for example in test_8]
        for schema, train, test, columns in (
            ("original_8_features", train_8, test_8, list(FEATURE_COLUMNS)),
            ("extended_11_features", train_11, test_11, list(FEATURE_COLUMNS_EXTENDED)),
        ):
            x_train, y_train = frame_of(train, columns)
            x_test, _ = frame_of(test, columns)
            for factory, name in (
                (decision_tree, "decision_tree"),
                (random_forest, "random_forest"),
                (logistic_regression, "logistic_regression"),
                (mlp_classifier, "mlp"),
            ):
                model = factory(columns).fit(x_train, y_train)
                actual = [example.label for example in test]
                predicted = model.predict(x_test)
                bundle = metric_bundle(actual, predicted)
                fold_rows.append(
                    {
                        "model": name,
                        "schema": schema,
                        "fold": fold_index,
                        "cv_method": method,
                        "split": "shared_cv_8feature_groups",
                        **bundle,
                    }
                )
    write_dicts(destination / "feature_ablation_cv.csv", fold_rows)
    summary: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in fold_rows:
        grouped[(str(row["model"]), str(row["schema"]))].append(float(row["macro_f1"]))
    for (model, schema), values in sorted(grouped.items()):
        summary.append(
            {
                "model": model,
                "schema": schema,
                "macro_f1_mean": float(np.mean(values)),
                "macro_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "folds": len(values),
            }
        )
    write_dicts(destination / "feature_ablation_cv_summary.csv", summary)
    return summary, method


def _write_ablation_holdout_ids(
    destination: Path,
    train: list[LabeledExample],
    test: list[LabeledExample],
    split_explanation: str,
) -> None:
    payload = {
        "grouping": "eight_feature_signature",
        "random_state": SPLIT_RANDOM_STATE,
        "split_explanation": split_explanation,
        "train_count": len(train),
        "test_count": len(test),
        "train_program_ids": [example.program_id for example in train],
        "test_program_ids": [example.program_id for example in test],
    }
    (destination / "ablation_holdout_program_ids.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def fit_production_models(examples_11: list[LabeledExample]) -> None:
    """Fit the four shipped production classifiers on all extended features."""
    columns = list(FEATURE_COLUMNS_EXTENDED)
    x_all, y_all = frame_of(examples_11, columns)
    fitted = {
        "decision_tree": decision_tree(columns).fit(x_all, y_all),
        "random_forest": random_forest(columns).fit(x_all, y_all),
        "logistic_regression": logistic_regression(columns).fit(x_all, y_all),
        "mlp": mlp_classifier(columns).fit(x_all, y_all),
    }
    paths = {
        "decision_tree": PRODUCTION_TREE_EXTENDED,
        "random_forest": PRODUCTION_FOREST_EXTENDED,
        "logistic_regression": PRODUCTION_LOGISTIC_EXTENDED,
        "mlp": PRODUCTION_MLP_EXTENDED,
    }
    metadata = _metadata(columns, "production")
    for name, model in fitted.items():
        save_model(model, paths[name], metadata)


def practical_five(test: list[LabeledExample], fitted: dict[str, object]) -> list[dict[str, object]]:
    selectors = {
        "rule_based": RuleSelector(),
        "decision_tree": ModelSelector("decision_tree", fitted["decision_tree"]),
        "random_forest": ModelSelector("random_forest", fitted["random_forest"]),
        "logistic_regression": ModelSelector("logistic_regression", fitted["logistic_regression"]),
        "mlp": ModelSelector("mlp", fitted["mlp"]),
        "fixed_order": FixedOrderSelector(),
    }
    rows = []
    for example in test:
        original_cost = weighted_operation_cost(example.program)
        for name, selector in selectors.items():
            run = optimize_program(example.program, selector, example.program_id)
            rows.append(
                {
                    "program_id": example.program_id,
                    "selector": name,
                    "instruction_gain": run.total_instruction_gain,
                    "weighted_cost_gain": run.total_weighted_cost_gain,
                    "candidate_attempts": run.candidate_attempts,
                    "failed_candidate_attempts": run.failed_candidate_attempts,
                    "successful_iterations": run.successful_iterations,
                    "instruction_reduction_percentage": reduction_percentage(example.program, run.final_program),
                    "initial_weighted_cost": original_cost,
                    "final_weighted_cost": weighted_operation_cost(run.final_program),
                    "strategy_sequence": " -> ".join(run.strategy_sequence),
                }
            )
    return rows


def _metadata(feature_columns: list[str], stage: str) -> dict[str, object]:
    schema = FEATURE_SCHEMA_EXTENDED if len(feature_columns) == len(FEATURE_COLUMNS_EXTENDED) else FEATURE_SCHEMA_ORIGINAL
    return {
        "feature_schema": schema,
        "feature_columns": feature_columns,
        "dataset_version": DATASET_VERSION,
        "dataset_seed_final": GENERATION_SEED,
        "dataset_seed_extension": EXTENSION_DATASET_SEED,
        "training_stage": stage,
        "logistic_parameters": LOGISTIC_PARAMETERS,
        "mlp_parameters": MLP_PARAMETERS,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
    }


def _write_logistic_coefficients(model, path: Path) -> None:
    rows = model.standardized_coefficients()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "feature", "coefficient"])
        writer.writeheader()
        writer.writerows(rows)
