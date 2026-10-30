"""Review-2 evaluation on dataset_v2.csv (does not overwrite Review-1 outputs)."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.generator import GENERATION_SEED
from dataset.v2_builder import audit_dataset_v2
from dataset.v2_loader import load_dataset_v2_rows, pair_v2_examples
from engine.loop import optimize_program
from engine.selectors import FixedOrderSelector, ModelSelector, RuleSelector
from evaluation.classification import confusion, metric_bundle
from evaluation.folds import grouped_folds
from evaluation.final_experiment import cv_statistics, write_dicts, write_matrix
from evaluation.metrics import reduction_percentage, weighted_operation_cost
from evaluation.plots import save_confusion_heatmap, save_feature_importance
from features.extended_extractor import FEATURE_COLUMNS_V2, FEATURE_SCHEMA_V2
from features.extractor import FEATURE_COLUMNS
from selection.classifiers import (
    decision_tree,
    logistic_regression,
    mlp_classifier,
    random_forest,
    save_model,
)
from selection.config import MODEL_RANDOM_STATE, SPLIT_RANDOM_STATE, STRICT_SPLIT
from selection.review2_config import (
    DATASET_VERSION_V2,
    LOGISTIC_PARAMETERS,
    MLP_PARAMETERS,
    REVIEW2_HOLDOUT_FOREST_V1,
    REVIEW2_HOLDOUT_FOREST_V2,
    REVIEW2_HOLDOUT_LOGISTIC_V1,
    REVIEW2_HOLDOUT_LOGISTIC_V2,
    REVIEW2_HOLDOUT_MLP_V1,
    REVIEW2_HOLDOUT_MLP_V2,
    REVIEW2_HOLDOUT_TREE_V1,
    REVIEW2_HOLDOUT_TREE_V2,
    REVIEW2_OUTPUT,
    REVIEW2_PRODUCTION_FOREST_V1,
    REVIEW2_PRODUCTION_FOREST_V2,
    REVIEW2_PRODUCTION_LOGISTIC_V1,
    REVIEW2_PRODUCTION_LOGISTIC_V2,
    REVIEW2_PRODUCTION_MLP_V1,
    REVIEW2_PRODUCTION_MLP_V2,
    REVIEW2_PRODUCTION_TREE_V1,
    REVIEW2_PRODUCTION_TREE_V2,
    V2_DATASET_SEED,
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


def run_review2_evaluation(output_directory: Path | None = None) -> str:
    destination = output_directory or REVIEW2_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_dataset_v2_rows()
    audit = audit_dataset_v2(rows)
    (destination / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    examples_v1 = pair_v2_examples(rows, FEATURE_COLUMNS)
    examples_v2 = pair_v2_examples(rows, FEATURE_COLUMNS_V2)

    holdout_v1 = evaluate_five_selectors(examples_v1, list(FEATURE_COLUMNS), destination, "v1")
    holdout_v2 = evaluate_five_selectors(examples_v2, list(FEATURE_COLUMNS_V2), destination, "v2")
    cv_v1, cv_method_v1 = cross_validate_five(examples_v1, list(FEATURE_COLUMNS), "v1")
    cv_v2, cv_method_v2 = cross_validate_five(examples_v2, list(FEATURE_COLUMNS_V2), "v2")
    ablation = feature_ablation(examples_v1, examples_v2, destination)
    ablation_cv, ablation_cv_method = feature_ablation_cv(examples_v1, examples_v2, destination)
    (destination / "cv_methods.json").write_text(
        json.dumps(
            {
                "five_selector_cv_v1_8_features": cv_method_v1,
                "five_selector_cv_v2_11_features": cv_method_v2,
                "feature_ablation_cv_8_feature_groups": ablation_cv_method,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    practical = practical_five(holdout_v1["test"], holdout_v1["fitted"])
    write_dicts(destination / "optimization_results.csv", practical)
    write_dicts(destination / "holdout_metrics.csv", holdout_v1["metric_rows"])
    write_dicts(destination / "cross_validation_metrics.csv", cv_statistics(cv_v1 + cv_v2))

    summary_lines = [
        "Review-2 evaluation summary",
        f"Dataset: {DATASET_VERSION_V2} ({audit['total_rows']} rows)",
        f"Unique 8-feature vectors: {audit['unique_feature_vectors_v1']}",
        f"Unique 11-feature vectors: {audit['unique_feature_vectors_v2']}",
        f"Extension seed: {V2_DATASET_SEED}, final-200 seed: {GENERATION_SEED}",
        "",
        "Strict holdout (8 features, five selectors):",
    ]
    for name in FIVE_SELECTORS:
        metrics = holdout_v1["metrics"][name]
        summary_lines.append(
            f"  {name}: accuracy {metrics['accuracy']:.3f}, macro F1 {metrics['macro_f1']:.3f}"
        )
    summary_lines.append("")
    summary_lines.append(
        "Feature ablation (shared 8-feature-grouped holdout, same program IDs for both schemas):"
    )
    holdout_ids = json.loads((destination / "ablation_holdout_program_ids.json").read_text(encoding="utf-8"))
    summary_lines.append(f"  train programs: {holdout_ids['train_count']}, test programs: {holdout_ids['test_count']}")
    for row in ablation:
        summary_lines.append(f"  {row['model']} ({row['schema']}): {row['macro_f1']:.3f}")
    summary_lines.append("")
    summary_lines.append("Feature ablation CV (shared 8-feature-grouped folds, mean macro F1):")
    for row in ablation_cv:
        summary_lines.append(f"  {row['model']} ({row['schema']}): {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}")
    text = "\n".join(summary_lines)
    (destination / "experiment_summary.txt").write_text(text + "\n", encoding="utf-8")
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
    paths = _holdout_paths(tag)
    for key, model in fitted.items():
        save_model(
            model,
            paths[key],
            _metadata(feature_columns, "holdout"),
        )
    actual = [example.label for example in test]
    predictions = {
        "rule_based": [predict_rule(example.features) for example in test],
        **{name: model.predict(x_test) for name, model in fitted.items()},
    }
    metrics = {name: metric_bundle(actual, predicted) for name, predicted in predictions.items()}
    metric_rows = [
        {"selector": name, "split": "strict_holdout", **metrics[name]}
        for name in FIVE_SELECTORS
    ]
    matrix = confusion(actual, predictions["random_forest"])
    write_matrix(destination / f"confusion_matrix_random_forest_{tag}.csv", matrix)
    save_confusion_heatmap(
        matrix,
        destination / f"confusion_matrix_random_forest_{tag}.png",
        f"Strict holdout confusion matrix (random forest, {tag})",
    )
    if tag == "v2":
        save_feature_importance(
            fitted["random_forest"].feature_importances(),
            destination / "random_forest_feature_importance_v2.png",
        )
        _write_logistic_coefficients(fitted["logistic_regression"], destination / "logistic_coefficients_v2.csv")
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
    examples_v1: list[LabeledExample],
    examples_v2: list[LabeledExample],
    destination: Path,
) -> list[dict[str, object]]:
    """Holdout ablation with one split grouped by eight-feature signatures."""
    train_v1, test_v1, split = split_examples(examples_v1, STRICT_SPLIT, list(FEATURE_COLUMNS))
    by_id_v2 = {example.program_id: example for example in examples_v2}
    train_v2 = [by_id_v2[example.program_id] for example in train_v1]
    test_v2 = [by_id_v2[example.program_id] for example in test_v1]
    _write_ablation_holdout_ids(destination, train_v1, test_v1, split.explanation)

    rows: list[dict[str, object]] = []
    for schema, train, test, columns in (
        ("v1_8_features", train_v1, test_v1, list(FEATURE_COLUMNS)),
        ("v2_11_features", train_v2, test_v2, list(FEATURE_COLUMNS_V2)),
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
    examples_v1: list[LabeledExample],
    examples_v2: list[LabeledExample],
    destination: Path,
) -> tuple[list[dict[str, object]], str]:
    """Cross-validation ablation using folds grouped by eight-feature signatures."""
    method, folds = grouped_folds(examples_v1, 5, list(FEATURE_COLUMNS))
    by_id_v2 = {example.program_id: example for example in examples_v2}
    fold_rows: list[dict[str, object]] = []
    for fold_index, (train_index, test_index) in enumerate(folds, start=1):
        train_v1 = [examples_v1[index] for index in train_index]
        test_v1 = [examples_v1[index] for index in test_index]
        train_v2 = [by_id_v2[example.program_id] for example in train_v1]
        test_v2 = [by_id_v2[example.program_id] for example in test_v1]
        for schema, train, test, columns in (
            ("v1_8_features", train_v1, test_v1, list(FEATURE_COLUMNS)),
            ("v2_11_features", train_v2, test_v2, list(FEATURE_COLUMNS_V2)),
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


def fit_review2_production(
    examples_v1: list[LabeledExample],
    examples_v2: list[LabeledExample],
) -> None:
    for examples, columns, paths in (
        (examples_v1, list(FEATURE_COLUMNS), _production_paths("v1")),
        (examples_v2, list(FEATURE_COLUMNS_V2), _production_paths("v2")),
    ):
        x_all, y_all = frame_of(examples, columns)
        save_model(decision_tree(columns).fit(x_all, y_all), paths["decision_tree"], _metadata(columns, "production"))
        save_model(random_forest(columns).fit(x_all, y_all), paths["random_forest"], _metadata(columns, "production"))
        save_model(
            logistic_regression(columns).fit(x_all, y_all),
            paths["logistic_regression"],
            _metadata(columns, "production"),
        )
        save_model(mlp_classifier(columns).fit(x_all, y_all), paths["mlp"], _metadata(columns, "production"))


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
    schema = FEATURE_SCHEMA_V2 if len(feature_columns) == len(FEATURE_COLUMNS_V2) else "v1"
    return {
        "feature_schema": schema,
        "feature_columns": feature_columns,
        "dataset_version": DATASET_VERSION_V2,
        "dataset_seed_final": GENERATION_SEED,
        "dataset_seed_extension": V2_DATASET_SEED,
        "training_stage": stage,
        "logistic_parameters": LOGISTIC_PARAMETERS,
        "mlp_parameters": MLP_PARAMETERS,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
    }


def _holdout_paths(tag: str) -> dict[str, Path]:
    mapping = {
        "v1": {
            "decision_tree": REVIEW2_HOLDOUT_TREE_V1,
            "random_forest": REVIEW2_HOLDOUT_FOREST_V1,
            "logistic_regression": REVIEW2_HOLDOUT_LOGISTIC_V1,
            "mlp": REVIEW2_HOLDOUT_MLP_V1,
        },
        "v2": {
            "decision_tree": REVIEW2_HOLDOUT_TREE_V2,
            "random_forest": REVIEW2_HOLDOUT_FOREST_V2,
            "logistic_regression": REVIEW2_HOLDOUT_LOGISTIC_V2,
            "mlp": REVIEW2_HOLDOUT_MLP_V2,
        },
    }
    return mapping[tag]


def _production_paths(tag: str) -> dict[str, Path]:
    mapping = {
        "v1": {
            "decision_tree": REVIEW2_PRODUCTION_TREE_V1,
            "random_forest": REVIEW2_PRODUCTION_FOREST_V1,
            "logistic_regression": REVIEW2_PRODUCTION_LOGISTIC_V1,
            "mlp": REVIEW2_PRODUCTION_MLP_V1,
        },
        "v2": {
            "decision_tree": REVIEW2_PRODUCTION_TREE_V2,
            "random_forest": REVIEW2_PRODUCTION_FOREST_V2,
            "logistic_regression": REVIEW2_PRODUCTION_LOGISTIC_V2,
            "mlp": REVIEW2_PRODUCTION_MLP_V2,
        },
    }
    return mapping[tag]


def _write_logistic_coefficients(model, path: Path) -> None:
    rows = model.standardized_coefficients()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "feature", "coefficient"])
        writer.writeheader()
        writer.writerows(rows)
