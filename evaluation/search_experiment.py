"""Search controls, fair rule baseline, required plots and unseen benchmark.

Runs on dataset/optimization_dataset.csv (400 programs) with one strict holdout
split grouped by eight-feature signatures, shared by the 8- and 11-feature
schemas. Outputs go to outputs/search/. Run with `python evaluate_search.py`.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.examples import load_sample_programs, load_sample_sources
from dataset.labeler import label_program
from dataset.loader import load_dataset_rows, pair_dataset_examples, programs_for_dataset
from engine.loop import optimize_program
from engine.selectors import (
    FixedOrderSelector,
    ModelSelector,
    RandomOrderSelector,
    RuleSelector,
    RuleSelectorV2,
)
from evaluation.classification import CLASS_ORDER, confusion, metric_bundle
from evaluation.final_experiment import write_dicts, write_matrix
from evaluation.folds import grouped_folds
from evaluation.label_policy import NO_IMPROVEMENT
from evaluation.plots import (
    save_confidence_line,
    save_confusion_heatmap,
    save_constants_vs_score,
    save_feature_importance,
    save_gain_boxplots,
    save_search_efficiency,
    save_strategy_sequences,
)
from features.extended_extractor import FEATURE_COLUMNS_EXTENDED as FEATURE_COLUMNS_V2
from features.extractor import FEATURE_COLUMNS
from optimizers.registry import STRATEGY_NAMES
from selection.classifiers import decision_tree, logistic_regression, mlp_classifier, random_forest
from selection.config import SPLIT_RANDOM_STATE, STRICT_SPLIT
from selection.search_config import (
    ATTEMPT_BUDGETS,
    DEFAULT_EARLY_STOP_THRESHOLD,
    EARLY_STOP_THRESHOLDS,
    RANDOM_ORDER_SEED,
    SEARCH_OUTPUT,
)
from selection.model_config import (
    PRODUCTION_FOREST_EXTENDED,
    PRODUCTION_LOGISTIC_EXTENDED,
    PRODUCTION_MLP_EXTENDED,
    PRODUCTION_TREE_EXTENDED,
)
from selection.rule_based import predict_rule, predict_rule_v2, rule_scores
from selection.splitting import LabeledExample, split_examples

MODEL_FACTORIES = {
    "decision_tree": decision_tree,
    "random_forest": random_forest,
    "logistic_regression": logistic_regression,
    "mlp": mlp_classifier,
}
ML_MODELS = tuple(MODEL_FACTORIES)
# The early-stop policy shown in the plot and used for the unseen benchmark.
REPORTED_EARLY_POLICY = f"early_stop_{DEFAULT_EARLY_STOP_THRESHOLD}"
UNSEEN_EARLY_RUNS = f"optimizer_runs_early_stop_{DEFAULT_EARLY_STOP_THRESHOLD}"
SCHEMAS = {"v1": list(FEATURE_COLUMNS), "v2": list(FEATURE_COLUMNS_V2)}
SCHEMA_NAMES = {"v1": "8 features", "v2": "11 features"}
PER_CLASS_TARGET = 0.70
ACCURACY_TARGET = 0.75
BASELINES = ("fixed_order", "random_order")
UNSEEN_FILE = Path(__file__).resolve().parents[1] / "examples" / "unseen_benchmark.txt"
PRODUCTION_MODELS = {
    "decision_tree": PRODUCTION_TREE_EXTENDED,
    "random_forest": PRODUCTION_FOREST_EXTENDED,
    "logistic_regression": PRODUCTION_LOGISTIC_EXTENDED,
    "mlp": PRODUCTION_MLP_EXTENDED,
}


def run_search_evaluation(output_directory: Path | None = None) -> str:
    destination = output_directory or SEARCH_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_dataset_rows()
    audit = dataset_audit(rows)
    (destination / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    examples = {tag: pair_dataset_examples(rows, columns) for tag, columns in SCHEMAS.items()}
    parts = shared_split(examples, destination)

    holdout = {tag: holdout_schema(tag, *parts[tag]) for tag in SCHEMAS}
    holdout_rows = holdout_metric_rows(holdout)
    write_dicts(destination / "holdout_metrics.csv", holdout_rows)
    for tag in SCHEMAS:
        matrix = confusion(holdout[tag]["actual"], holdout[tag]["predictions"]["random_forest"])
        write_matrix(destination / f"confusion_matrix_random_forest_{tag}.csv", matrix)
    save_confusion_heatmap(
        confusion(holdout["v2"]["actual"], holdout["v2"]["predictions"]["random_forest"]),
        destination / "confusion_matrix_random_forest_v2.png",
        "Strict holdout confusion matrix (random forest, 11 features)",
    )
    save_feature_importance(
        holdout["v2"]["fitted"]["random_forest"].feature_importances(),
        destination / "random_forest_feature_importance_v2.png",
    )

    cv_rows, cv_method = cross_validate_both(examples)
    write_dicts(destination / "cross_validation_folds.csv", cv_rows)
    cv_summary = summarize_cv(cv_rows)
    write_dicts(destination / "cross_validation_summary.csv", cv_summary)

    search_rows, attempt_points, sequences = search_experiments(parts["v2"][1], holdout["v2"]["fitted"])
    write_dicts(destination / "search_runs.csv", search_rows)
    search_summary = summarize_search(search_rows)
    write_dicts(destination / "search_summary.csv", search_summary)
    save_search_efficiency(
        search_summary,
        destination / "search_efficiency.png",
        REPORTED_EARLY_POLICY,
        DEFAULT_EARLY_STOP_THRESHOLD,
    )

    write_required_plots(destination, rows, examples, holdout, attempt_points, sequences)

    unseen_rows = unseen_benchmark()
    write_dicts(destination / "unseen_benchmark.csv", unseen_rows)

    text = format_summary(audit, parts, holdout, cv_summary, cv_method, search_summary, unseen_rows)
    (destination / "experiment_summary.txt").write_text(text + "\n", encoding="utf-8")
    return text


# --- splits and classification ---------------------------------------------------


def shared_split(examples: dict[str, list[LabeledExample]], destination: Path):
    """One strict split grouped by eight-feature signatures, reused for both schemas."""
    train_v1, test_v1, split = split_examples(examples["v1"], STRICT_SPLIT, SCHEMAS["v1"])
    by_id = {tag: {example.program_id: example for example in items} for tag, items in examples.items()}
    parts = {
        tag: (
            [by_id[tag][example.program_id] for example in train_v1],
            [by_id[tag][example.program_id] for example in test_v1],
        )
        for tag in SCHEMAS
    }
    payload = {
        "grouping": "eight_feature_signature",
        "random_state": SPLIT_RANDOM_STATE,
        "split_explanation": split.explanation,
        "train_count": len(train_v1),
        "test_count": len(test_v1),
        "test_program_ids": [example.program_id for example in test_v1],
    }
    (destination / "holdout_program_ids.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return parts


def frame_of(examples: list[LabeledExample], columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame([example.features for example in examples], columns=columns)
    labels = pd.Series([example.label for example in examples], name="best_strategy")
    return features, labels


def fit_models(train: list[LabeledExample], columns: list[str]) -> dict[str, object]:
    x_train, y_train = frame_of(train, columns)
    return {name: factory(columns).fit(x_train, y_train) for name, factory in MODEL_FACTORIES.items()}


def predict_all(tag: str, fitted: dict[str, object], test: list[LabeledExample]) -> dict[str, list[str]]:
    x_test, _ = frame_of(test, SCHEMAS[tag])
    predictions = {"rule_based": [predict_rule(example.features) for example in test]}
    if tag == "v2":
        predictions["rule_based_v2"] = [predict_rule_v2(example.features) for example in test]
    predictions.update({name: list(model.predict(x_test)) for name, model in fitted.items()})
    return predictions


def holdout_schema(tag: str, train: list[LabeledExample], test: list[LabeledExample]) -> dict[str, object]:
    columns = SCHEMAS[tag]
    fitted = fit_models(train, columns)
    actual = [example.label for example in test]
    predictions = predict_all(tag, fitted, test)
    metrics = {name: metric_bundle(actual, predicted) for name, predicted in predictions.items()}
    return {"fitted": fitted, "actual": actual, "predictions": predictions, "metrics": metrics, "test": test}


def holdout_metric_rows(holdout: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tag, result in holdout.items():
        for name, bundle in result["metrics"].items():
            per_class = bundle["per_class"]
            rows.append(
                {
                    "schema": SCHEMA_NAMES[tag],
                    "selector": name,
                    "scope": "overall",
                    "class": "all",
                    "accuracy": round(bundle["accuracy"], 4),
                    "precision": round(bundle["macro_precision"], 4),
                    "recall": round(bundle["macro_recall"], 4),
                    "f1": round(bundle["macro_f1"], 4),
                    "support": len(result["actual"]),
                }
            )
            for label in CLASS_ORDER:
                row = per_class.get(label, {})
                rows.append(
                    {
                        "schema": SCHEMA_NAMES[tag],
                        "selector": name,
                        "scope": "class",
                        "class": label,
                        "accuracy": "",
                        "precision": round(float(row.get("precision", 0.0)), 4),
                        "recall": round(float(row.get("recall", 0.0)), 4),
                        "f1": round(float(row.get("f1-score", 0.0)), 4),
                        "support": int(row.get("support", 0)),
                    }
                )
    return rows


def cross_validate_both(examples: dict[str, list[LabeledExample]]) -> tuple[list[dict[str, object]], str]:
    """Five folds grouped by eight-feature signatures, shared by both schemas."""
    method, folds = grouped_folds(examples["v1"], 5, SCHEMAS["v1"])
    rows: list[dict[str, object]] = []
    for fold_index, (train_index, test_index) in enumerate(folds, start=1):
        for tag in SCHEMAS:
            train = [examples[tag][index] for index in train_index]
            test = [examples[tag][index] for index in test_index]
            fitted = fit_models(train, SCHEMAS[tag])
            actual = [example.label for example in test]
            for name, predicted in predict_all(tag, fitted, test).items():
                bundle = metric_bundle(actual, predicted)
                row = {
                    "schema": SCHEMA_NAMES[tag],
                    "selector": name,
                    "fold": fold_index,
                    "cv_method": method,
                    "accuracy": bundle["accuracy"],
                    "macro_f1": bundle["macro_f1"],
                }
                for label in CLASS_ORDER:
                    row[f"f1_{label}"] = float(bundle["per_class"].get(label, {}).get("f1-score", 0.0))
                    row[f"support_{label}"] = int(bundle["per_class"].get(label, {}).get("support", 0))
                rows.append(row)
    return rows, method


def summarize_cv(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["schema"]), str(row["selector"]))].append(row)
    summary: list[dict[str, object]] = []
    for (schema, selector), items in grouped.items():
        out: dict[str, object] = {"schema": schema, "selector": selector, "folds": len(items)}
        for key in ("accuracy", "macro_f1", *[f"f1_{label}" for label in CLASS_ORDER]):
            values = [float(item[key]) for item in items]
            out[f"{key}_mean"] = round(float(np.mean(values)), 4)
            out[f"{key}_std"] = round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0
        out["min_class_f1_mean"] = min(out[f"f1_{label}_mean"] for label in CLASS_ORDER)
        summary.append(out)
    return summary


# --- search experiments --------------------------------------------------------------


def _policies(selector: str) -> list[tuple[str, dict[str, object]]]:
    policies: list[tuple[str, dict[str, object]]] = [("full", {})]
    if selector not in BASELINES:
        policies.extend(
            (f"early_stop_{threshold}", {"stop_on_no_improvement": threshold})
            for threshold in EARLY_STOP_THRESHOLDS
        )
    policies.extend(
        (f"top_{budget}", {"max_attempts_per_iteration": budget}) for budget in ATTEMPT_BUDGETS
    )
    return policies


def _make_selector(name: str, fitted: dict[str, object], program_index: int):
    if name == "fixed_order":
        return FixedOrderSelector()
    if name == "random_order":
        return RandomOrderSelector(RANDOM_ORDER_SEED + program_index)
    if name == "rule_based":
        return RuleSelector()
    if name == "rule_based_v2":
        return RuleSelectorV2()
    return ModelSelector(name, fitted[name])


SEARCH_SELECTORS = ("fixed_order", "random_order", "rule_based", "rule_based_v2", *ML_MODELS)


def search_experiments(test: list[LabeledExample], fitted: dict[str, object]):
    """Every selector and search policy on each held-out program (11-feature models)."""
    rows: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    sequences: list[tuple[str, tuple[str, ...]]] = []
    for program_index, example in enumerate(test):
        reference = optimize_program(example.program, FixedOrderSelector(), example.program_id)
        reference_text = reference.final_program.to_tac()
        for name in SEARCH_SELECTORS:
            for policy, options in _policies(name):
                selector = _make_selector(name, fitted, program_index)
                run = optimize_program(example.program, selector, example.program_id, **options)
                rows.append(
                    {
                        "program_id": example.program_id,
                        "label": example.label,
                        "selector": name,
                        "policy": policy,
                        "optimizer_runs": run.optimizer_evaluations_total,
                        "failed_runs": run.failed_candidate_attempts,
                        "successful_iterations": run.successful_iterations,
                        "instruction_gain": run.total_instruction_gain,
                        "weighted_cost_gain": run.total_weighted_cost_gain,
                        "reference_instruction_gain": reference.total_instruction_gain,
                        "reference_weighted_cost_gain": reference.total_weighted_cost_gain,
                        "same_final_code_as_full_search": run.final_program.to_tac() == reference_text,
                        "stopped_because": run.stopped_because,
                        "strategy_sequence": " -> ".join(run.strategy_sequence),
                    }
                )
                if name == "random_forest" and policy == "full":
                    for record in run.history:
                        for attempt in record.attempts:
                            points.append(
                                {
                                    "confidence": float(record.candidate_probabilities.get(attempt.strategy, 0.0)),
                                    "instruction_gain": attempt.instruction_gain,
                                    "weighted_cost_gain": attempt.weighted_cost_gain,
                                }
                            )
                    if len(run.strategy_sequence) >= 2:
                        sequences.append((example.program_id, run.strategy_sequence))
    sequences.sort(key=lambda item: (-len(item[1]), item[0]))
    return rows, points, _diverse_sequences(sequences, 5)


def _diverse_sequences(sequences, count):
    """Prefer programs whose accepted sequences differ, so the chart shows variety."""
    chosen, seen = [], set()
    for program_id, steps in sequences:
        if steps not in seen:
            chosen.append((program_id, steps))
            seen.add(steps)
        if len(chosen) == count:
            return chosen
    for item in sequences:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) == count:
            break
    return chosen


def summarize_search(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["selector"]), str(row["policy"]))].append(row)
    baseline_runs = np.mean([float(r["optimizer_runs"]) for r in grouped[("fixed_order", "full")]])
    summary = []
    for selector in SEARCH_SELECTORS:
        for policy, _options in _policies(selector):
            items = grouped[(selector, policy)]
            reference_weighted = sum(int(r["reference_weighted_cost_gain"]) for r in items)
            reference_instruction = sum(int(r["reference_instruction_gain"]) for r in items)
            runs = float(np.mean([float(r["optimizer_runs"]) for r in items]))
            summary.append(
                {
                    "selector": selector,
                    "policy": policy,
                    "programs": len(items),
                    "mean_optimizer_runs": round(runs, 3),
                    "runs_saved_vs_fixed_order_pct": round(100 * (1 - runs / baseline_runs), 1),
                    "mean_failed_runs": round(float(np.mean([float(r["failed_runs"]) for r in items])), 3),
                    "mean_instruction_gain": round(float(np.mean([float(r["instruction_gain"]) for r in items])), 3),
                    "mean_weighted_cost_gain": round(float(np.mean([float(r["weighted_cost_gain"]) for r in items])), 3),
                    "weighted_gain_retained_pct": round(
                        100 * sum(int(r["weighted_cost_gain"]) for r in items) / reference_weighted, 1
                    ) if reference_weighted else 100.0,
                    "instruction_gain_retained_pct": round(
                        100 * sum(int(r["instruction_gain"]) for r in items) / reference_instruction, 1
                    ) if reference_instruction else 100.0,
                    "programs_with_full_result_pct": round(
                        100 * sum(1 for r in items if r["same_final_code_as_full_search"] in (True, "True")) / len(items), 1
                    ),
                }
            )
    return summary


# --- plots ------------------------------------------------------------------------------


_PREFIX = {
    "constant_folding": "cf",
    "constant_propagation": "cp",
    "algebraic_simplification": "alg",
    "common_subexpression_elimination": "cse",
    "dead_code_elimination": "dce",
    "strength_reduction": "sr",
}


def write_required_plots(destination, rows, examples, holdout, points, sequences) -> None:
    """The six plots listed in the problem statement (confusion matrix and feature
    importance are written with the holdout results)."""
    save_confidence_line(points, destination / "confidence_vs_gain.png", "random forest, 11 features, held-out programs")
    save_strategy_sequences(sequences, destination / "strategy_sequences.png")
    instruction = {name: [] for name in STRATEGY_NAMES}
    weighted = {name: [] for name in STRATEGY_NAMES}
    for row in rows:
        label = str(row["best_strategy"])
        if label in _PREFIX:
            instruction[label].append(int(row[f"{_PREFIX[label]}_instruction_gain"]))
            weighted[label].append(int(row[f"{_PREFIX[label]}_weighted_cost_gain"]))
    save_gain_boxplots(instruction, weighted, destination / "reduction_by_strategy_boxplot.png")
    forest = holdout["v2"]["fitted"]["random_forest"]
    frame, _ = frame_of(examples["v2"], SCHEMAS["v2"])
    probabilities = forest.predict_proba(frame)
    folding = [float(row.get("constant_folding", 0.0)) for row in probabilities]
    save_constants_vs_score(
        [float(example.features["num_constants"]) for example in examples["v2"]],
        [rule_scores(example.features)["constant_folding"] for example in examples["v2"]],
        folding,
        [example.label == "constant_folding" for example in examples["v2"]],
        destination / "constants_vs_constant_folding_score.png",
        "Random forest",
    )


# --- production models and the unseen benchmark ---------------------------------------


def unseen_benchmark() -> list[dict[str, object]]:
    """Hand-written programs never used as training rows, run through final production models."""
    from selection.classifiers import load_model

    missing = [name for name in ML_MODELS if not PRODUCTION_MODELS[name].exists()]
    if missing:
        raise FileNotFoundError(
            f"production models missing for {missing}. Run python train_models.py first."
        )
    selectors = {
        "fixed_order": FixedOrderSelector(),
        "rule_based": RuleSelector(),
        "rule_based_v2": RuleSelectorV2(),
        **{name: ModelSelector(name, load_model(path)) for name, path in PRODUCTION_MODELS.items()},
    }
    dataset_texts = {program.to_tac() for _program_id, program in programs_for_dataset()}
    programs = dict(load_sample_programs(UNSEEN_FILE))
    rows = []
    for program_id, _source in load_sample_sources(UNSEEN_FILE):
        program = programs[program_id]
        labeled = label_program(program)
        for name, selector in selectors.items():
            features = selector.prepare_features(program) if hasattr(selector, "prepare_features") else None
            if features is None:
                from features.extractor import extract_features

                features = extract_features(program)
            advice = selector.advise(features)
            run = optimize_program(program, selector, program_id)
            early = (
                optimize_program(program, selector, program_id, stop_on_no_improvement=DEFAULT_EARLY_STOP_THRESHOLD)
                if name != "fixed_order"
                else run
            )
            rows.append(
                {
                    "program_id": program_id,
                    "selector": name,
                    "predicted_strategy": advice.selected_strategy,
                    "actual_strategy": labeled.best_strategy,
                    "correct": advice.selected_strategy == labeled.best_strategy,
                    "final_instruction_gain": run.total_instruction_gain,
                    "final_weighted_cost_gain": run.total_weighted_cost_gain,
                    "optimizer_runs_full": run.optimizer_evaluations_total,
                    UNSEEN_EARLY_RUNS: early.optimizer_evaluations_total,
                    "early_stop_same_final_code": early.final_program.to_tac() == run.final_program.to_tac(),
                    "program_in_dataset": program.to_tac() in dataset_texts,
                    "strategy_sequence": " -> ".join(run.strategy_sequence),
                }
            )
    return rows


# --- summary ------------------------------------------------------------------------------


def format_summary(audit, parts, holdout, cv_summary, cv_method, search_summary, unseen_rows) -> str:
    lines = [
        "SEARCH AND SELECTOR EVALUATION (optimization_dataset.csv)",
        f"Rows: {audit['total_rows']}",
        f"Class distribution: {audit['class_distribution']}",
        f"Unique 8-feature vectors: {audit['unique_feature_vectors_v1']} "
        f"({audit['conflicting_signatures_v1']} conflicting signatures)",
        f"Unique 11-feature vectors: {audit['unique_feature_vectors_v2']} "
        f"({audit['conflicting_signatures_v2']} conflicting signatures)",
        f"Shared strict holdout: {len(parts['v1'][0])} train / {len(parts['v1'][1])} test programs "
        "(grouped by eight-feature signature; same programs for both schemas)",
        "",
    ]
    for tag in SCHEMAS:
        lines.append(f"Strict holdout, {SCHEMA_NAMES[tag]}:")
        for name, bundle in holdout[tag]["metrics"].items():
            per_class = bundle["per_class"]
            f1s = {label: float(per_class.get(label, {}).get("f1-score", 0.0)) for label in CLASS_ORDER}
            below = {label: round(value, 2) for label, value in f1s.items() if value < PER_CLASS_TARGET}
            lines.append(
                f"  {name:<20} accuracy {bundle['accuracy']:.3f}  macro F1 {bundle['macro_f1']:.3f}  "
                f"min class F1 {min(f1s.values()):.2f}  below {PER_CLASS_TARGET}: {below or 'none'}"
            )
        lines.append("")
    lines.append(f"5-fold grouped CV ({cv_method}, folds grouped by eight-feature signature), mean ± std:")
    for row in cv_summary:
        lines.append(
            f"  {row['schema']:<12} {row['selector']:<20} accuracy {row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}  "
            f"macro F1 {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}  min class F1 (mean) {row['min_class_f1_mean']:.2f}"
        )
    lines.append("")
    lines.append("Search experiments on held-out programs (11-feature models):")
    for row in search_summary:
        lines.append(
            f"  {row['selector']:<20} {row['policy']:<15} runs {row['mean_optimizer_runs']:6.2f} "
            f"({row['runs_saved_vs_fixed_order_pct']:5.1f}% fewer than fixed order)  "
            f"weighted gain kept {row['weighted_gain_retained_pct']:5.1f}%  "
            f"full result on {row['programs_with_full_result_pct']:5.1f}% of programs"
        )
    lines.append("")
    lines.append("Unseen hand-written benchmark (final production models, 11 features):")
    by_selector: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in unseen_rows:
        by_selector[str(row["selector"])].append(row)
    for name, items in by_selector.items():
        correct = sum(1 for item in items if item["correct"])
        runs = np.mean([item["optimizer_runs_full"] for item in items])
        early = np.mean([item[UNSEEN_EARLY_RUNS] for item in items])
        lines.append(
            f"  {name:<20} correct {correct}/{len(items)}  runs full {runs:.2f}, "
            f"early stop ({DEFAULT_EARLY_STOP_THRESHOLD}) {early:.2f}"
        )
    lines.append(
        f"  programs also present in the dataset: {sum(1 for item in by_selector['fixed_order'] if item['program_in_dataset'])}"
        f"/{len(by_selector['fixed_order'])}"
    )
    return "\n".join(lines)


def dataset_audit(rows: list[dict[str, object]]) -> dict[str, object]:
    """Class counts and feature-signature conflicts for both schemas."""
    labels = [str(row["best_strategy"]) for row in rows]
    out: dict[str, object] = {
        "total_rows": len(rows),
        "class_distribution": {label: labels.count(label) for label in sorted(set(labels))},
    }
    for tag, columns in SCHEMAS.items():
        seen: dict[tuple, set[str]] = defaultdict(set)
        for row, label in zip(rows, labels):
            seen[tuple(float(row[column]) for column in columns)].add(label)
        out[f"unique_feature_vectors_{tag}"] = len(seen)
        out[f"conflicting_signatures_{tag}"] = sum(1 for values in seen.values() if len(values) > 1)
    return out
