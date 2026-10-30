"""Phase 7.5 checks that do not train a new model.

The 200-program runtime comparison uses the holdout Decision Tree and
Random Forest. Those models were fit only on the strict training groups.
Running them on all 200 programs is an operating comparison, and some of
those programs were seen during training. It is not a pure generalization
estimate. The hand-written unseen file is the separate check.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import pandas as pd

from dataset.dataset_builder import FINAL_DATASET_PATH
from dataset.diversity import diversity_report, feature_signature, format_diversity_report
from dataset.examples import load_sample_programs, load_sample_sources
from dataset.generator import FINAL_GENERATED_COUNT, GENERATION_SEED, generate_final_programs
from dataset.labeler import GAIN_PREFIX, label_program
from engine.loop import optimize_program
from engine.selectors import ModelSelector, RuleSelector
from evaluation.final_experiment import pair_final_examples
from evaluation.label_policy import NO_IMPROVEMENT
from features.extractor import FEATURE_COLUMNS, extract_features
from selection.classifiers import load_model
from selection.config import (
    HOLDOUT_FOREST_PATH,
    HOLDOUT_TREE_PATH,
    OUTPUT_DIRECTORY,
    STRICT_SPLIT,
)
from selection.splitting import split_examples
from tac.parser import parse_tac

UNSEEN_FILE = Path(__file__).resolve().parents[1] / "examples" / "unseen_benchmark.txt"
LOW_CONFIDENCE = 0.5
HOLDOUT_NOTE = (
    "This is a full-dataset runtime comparison using the holdout-trained models, "
    "so some programs were seen during training; it is not a pure generalization estimate."
)


def load_final_rows(path: Path | None = None) -> list[dict[str, object]]:
    """Read the existing final CSV. This does not regenerate or relabel it."""
    frame = pd.read_csv(path or FINAL_DATASET_PATH)
    return frame.to_dict(orient="records")


def audit_final_dataset(rows: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    """Describe the final CSV without changing it."""
    report = diversity_report(rows)
    frame = pd.DataFrame(rows)
    numeric = frame[list(FEATURE_COLUMNS)].apply(pd.to_numeric)
    zero_variance = [column for column in FEATURE_COLUMNS if numeric[column].nunique(dropna=False) <= 1]
    correlations = {}
    for left in FEATURE_COLUMNS:
        correlations[left] = {}
        for right in FEATURE_COLUMNS:
            if left in zero_variance or right in zero_variance:
                correlations[left][right] = "undefined"
            else:
                correlations[left][right] = round(float(numeric[left].corr(numeric[right])), 3)
    categories = _categories()
    by_category: dict[str, dict[str, int]] = {}
    unknown = 0
    for row in rows:
        category = categories.get(str(row["program_id"]))
        if category is None:
            category = "unknown"
            unknown += 1
        counts = by_category.setdefault(category, {})
        label = str(row["best_strategy"])
        counts[label] = counts.get(label, 0) + 1
    conflicts = [
        {
            "signature": list(record.signature),
            "labels": list(record.labels),
            "program_ids": list(record.program_ids),
        }
        for record in report.inconsistent_signatures
    ]
    payload = {
        "total_samples": report.row_count,
        "unique_feature_vectors": report.unique_feature_vectors,
        "duplicate_rows": report.duplicate_rows,
        "conflicting_signatures": conflicts,
        "class_distribution": dict(sorted(report.class_counts.items())),
        "feature_min_max": {
            column: {"min": report.ranges[column][0], "max": report.ranges[column][1]}
            for column in FEATURE_COLUMNS
        },
        "zero_variance_features": zero_variance,
        "pairwise_pearson": correlations,
        "label_distribution_by_category": by_category,
        "unknown_category_rows": unknown,
        "label_distribution": dict(sorted(report.class_counts.items())),
    }
    correlation_frame = pd.DataFrame(correlations).reindex(index=list(FEATURE_COLUMNS), columns=list(FEATURE_COLUMNS))
    lines = [
        "DATASET AUDIT",
        "The CSV was not modified.",
        format_diversity_report(report),
        "",
        "Zero-variance features: " + (", ".join(zero_variance) if zero_variance else "none"),
        "",
        "Pairwise feature correlations (Pearson). Undefined when a feature has zero variance:",
        correlation_frame.to_string(),
        "",
        "Label distribution by generator category:",
    ]
    for category in sorted(by_category):
        parts = ", ".join(f"{name} {count}" for name, count in sorted(by_category[category].items()))
        lines.append(f"  {category}: {parts}")
    lines.extend(["", "Label distribution independent of generator category:"])
    for name, count in sorted(report.class_counts.items()):
        lines.append(f"  {name}: {count}")
    lines.extend(["", _limitation_example()])
    return "\n".join(lines), payload


def evaluate_unseen(selectors: dict[str, object]) -> list[dict[str, object]]:
    """Label and run the hand-written programs that were never training rows."""
    programs = dict(load_sample_programs(UNSEEN_FILE))
    rows = []
    for program_id, source in load_sample_sources(UNSEEN_FILE):
        program = programs[program_id]
        features = extract_features(program)
        labeled = label_program(program)
        score_by_name = {score.strategy: score for score in labeled.scores}
        for name, selector in selectors.items():
            advice = selector.advise(features)
            run = optimize_program(program, selector, program_id)
            chosen = score_by_name.get(advice.selected_strategy)
            rows.append(
                {
                    "program_id": program_id,
                    "selector": name,
                    "predicted_strategy": advice.selected_strategy,
                    "actual_strategy": labeled.best_strategy,
                    "correct": advice.selected_strategy == labeled.best_strategy,
                    "confidence": "" if name == "rule_based" else advice.confidence,
                    "predicted_instruction_gain": 0 if chosen is None else chosen.instruction_gain,
                    "predicted_weighted_cost_gain": 0 if chosen is None else chosen.weighted_cost_gain,
                    "final_instruction_gain": run.total_instruction_gain,
                    "final_weighted_cost_gain": run.total_weighted_cost_gain,
                    "strategy_sequence": " -> ".join(run.strategy_sequence),
                    "tac": source,
                }
            )
    return rows


def runtime_comparison(examples, selectors: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run every final program through each selector.

    The second list is one row per attempted optimizer, used for confidence.
    """
    rows = []
    attempts = []
    for example in examples:
        for name, selector in selectors.items():
            run = optimize_program(example.program, selector, example.program_id)
            rows.append(_runtime_row(example.program_id, name, run))
            if name == "rule_based":
                continue
            for record in run.history:
                for attempt in record.attempts:
                    attempts.append(
                        {
                            "program_id": example.program_id,
                            "selector": name,
                            "strategy": attempt.strategy,
                            "confidence": record.candidate_probabilities.get(attempt.strategy, 0.0),
                            "instruction_gain": attempt.instruction_gain,
                            "weighted_cost_gain": attempt.weighted_cost_gain,
                        }
                    )
    return rows, attempts


def _runtime_row(program_id: str, selector: str, run) -> dict[str, object]:
    return {
        "program_id": program_id,
        "selector": selector,
        "initial_instruction_count": run.initial_instruction_count,
        "final_instruction_count": run.final_instruction_count,
        "instruction_gain": run.total_instruction_gain,
        "weighted_cost_gain": run.total_weighted_cost_gain,
        "successful_iterations": run.successful_iterations,
        "strategy_sequence": " -> ".join(run.strategy_sequence),
        "candidate_attempts": run.candidate_attempts,
        "successful_candidate_attempts": run.successful_candidate_attempts,
        "failed_candidate_attempts": run.failed_candidate_attempts,
        "optimizer_evaluations_total": run.optimizer_evaluations_total,
    }


def search_efficiency_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Per-program search metrics for outputs/search_efficiency.csv."""
    return [
        {
            "program_id": row["program_id"],
            "selector": row["selector"],
            "candidate_attempts": row["candidate_attempts"],
            "successful_candidate_attempts": row["successful_candidate_attempts"],
            "failed_candidate_attempts": row["failed_candidate_attempts"],
            "optimizer_evaluations_total": row["optimizer_evaluations_total"],
            "final_instruction_gain": row["instruction_gain"],
            "final_weighted_cost_gain": row["weighted_cost_gain"],
            "successful_iterations": row["successful_iterations"],
            "strategy_sequence": row["strategy_sequence"],
        }
        for row in rows
    ]


def summarize_search_efficiency(rows: list[dict[str, object]]) -> str:
    """Mean, median, and std of search counts; summary table for Review-1."""
    display = {
        "rule_based": "Rule-Based",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
    }
    lines = [
        "SEARCH EFFICIENCY (200-program runtime, holdout ML models)",
        HOLDOUT_NOTE,
        "Fewer attempts does not automatically mean better final optimization quality.",
        "These counts measure how much search each selector performs.",
        "",
        "| Selector | Mean Attempts | Median Attempts | Std Attempts | "
        "Mean Successful Iterations | Mean Failed Attempts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in display.items():
        subset = [row for row in rows if row["selector"] == key]
        attempts = [int(row["candidate_attempts"]) for row in subset]
        failed = [int(row["failed_candidate_attempts"]) for row in subset]
        iterations = [int(row["successful_iterations"]) for row in subset]
        evals = [int(row["optimizer_evaluations_total"]) for row in subset]
        lines.append(
            f"| {label} | {statistics.mean(attempts):.3f} | {statistics.median(attempts):.3f} | "
            f"{statistics.stdev(attempts):.3f} | {statistics.mean(iterations):.3f} | "
            f"{statistics.mean(failed):.3f} |"
        )
        lines.append(
            f"  optimizer_evaluations_total: mean {statistics.mean(evals):.3f}, "
            f"median {statistics.median(evals):.3f}, std {statistics.stdev(evals):.3f}"
        )
    return "\n".join(lines)


def summarize_gains(rows: list[dict[str, object]]) -> str:
    """Mean, median, and sample standard deviation for each selector."""
    lines = [
        "200-PROGRAM RUNTIME COMPARISON",
        "Decision Tree and Random Forest are the holdout models "
        "(models/holdout_decision_tree.pkl and models/holdout_random_forest.pkl).",
        HOLDOUT_NOTE,
    ]
    for selector in ("rule_based", "decision_tree", "random_forest"):
        subset = [row for row in rows if row["selector"] == selector]
        lines.append(selector)
        for column in ("instruction_gain", "weighted_cost_gain"):
            values = [int(row[column]) for row in subset]
            lines.append(
                f"  {column}: mean {statistics.mean(values):.3f}, "
                f"median {statistics.median(values):.3f}, "
                f"std {statistics.stdev(values):.3f}"
            )
    return "\n".join(lines)


def confidence_report(records: list[dict[str, object]]) -> str:
    """Confidence of each attempted optimizer, with the gain of that attempt."""
    lines = [
        "CONFIDENCE ANALYSIS",
        "Models: holdout Decision Tree and holdout Random Forest, run on all 200 final programs.",
        HOLDOUT_NOTE,
        f"Low confidence is below {LOW_CONFIDENCE}. High confidence is at least {LOW_CONFIDENCE}.",
        "These are associations. They do not show that confidence causes gain.",
    ]
    for selector in ("decision_tree", "random_forest"):
        subset = [row for row in records if row["selector"] == selector]
        if not subset:
            lines.append(f"{selector}: no attempts")
            continue
        confidences = [float(row["confidence"]) for row in subset]
        lines.append(
            f"{selector}: attempts {len(subset)}, "
            f"mean confidence {statistics.mean(confidences):.3f}, "
            f"median confidence {statistics.median(confidences):.3f}"
        )
        for column in ("instruction_gain", "weighted_cost_gain"):
            lines.append(
                f"  Pearson correlation with {column}: "
                f"{_pearson(confidences, [int(row[column]) for row in subset])}"
            )
        low = [row for row in subset if float(row["confidence"]) < LOW_CONFIDENCE]
        high = [row for row in subset if float(row["confidence"]) >= LOW_CONFIDENCE]
        lines.append("  " + _gain_band("low", low))
        lines.append("  " + _gain_band("high", high))
    return "\n".join(lines)


def holdout_failures(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    """Incorrect Decision Tree and Random Forest predictions on the strict holdout.

    The loaded pickles were trained on that same split. Overlap between the
    training groups and the test groups is reported and is expected to be 0.
    """
    examples = pair_final_examples(rows)
    _train, test, split = split_examples(examples, STRICT_SPLIT)
    by_id = {str(row["program_id"]): row for row in rows}
    failures = []
    models = {
        "decision_tree": load_model(HOLDOUT_TREE_PATH),
        "random_forest": load_model(HOLDOUT_FOREST_PATH),
    }
    for name, model in models.items():
        selector = ModelSelector(name, model)
        for example in test:
            advice = selector.advise(example.features)
            if advice.selected_strategy == example.label:
                continue
            row = by_id[example.program_id]
            failures.append(
                _failure("strict_holdout", name, example.program.to_tac(), example.features, example.label, advice, row)
            )
    return failures, split.overlap_rows


def unseen_failures(selectors: dict[str, object]) -> list[dict[str, object]]:
    """Incorrect ML predictions on the hand-written unseen programs."""
    failures = []
    for program_id, program in load_sample_programs(UNSEEN_FILE):
        labeled = label_program(program)
        features = extract_features(program)
        for name in ("decision_tree", "random_forest"):
            advice = selectors[name].advise(features)
            if advice.selected_strategy == labeled.best_strategy:
                continue
            failures.append(
                _failure_from_label("unseen", name, program_id, program.to_tac(), features, labeled, advice)
            )
    return failures


def write_validation(output_directory: Path | None = None) -> str:
    """Write the audit, unseen benchmark, runtime table, and failure file."""
    destination = output_directory or OUTPUT_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_final_rows()
    audit, audit_payload = audit_final_dataset(rows)
    examples = pair_final_examples(rows)
    selectors = _holdout_selectors()
    unseen = evaluate_unseen(selectors)
    runtime, attempts = runtime_comparison(examples, selectors)
    confidence = confidence_report(attempts)
    holdout, overlap = holdout_failures(rows)
    unseen_misses = unseen_failures(selectors)
    failures = holdout + unseen_misses
    gains = summarize_gains(runtime)
    search_rows = search_efficiency_rows(runtime)
    search_summary = summarize_search_efficiency(runtime)
    unseen_text = _unseen_text(unseen)
    failure_text = _failure_text(failures, overlap)
    (destination / "dataset_audit.txt").write_text(audit + "\n", encoding="utf-8")
    (destination / "dataset_audit.json").write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    _write_csv(destination / "unseen_benchmark.csv", unseen)
    _write_csv(destination / "runtime_comparison.csv", runtime)
    (destination / "runtime_comparison_summary.txt").write_text(gains + "\n", encoding="utf-8")
    _write_csv(destination / "search_efficiency.csv", search_rows)
    (destination / "search_efficiency_summary.txt").write_text(search_summary + "\n", encoding="utf-8")
    (destination / "confidence_analysis.txt").write_text(confidence + "\n", encoding="utf-8")
    (destination / "failure_analysis.txt").write_text(failure_text + "\n", encoding="utf-8")
    _write_csv(destination / "failure_analysis.csv", [_flat_failure(item) for item in failures])
    print(audit)
    print(unseen_text)
    print(gains)
    print(search_summary)
    print(confidence)
    print(failure_text)
    return gains


def _holdout_selectors() -> dict[str, object]:
    return {
        "rule_based": RuleSelector(),
        "decision_tree": ModelSelector("decision_tree", load_model(HOLDOUT_TREE_PATH)),
        "random_forest": ModelSelector("random_forest", load_model(HOLDOUT_FOREST_PATH)),
    }


def _limitation_example() -> str:
    """Two programs with one feature vector and two measured labels.

    They are reported here and are not added to the training CSV.
    """
    killed = parse_tac("t1 = a + b\na = 4\nt2 = a + b\nprint t1\nprint t2\n")
    repeated = parse_tac("t1 = left + right\nt2 = left + right\nprint t1\nprint t2\nprint 0\n")
    rows = []
    for program_id, program in (("expression_redefined", killed), ("safe_repeated_add", repeated)):
        features = extract_features(program)
        labeled = label_program(program)
        rows.append((program_id, feature_signature(features), labeled.best_strategy))
    lines = ["Constructed same-feature examples (not added to the dataset):"]
    for program_id, signature, label in rows:
        lines.append(f"  {program_id}: label {label}, signature {signature}")
    if rows[0][1] == rows[1][1] and rows[0][2] != rows[1][2]:
        lines.append(
            "Feature representation is not sufficient to uniquely determine the label for this signature."
        )
    return "\n".join(lines)


def _categories() -> dict[str, str]:
    categories = {program_id: "handwritten" for program_id, _program in load_sample_programs()}
    for program_id, _program, category in generate_final_programs(FINAL_GENERATED_COUNT, GENERATION_SEED):
        categories[program_id] = category
    return categories


def _failure(split_name, selector, program, features, actual, advice, row) -> dict[str, object]:
    gains = {}
    for strategy, prefix in GAIN_PREFIX.items():
        gains[strategy] = {
            "instruction_gain": _plain(row[f"{prefix}_instruction_gain"]),
            "weighted_cost_gain": _plain(row[f"{prefix}_weighted_cost_gain"]),
        }
    gains[NO_IMPROVEMENT] = {"instruction_gain": 0, "weighted_cost_gain": 0}
    return {
        "set": split_name,
        "selector": selector,
        "program_id": row["program_id"],
        "program": program,
        "features": {key: _plain(value) for key, value in features.items()},
        "actual_strategy": actual,
        "predicted_strategy": advice.selected_strategy,
        "probabilities": {key: _plain(value) for key, value in advice.probabilities.items()},
        "strategy_gains": gains,
    }


def _failure_from_label(split_name, selector, program_id, program, features, labeled, advice) -> dict[str, object]:
    gains = {
        score.strategy: {
            "instruction_gain": score.instruction_gain,
            "weighted_cost_gain": score.weighted_cost_gain,
        }
        for score in labeled.scores
    }
    gains[NO_IMPROVEMENT] = {"instruction_gain": 0, "weighted_cost_gain": 0}
    return {
        "set": split_name,
        "selector": selector,
        "program_id": program_id,
        "program": program,
        "features": {key: _plain(value) for key, value in features.items()},
        "actual_strategy": labeled.best_strategy,
        "predicted_strategy": advice.selected_strategy,
        "probabilities": {key: _plain(value) for key, value in advice.probabilities.items()},
        "strategy_gains": gains,
    }


def _failure_text(failures: list[dict[str, object]], overlap: int) -> str:
    lines = [
        "FAILURE ANALYSIS",
        "Holdout models were loaded and applied to the strict test split.",
        f"Feature-signature overlap between that train split and test split: {overlap}.",
        "Unseen rows are the hand-written benchmark, which was not a training row.",
        f"Failure records: {len(failures)}.",
    ]
    for item in failures:
        lines.extend(
            [
                "",
                f"[{item['set']}] {item['selector']} / {item['program_id']}",
                f"actual {item['actual_strategy']}",
                f"predicted {item['predicted_strategy']}",
                f"probabilities {item['probabilities']}",
                f"features {item['features']}",
                f"strategy gains {item['strategy_gains']}",
                "TAC:",
                str(item["program"]),
            ]
        )
    return "\n".join(lines)


def _flat_failure(item: dict[str, object]) -> dict[str, object]:
    flat = {
        "set": item["set"],
        "selector": item["selector"],
        "program_id": item["program_id"],
        "actual_strategy": item["actual_strategy"],
        "predicted_strategy": item["predicted_strategy"],
        "program": item["program"],
    }
    for name, value in item["features"].items():
        flat[name] = value
    for name, value in item["probabilities"].items():
        flat[f"prob_{name}"] = value
    for name, gains in item["strategy_gains"].items():
        flat[f"{name}_instruction_gain"] = gains["instruction_gain"]
        flat[f"{name}_weighted_cost_gain"] = gains["weighted_cost_gain"]
    return flat


def _plain(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    return value


def _gain_band(name: str, rows: list[dict[str, object]]) -> str:
    if not rows:
        return f"{name}: count 0"
    instruction = [int(row["instruction_gain"]) for row in rows]
    weighted = [int(row["weighted_cost_gain"]) for row in rows]
    return (
        f"{name}: count {len(rows)}, "
        f"instruction gain mean {statistics.mean(instruction):.3f} median {statistics.median(instruction):.3f}, "
        f"weighted-cost gain mean {statistics.mean(weighted):.3f} median {statistics.median(weighted):.3f}"
    )


def _pearson(left: list[float], right: list[int]) -> str:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return "undefined"
    correlation = pd.Series(left).corr(pd.Series(right, dtype="float64"))
    if pd.isna(correlation):
        return "undefined"
    return f"{float(correlation):.3f}"


def _unseen_text(rows: list[dict[str, object]]) -> str:
    lines = [
        "UNSEEN BENCHMARK",
        "Selectors: RuleSelector, holdout Decision Tree, holdout Random Forest.",
        "These programs are not rows of dataset_final.csv.",
    ]
    for selector in ("rule_based", "decision_tree", "random_forest"):
        subset = [row for row in rows if row["selector"] == selector]
        correct = sum(1 for row in subset if row["correct"])
        instruction = [int(row["final_instruction_gain"]) for row in subset]
        weighted = [int(row["final_weighted_cost_gain"]) for row in subset]
        lines.append(
            f"{selector}: {correct}/{len(subset)} predicted labels match the measured label; "
            f"mean iterative instruction gain {statistics.mean(instruction):.3f}; "
            f"mean iterative weighted-cost gain {statistics.mean(weighted):.3f}"
        )
        for row in subset:
            lines.append(
                f"  {row['program_id']}: predicted {row['predicted_strategy']}, "
                f"actual {row['actual_strategy']}, confidence {row['confidence']}, "
                f"predicted gain {row['predicted_instruction_gain']}/{row['predicted_weighted_cost_gain']}, "
                f"final gain {row['final_instruction_gain']}/{row['final_weighted_cost_gain']}, "
                f"sequence {row['strategy_sequence'] or '(none)'}"
            )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
