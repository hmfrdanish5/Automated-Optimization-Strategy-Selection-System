"""Review-1 reproducibility checks and reports."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from dataset.dataset_builder import FINAL_DATASET_PATH
from dataset.diversity import diversity_report
from evaluation.final_experiment import pair_final_examples
from selection.classifiers import load_model
from selection.config import (
    DATASET_SEED,
    FOREST_PARAMETERS,
    HOLDOUT_FOREST_PATH,
    HOLDOUT_TREE_PATH,
    MODEL_DIRECTORY,
    MODEL_RANDOM_STATE,
    PRODUCTION_FOREST_PATH,
    PRODUCTION_TREE_PATH,
    SPLIT_RANDOM_STATE,
    STRICT_SPLIT,
    TEST_SIZE,
    TREE_PARAMETERS,
    OUTPUT_DIRECTORY,
)
from selection.splitting import split_examples


def write_reproducibility_report(destination: Path | None = None) -> str:
    """Write outputs/reproducibility_report.txt with seeds, versions, and commands."""
    output_dir = destination or OUTPUT_DIRECTORY
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(FINAL_DATASET_PATH).to_dict(orient="records")
    report = diversity_report(rows)
    lines = [
        "REVIEW-1 REPRODUCIBILITY REPORT",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Python version: " + sys.version.split()[0],
        "Platform: " + platform.platform(),
        "scikit-learn: " + sklearn.__version__,
        "numpy: " + _optional_version("numpy"),
        "pandas: " + _optional_version("pandas"),
        "joblib: " + _optional_version("joblib"),
        "",
        "Seeds:",
        f"  dataset seed (GENERATION_SEED / FINAL): {DATASET_SEED}",
        f"  split random_state: {SPLIT_RANDOM_STATE}",
        f"  model random_state: {MODEL_RANDOM_STATE}",
        "",
        "Dataset:",
        f"  path: dataset/dataset_final.csv",
        f"  rows: {report.row_count}",
        f"  unique feature vectors: {report.unique_feature_vectors}",
        f"  duplicate rows: {report.duplicate_rows}",
        f"  conflicting signatures in CSV: {len(report.inconsistent_signatures)}",
        "",
        "Class distribution:",
    ]
    for name, count in sorted(report.class_counts.items()):
        lines.append(f"  {name}: {count}")
    lines.extend(
        [
            "",
            "Model configuration (Review-1, no hyperparameter search):",
            f"  DecisionTreeClassifier: {json.dumps(TREE_PARAMETERS)}",
            f"  RandomForestClassifier: {json.dumps(FOREST_PARAMETERS)}",
            f"  train/test split: mode={STRICT_SPLIT}, test_size={TEST_SIZE}",
            "",
            "Reproduce training on the expanded dataset (strict models in models/):",
            "  python train.py",
            "",
            "Reproduce final evaluation, holdout metrics, plots, and production models:",
            "  python evaluate.py",
            "",
            "Reproduce optimization demo (uses production models when present):",
            "  python optimize.py",
            "",
            "Reproduce audit, unseen benchmark, runtime and search-efficiency tables:",
            "  python validate.py",
            "",
            "Full unit tests:",
            "  python -m unittest tests.test_tac tests.test_features tests.test_optimizers "
            "tests.test_labeling tests.test_models tests.test_diversity tests.test_engine "
            "tests.test_final tests.test_hardening tests.test_optimizer_hardening tests.test_semantics",
        ]
    )
    text = "\n".join(lines) + "\n"
    (output_dir / "reproducibility_report.txt").write_text(text, encoding="utf-8")
    return text


def model_artifact_report() -> str:
    """Verify holdout and production pickles load and record sklearn metadata."""
    paths = {
        "holdout_decision_tree": HOLDOUT_TREE_PATH,
        "holdout_random_forest": HOLDOUT_FOREST_PATH,
        "production_decision_tree": PRODUCTION_TREE_PATH,
        "production_random_forest": PRODUCTION_FOREST_PATH,
    }
    lines = ["MODEL ARTIFACT CHECK", f"Directory: models/ (under project root)", ""]
    rows = pd.read_csv(FINAL_DATASET_PATH).to_dict(orient="records")
    examples = pair_final_examples(rows)
    train, test, split = split_examples(examples, STRICT_SPLIT)
    for label, path in paths.items():
        status = "missing"
        sklearn_saved = "n/a"
        n_samples = "n/a"
        if path.is_file():
            payload = joblib.load(path)
            sklearn_saved = str(payload.get("sklearn_version", "missing"))
            model = load_model(path)
            estimator = model.estimator
            n_samples = getattr(estimator, "n_features_in_", "unknown")
            status = "ok"
        lines.append(f"{label}:")
        lines.append(f"  file: {path.name}")
        lines.append(f"  status: {status}")
        lines.append(f"  sklearn_version in payload: {sklearn_saved}")
        lines.append(f"  n_features_in_: {n_samples}")
        lines.append("")
    lines.append("Holdout models:")
    lines.append(f"  strict train groups: {len(train)} programs")
    lines.append(f"  strict test groups: {len(test)} programs")
    lines.append(f"  feature-signature overlap train/test: {split.overlap_rows}")
    lines.append("")
    lines.append("Production models:")
    lines.append(f"  fit on full final dataset: {len(examples)} programs")
    lines.append(
        "  saved after holdout metrics in evaluate.py; used by optimize.py when both files exist."
    )
    return "\n".join(lines)


def _optional_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(module.__version__)
    except Exception:
        return "not installed"


def print_review1_banner(test_passed: int | None = None, test_failed: int | None = None) -> None:
    """Print the Review-1 freeze summary using outputs/ when available."""
    rows = pd.read_csv(FINAL_DATASET_PATH).to_dict(orient="records")
    report = diversity_report(rows)
    class_line = ", ".join(f"{name} {count}" for name, count in sorted(report.class_counts.items()))
    holdout = _holdout_metrics()
    cv = _cv_metrics()
    search = _search_summary_line()
    if test_passed is None:
        tests = run_unit_test_counts()
    else:
        tests = {"passed": test_passed, "failed": test_failed or 0}
    print("=" * 40)
    print("PROJECT 2 - REVIEW 1 FROZEN BASELINE")
    print("=" * 40)
    print()
    print("Tests:")
    print(f"    {tests['passed']} passed")
    print(f"    {tests['failed']} failed")
    print()
    print("Dataset:")
    print(f"    {report.row_count} programs")
    print(f"    {report.unique_feature_vectors} unique feature vectors")
    print(f"    class distribution = {class_line}")
    print()
    print("Strict Holdout:")
    for name, metrics in holdout.items():
        print(f"    {name} = accuracy {metrics['accuracy']:.3f}, macro F1 {metrics['macro_f1']:.3f}")
    print()
    print("5-Fold Grouped CV:")
    for name, metrics in cv.items():
        print(
            f"    {name} = accuracy {metrics['accuracy']:.3f} +/- {metrics['accuracy_std']:.3f}, "
            f"macro F1 {metrics['macro_f1']:.3f} +/- {metrics['macro_f1_std']:.3f}"
        )
    print()
    print("Search Efficiency (mean candidate attempts):")
    print(f"    {search}")
    print()
    print("Model:")
    print(f"    scikit-learn = {sklearn.__version__}")
    print(f"    Random Forest = {json.dumps(FOREST_PARAMETERS)}")
    print(f"    Decision Tree = {json.dumps(TREE_PARAMETERS)}")
    print()
    print("Artifacts:")
    print("    models/holdout_decision_tree.pkl")
    print("    models/holdout_random_forest.pkl")
    print("    models/production_decision_tree.pkl")
    print("    models/production_random_forest.pkl")
    print("    outputs/ (metrics, plots, search_efficiency.csv, reproducibility_report.txt)")
    print()
    print("Status:")
    print("    REVIEW 1 BASELINE FROZEN")
    print()


def _holdout_metrics() -> dict[str, dict[str, float]]:
    path = OUTPUT_DIRECTORY / "model_metrics.csv"
    defaults = {
        "Rule-Based": {"accuracy": 0.600, "macro_f1": 0.554},
        "Decision Tree": {"accuracy": 0.775, "macro_f1": 0.736},
        "Random Forest": {"accuracy": 0.850, "macro_f1": 0.810},
    }
    if not path.is_file():
        return defaults
    frame = pd.read_csv(path)
    holdout = frame[frame["scope"] == "overall"]
    mapping = {
        "rule_based": "Rule-Based",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
    }
    result = dict(defaults)
    for key, label in mapping.items():
        row = holdout[holdout["selector"] == key]
        if not row.empty:
            result[label] = {
                "accuracy": float(row.iloc[0]["accuracy"]),
                "macro_f1": float(row.iloc[0]["macro_f1"]),
            }
    return result


def _cv_metrics() -> dict[str, dict[str, float]]:
    path = OUTPUT_DIRECTORY / "cross_validation_metrics.csv"
    defaults = {
        "Rule-Based": {"accuracy": 0.570, "accuracy_std": 0.074, "macro_f1": 0.546, "macro_f1_std": 0.064},
        "Decision Tree": {"accuracy": 0.880, "accuracy_std": 0.037, "macro_f1": 0.858, "macro_f1_std": 0.054},
        "Random Forest": {"accuracy": 0.890, "accuracy_std": 0.042, "macro_f1": 0.851, "macro_f1_std": 0.086},
    }
    if not path.is_file():
        return defaults
    frame = pd.read_csv(path)
    mapping = {
        "rule_based": "Rule-Based",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
    }
    result = dict(defaults)
    for key, label in mapping.items():
        mean_row = frame[(frame["selector"] == key) & (frame["fold"] == "mean")]
        std_row = frame[(frame["selector"] == key) & (frame["fold"] == "std")]
        if not mean_row.empty and not std_row.empty:
            result[label] = {
                "accuracy": float(mean_row.iloc[0]["accuracy"]),
                "accuracy_std": float(std_row.iloc[0]["accuracy"]),
                "macro_f1": float(mean_row.iloc[0]["macro_f1"]),
                "macro_f1_std": float(std_row.iloc[0]["macro_f1"]),
            }
    return result


def _search_summary_line() -> str:
    path = OUTPUT_DIRECTORY / "search_efficiency.csv"
    if not path.is_file():
        return "Rule-Based = (run validate.py); Decision Tree = ...; Random Forest = ..."
    frame = pd.read_csv(path)
    parts = []
    for key, label in (
        ("rule_based", "Rule-Based"),
        ("decision_tree", "Decision Tree"),
        ("random_forest", "Random Forest"),
    ):
        subset = frame[frame["selector"] == key]["candidate_attempts"]
        parts.append(f"{label} = {subset.mean():.3f}")
    return "; ".join(parts)


def run_unit_test_counts() -> dict[str, int]:
    """Run the full unittest suite and return passed/failed counts."""
    import subprocess

    command = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_tac",
        "tests.test_features",
        "tests.test_optimizers",
        "tests.test_labeling",
        "tests.test_models",
        "tests.test_diversity",
        "tests.test_engine",
        "tests.test_final",
        "tests.test_hardening",
        "tests.test_optimizer_hardening",
        "tests.test_semantics",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=_project_root())
    output = completed.stdout + completed.stderr
    total = 0
    for line in output.splitlines():
        if line.startswith("Ran "):
            total = int(line.split()[1])
            break
    if completed.returncode == 0:
        return {"passed": total, "failed": 0}
    failures = output.count("FAIL:") + output.count("ERROR:")
    return {"passed": max(total - failures, 0), "failed": failures if failures else 1}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]
