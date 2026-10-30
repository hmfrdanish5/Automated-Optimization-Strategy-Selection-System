"""Verify a Git-ready project archive has the expected source and runtime artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

DEFAULT_ZIP = Path(__file__).resolve().parents[2] / "project2_optimizer_git_ready.zip"
PREFIX = "project2_optimizer/"
DATASET_FEATURES = (
    "num_statements", "num_constants", "num_repeated_expr", "num_dead_stmts",
    "num_algebraic_ops", "num_loop_ops", "num_tmp_vars", "const_ratio",
    "num_constant_propagation_uses", "num_safe_repeated_expr", "num_mul_by_two_in_loops",
)
REQUIRED_PATHS = (
    ".gitignore", "README.md", "requirements.txt", "app.py", "train_models.py",
    "evaluate_models.py", "source_frontend/compiler.py", "source_frontend/parser.py",
    "selection/model_config.py", "engine/selectors.py", "dataset/optimization_dataset.csv",
    "models/production_decision_tree_11_features.pkl",
    "models/production_random_forest_11_features.pkl",
    "models/production_logistic_regression_11_features.pkl",
    "models/production_mlp_11_features.pkl",
    "outputs/evaluation/holdout_metrics.csv",
    "outputs/evaluation/cross_validation_metrics.csv",
    "outputs/evaluation/feature_ablation.csv", "outputs/evaluation/summary.txt",
    "tests/test_source_compiler.py", "tests/test_source_correctness.py",
    "tests/test_semantic_fuzz.py", "package_submission.py",
)


def _is_forbidden(name: str) -> bool:
    lowered = name.lower()
    return (
        "__pycache__" in lowered or ".pytest_cache" in lowered
        or "/.git/" in lowered or lowered.endswith("/.git")
        or lowered.endswith((".pyc", ".pyo", ".zip"))
    )


def verify(zip_path: Path) -> list[str]:
    failures: list[str] = []
    if not zip_path.is_file():
        return [f"archive not found: {zip_path}"]
    try:
        with zipfile.ZipFile(zip_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                return [f"corrupt archive member: {bad_member}"]
            names = set(archive.namelist())
            if not names:
                return ["archive is empty"]
            wrong_root = sorted(name for name in names if not name.startswith(PREFIX))
            if wrong_root:
                failures.append(f"paths outside {PREFIX} ({len(wrong_root)}), e.g. {wrong_root[0]}")
            forbidden = sorted(name for name in names if _is_forbidden(name))
            if forbidden:
                failures.append(f"unwanted cache, Git, or nested archive paths ({len(forbidden)}), e.g. {forbidden[0]}")
            missing = [PREFIX + path for path in REQUIRED_PATHS if PREFIX + path not in names]
            if missing:
                failures.append("missing required files: " + ", ".join(missing))

            expected_models = {PREFIX + "models/" + name for name in (
                "production_decision_tree_11_features.pkl",
                "production_random_forest_11_features.pkl",
                "production_logistic_regression_11_features.pkl",
                "production_mlp_11_features.pkl",
            )}
            actual_models = {name for name in names if name.startswith(PREFIX + "models/") and name.endswith(".pkl")}
            if actual_models != expected_models:
                failures.append("models directory must contain exactly the four production classifiers")

            dataset_path = PREFIX + "dataset/optimization_dataset.csv"
            if dataset_path in names:
                raw = archive.read(dataset_path).decode("utf-8-sig")
                reader = csv.DictReader(io.StringIO(raw))
                rows = list(reader)
                if len(rows) != 400:
                    failures.append(f"dataset row count is {len(rows)}, expected 400")
                fields = reader.fieldnames or []
                missing_features = [name for name in DATASET_FEATURES if name not in fields]
                if missing_features:
                    failures.append("dataset is missing model features: " + ", ".join(missing_features))
                ids = [row.get("program_id") for row in rows]
                if any(not value for value in ids):
                    failures.append("dataset contains a row without program_id")
                if len(ids) != len(set(ids)):
                    failures.append("dataset program_id values are not unique")

            holdout_path = PREFIX + "outputs/evaluation/holdout_metrics.csv"
            if holdout_path in names:
                raw = archive.read(holdout_path).decode("utf-8-sig")
                schemas = {row.get("feature_schema") for row in csv.DictReader(io.StringIO(raw))}
                if schemas != {"8_features", "11_features"}:
                    failures.append(f"unexpected holdout feature schemas: {sorted(str(x) for x in schemas)}")

            ablation_path = PREFIX + "outputs/evaluation/feature_ablation.csv"
            if ablation_path in names:
                raw = archive.read(ablation_path).decode("utf-8-sig")
                schemas = {row.get("schema") for row in csv.DictReader(io.StringIO(raw))}
                if schemas != {"original_8_features", "extended_11_features"}:
                    failures.append(f"unexpected ablation schemas: {sorted(str(x) for x in schemas)}")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as error:
        failures.append(f"could not inspect archive: {error}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Git-ready project ZIP.")
    parser.add_argument("zip_path", nargs="?", type=Path, default=DEFAULT_ZIP)
    args = parser.parse_args()
    failures = verify(args.zip_path)
    if failures:
        print(f"FAILED: {args.zip_path}")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print(f"OK: {args.zip_path}")
    print("  archive integrity, clean paths, runtime models, dataset, and evaluation artifacts verified")


if __name__ == "__main__":
    main()
