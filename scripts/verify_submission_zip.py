"""Verify submission zip contents before hand-in."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_ZIP = Path(__file__).resolve().parents[2] / "project2_optimizer_submission.zip"
ARCHIVE_PREFIX = "project2_optimizer/"

REQUIRED_PATHS = (
    f"{ARCHIVE_PREFIX}dataset/dataset_v2.csv",
    f"{ARCHIVE_PREFIX}dataset/dataset_final.csv",
    f"{ARCHIVE_PREFIX}dataset/dataset.csv",
    f"{ARCHIVE_PREFIX}models/production_decision_tree.pkl",
    f"{ARCHIVE_PREFIX}models/production_random_forest.pkl",
    f"{ARCHIVE_PREFIX}models/review2_production_decision_tree_v2.pkl",
    f"{ARCHIVE_PREFIX}models/review2_production_random_forest_v2.pkl",
    f"{ARCHIVE_PREFIX}models/review2_production_logistic_regression_v2.pkl",
    f"{ARCHIVE_PREFIX}models/review2_production_mlp_v2.pkl",
    f"{ARCHIVE_PREFIX}outputs/review2/feature_ablation.csv",
    f"{ARCHIVE_PREFIX}outputs/review2/experiment_summary.txt",
    f"{ARCHIVE_PREFIX}REVIEW_2_FINAL_REPORT.md",
    f"{ARCHIVE_PREFIX}REVIEW_1_VS_REVIEW_2.md",
    f"{ARCHIVE_PREFIX}requirements.txt",
    f"{ARCHIVE_PREFIX}tests/test_semantic_fuzz.py",
    f"{ARCHIVE_PREFIX}tests/test_semantics.py",
    f"{ARCHIVE_PREFIX}tests/test_review2.py",
    f"{ARCHIVE_PREFIX}dataset/v2_loader.py",
    f"{ARCHIVE_PREFIX}package_submission.py",
)

EXPECTED_V2_ROWS = 400


def _is_forbidden(name: str) -> bool:
    if "Automated-Optimizer-Selection-System" in name:
        return True
    if "__pycache__" in name:
        return True
    if ".git/" in name or name.endswith("/.git"):
        return True
    if name.endswith(".pyc") or name.endswith(".pyo"):
        return True
    if name.endswith("project2_optimizer_submission.zip"):
        return True
    return False


def verify(zip_path: Path) -> list[str]:
    failures: list[str] = []
    if not zip_path.is_file():
        return [f"archive not found: {zip_path}"]

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        forbidden = sorted(name for name in names if _is_forbidden(name))
        if forbidden:
            failures.append(f"forbidden paths ({len(forbidden)}), e.g. {forbidden[0]}")

        missing = [path for path in REQUIRED_PATHS if path not in names]
        if missing:
            failures.append(f"missing required paths ({len(missing)}): " + ", ".join(missing[:5]))
            if len(missing) > 5:
                failures[-1] += ", ..."

        summary_path = f"{ARCHIVE_PREFIX}outputs/review2/experiment_summary.txt"
        ablation_path = f"{ARCHIVE_PREFIX}outputs/review2/feature_ablation.csv"
        dataset_path = f"{ARCHIVE_PREFIX}dataset/dataset_v2.csv"

        if dataset_path in names:
            rows = list(
                csv.DictReader(io.StringIO(archive.read(dataset_path).decode("utf-8")))
            )
            if len(rows) != EXPECTED_V2_ROWS:
                failures.append(
                    f"dataset_v2.csv row count {len(rows)}, expected {EXPECTED_V2_ROWS}"
                )
        elif dataset_path not in missing:
            failures.append("dataset_v2.csv present in listing but unreadable")

        if summary_path in names and ablation_path in names:
            summary = archive.read(summary_path).decode("utf-8")
            abl = pd.read_csv(io.BytesIO(archive.read(ablation_path)))
            mismatches = []
            for _, row in abl.iterrows():
                line = f"{row['model']} ({row['schema']}): {row['macro_f1']:.3f}"
                if line not in summary:
                    mismatches.append(line)
            if mismatches:
                failures.append(
                    f"ablation vs in-archive summary mismatches ({len(mismatches)}): "
                    + "; ".join(mismatches[:3])
                )
        elif summary_path not in names or ablation_path not in names:
            if not missing:
                failures.append("cannot compare ablation to summary (files missing from archive)")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify project2_optimizer submission zip.")
    parser.add_argument(
        "zip_path",
        nargs="?",
        type=Path,
        default=DEFAULT_ZIP,
        help=f"Path to submission zip (default: {DEFAULT_ZIP})",
    )
    args = parser.parse_args()
    failures = verify(args.zip_path)
    if failures:
        print(f"FAILED: {args.zip_path}")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print(f"OK: {args.zip_path}")
    print(f"  entries checked; required artifacts, {EXPECTED_V2_ROWS} dataset rows, ablation summary consistent")


if __name__ == "__main__":
    main()
