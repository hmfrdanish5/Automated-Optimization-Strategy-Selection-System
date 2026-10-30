"""Build the frozen optimization dataset with core and pass-aware features."""

from __future__ import annotations

import csv
from pathlib import Path

from dataset.diversity import diversity_report
from dataset.examples import load_sample_programs
from dataset.generator import (
    FINAL_GENERATED_COUNT,
    GENERATION_SEED,
    generate_final_programs,
    generate_extension_programs,
)
from dataset.labeler import diagnostic_gain_columns, label_row
from features.extended_extractor import FEATURE_COLUMNS_EXTENDED, extract_extended_features
from features.extractor import FEATURE_COLUMNS
from selection.model_config import OPTIMIZATION_DATASET_PATH, EXTENSION_DATASET_SEED, EXTENSION_PROGRAM_COUNT
from tac.program import TACProgram


def dataset_columns() -> list[str]:
    return [
        "program_id",
        *FEATURE_COLUMNS_EXTENDED,
        *diagnostic_gain_columns(),
        "best_strategy",
        "best_primary_gain",
        "best_secondary_gain",
    ]


def label_row_extended(program_id: str, program: TACProgram) -> dict[str, object]:
    row = label_row(program_id, program)
    row.update(extract_extended_features(program))
    return row


def build_dataset() -> list[dict[str, object]]:
    """400 deterministic programs: 200 core programs and 200 additional programs."""
    rows: list[dict[str, object]] = []
    for program_id, program in load_sample_programs():
        rows.append(label_row_extended(program_id, program))
    for program_id, program, _category in generate_final_programs(FINAL_GENERATED_COUNT, GENERATION_SEED):
        rows.append(label_row_extended(program_id, program))
    for program_id, program, _category in generate_extension_programs(EXTENSION_PROGRAM_COUNT, EXTENSION_DATASET_SEED):
        rows.append(label_row_extended(program_id, program))
    return rows


def write_dataset(path: Path | None = None) -> list[dict[str, object]]:
    rows = build_dataset()
    destination = path or OPTIMIZATION_DATASET_PATH
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset_columns(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def audit_dataset(rows: list[dict[str, object]]) -> dict[str, object]:
    report_8_features = diversity_report(
        [{**row, **{column: row[column] for column in FEATURE_COLUMNS}} for row in rows]
    )
    report_11_features = diversity_report(rows, feature_columns=FEATURE_COLUMNS_EXTENDED)
    return {
        "total_rows": len(rows),
        "seed_final_200": GENERATION_SEED,
        "seed_extension_200": EXTENSION_DATASET_SEED,
        "unique_feature_vectors_8_features": report_8_features.unique_feature_vectors,
        "unique_feature_vectors_11_features": report_11_features.unique_feature_vectors,
        "conflicting_signatures_8_features": len(report_8_features.inconsistent_signatures),
        "conflicting_signatures_11_features": len(report_11_features.inconsistent_signatures),
        "class_distribution": dict(sorted(report_11_features.class_counts.items())),
        "feature_ranges_11_features": report_11_features.ranges,
    }
