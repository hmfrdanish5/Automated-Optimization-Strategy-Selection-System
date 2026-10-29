"""Build dataset_v2.csv without touching Review-1 CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

from dataset.diversity import diversity_report
from dataset.examples import load_sample_programs
from dataset.generator import (
    FINAL_GENERATED_COUNT,
    GENERATION_SEED,
    generate_final_programs,
    generate_v2_extension_programs,
)
from dataset.labeler import diagnostic_gain_columns, label_row
from features.extended_extractor import FEATURE_COLUMNS_V2, extract_features_v2
from features.extractor import FEATURE_COLUMNS
from selection.review2_config import V2_DATASET_PATH, V2_DATASET_SEED, V2_EXTENSION_COUNT
from tac.program import TACProgram


def v2_dataset_columns() -> list[str]:
    return [
        "program_id",
        *FEATURE_COLUMNS_V2,
        *diagnostic_gain_columns(),
        "best_strategy",
        "best_primary_gain",
        "best_secondary_gain",
    ]


def label_row_v2(program_id: str, program: TACProgram) -> dict[str, object]:
    row = label_row(program_id, program)
    row.update(extract_features_v2(program))
    return row


def build_dataset_v2() -> list[dict[str, object]]:
    """200 programs from the final builder plus 200 extension programs."""
    rows: list[dict[str, object]] = []
    for program_id, program in load_sample_programs():
        rows.append(label_row_v2(program_id, program))
    for program_id, program, _category in generate_final_programs(FINAL_GENERATED_COUNT, GENERATION_SEED):
        rows.append(label_row_v2(program_id, program))
    for program_id, program, _category in generate_v2_extension_programs(V2_EXTENSION_COUNT, V2_DATASET_SEED):
        rows.append(label_row_v2(program_id, program))
    return rows


def write_dataset_v2(path: Path | None = None) -> list[dict[str, object]]:
    rows = build_dataset_v2()
    destination = path or V2_DATASET_PATH
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=v2_dataset_columns(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def audit_dataset_v2(rows: list[dict[str, object]]) -> dict[str, object]:
    report_v1 = diversity_report(
        [{**row, **{column: row[column] for column in FEATURE_COLUMNS}} for row in rows]
    )
    report_v2 = diversity_report(rows, feature_columns=FEATURE_COLUMNS_V2)
    return {
        "total_rows": len(rows),
        "seed_final_200": GENERATION_SEED,
        "seed_extension_200": V2_DATASET_SEED,
        "unique_feature_vectors_v1": report_v1.unique_feature_vectors,
        "unique_feature_vectors_v2": report_v2.unique_feature_vectors,
        "conflicting_signatures_v1": len(report_v1.inconsistent_signatures),
        "conflicting_signatures_v2": len(report_v2.inconsistent_signatures),
        "class_distribution": dict(sorted(report_v2.class_counts.items())),
        "feature_ranges_v2": report_v2.ranges,
    }
