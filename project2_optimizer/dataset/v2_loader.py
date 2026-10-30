"""Load persisted dataset_v2.csv and pair rows with reconstructed TAC programs."""

from __future__ import annotations

import csv
from pathlib import Path

from dataset.examples import load_sample_programs
from dataset.generator import (
    FINAL_GENERATED_COUNT,
    GENERATION_SEED,
    generate_final_programs,
    generate_v2_extension_programs,
)
from dataset.labeler import diagnostic_gain_columns
from dataset.v2_builder import label_row_v2, v2_dataset_columns
from features.extended_extractor import FEATURE_COLUMNS_V2
from features.extractor import FEATURE_COLUMNS
from selection.review2_config import V2_DATASET_PATH, V2_DATASET_SEED, V2_EXTENSION_COUNT
from selection.splitting import LabeledExample
from tac.program import TACProgram


class DatasetV2MismatchError(ValueError):
    """Saved CSV row does not match a reconstructed program."""


class DatasetV2FormatError(ValueError):
    """Saved CSV row has an invalid numeric or gain field."""


def load_dataset_v2_rows(path: Path | None = None) -> list[dict[str, object]]:
    """Read dataset_v2.csv. Does not build or overwrite the file."""
    destination = path or V2_DATASET_PATH
    if not destination.is_file():
        raise FileNotFoundError(
            f"{destination} is missing. Run python train_review2.py to generate it explicitly."
        )
    expected = v2_dataset_columns()
    with destination.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise ValueError(
                f"{destination.name} columns do not match v2_dataset_columns(); "
                f"expected {len(expected)} columns, got {reader.fieldnames}"
            )
        rows = [dict(row) for row in reader]
    for row in rows:
        _coerce_row_types(row)
    return rows


def pair_v2_examples(
    rows: list[dict[str, object]],
    feature_columns: tuple[str, ...] | list[str],
    *,
    validate: bool = True,
) -> list[LabeledExample]:
    """Pair CSV rows with programs rebuilt from program_id (deterministic seeds)."""
    programs = programs_for_dataset_v2()
    program_ids = [program_id for program_id, _program in programs]
    csv_ids = [str(row["program_id"]) for row in rows]
    if program_ids != csv_ids:
        raise ValueError(
            "dataset_v2 program_id order does not match the deterministic generator order. "
            f"First mismatch index "
            f"{next((index for index, (left, right) in enumerate(zip(program_ids, csv_ids)) if left != right), len(csv_ids))}."
        )
    columns = list(feature_columns)
    examples: list[LabeledExample] = []
    for row, (program_id, program) in zip(rows, programs):
        if validate:
            validate_row_against_program(row, program)
        examples.append(
            LabeledExample(
                program_id=program_id,
                program=program,
                features={column: row[column] for column in columns},
                label=str(row["best_strategy"]),
            )
        )
    return examples


def programs_for_dataset_v2() -> list[tuple[str, TACProgram]]:
    """Same program list and order as dataset_v2 generation."""
    programs = [(program_id, program) for program_id, program in load_sample_programs()]
    programs.extend(
        (program_id, program)
        for program_id, program, _category in generate_final_programs(FINAL_GENERATED_COUNT, GENERATION_SEED)
    )
    programs.extend(
        (program_id, program)
        for program_id, program, _category in generate_v2_extension_programs(
            V2_EXTENSION_COUNT,
            V2_DATASET_SEED,
        )
    )
    return programs


def validate_row_against_program(row: dict[str, object], program: TACProgram) -> None:
    """Ensure CSV features, labels, gains, and summary gains match fresh measurement."""
    program_id = str(row["program_id"])
    expected = label_row_v2(program_id, program)
    for column in v2_dataset_columns():
        if column == "program_id":
            continue
        saved = row[column]
        measured = expected[column]
        if column == "const_ratio":
            if not _feature_values_equal(column, saved, float(measured)):
                raise DatasetV2MismatchError(
                    f"{program_id}: feature {column} in CSV is {saved} "
                    f"but reconstructed program gives {measured}"
                )
        elif column in FEATURE_COLUMNS_V2 or column in (
            "best_primary_gain",
            "best_secondary_gain",
        ) or column in diagnostic_gain_columns():
            if int(saved) != int(measured):
                raise DatasetV2MismatchError(
                    f"{program_id}: {column} in CSV is {saved} but remeasured value is {measured}"
                )
        elif column == "best_strategy":
            if str(saved) != str(measured):
                raise DatasetV2MismatchError(
                    f"{program_id}: best_strategy in CSV is {saved} "
                    f"but remeasured label is {measured}"
                )


def _coerce_row_types(row: dict[str, object]) -> None:
    program_id = str(row.get("program_id", "?"))
    for column in FEATURE_COLUMNS:
        if column == "const_ratio":
            row[column] = float(row[column])
        else:
            row[column] = _require_int(row[column], column, program_id)
    for column in FEATURE_COLUMNS_V2[len(FEATURE_COLUMNS) :]:
        row[column] = _require_int(row[column], column, program_id)
    row["best_primary_gain"] = _require_int(row["best_primary_gain"], "best_primary_gain", program_id)
    row["best_secondary_gain"] = _require_int(
        row["best_secondary_gain"],
        "best_secondary_gain",
        program_id,
    )
    for column in diagnostic_gain_columns():
        row[column] = _require_int(row[column], column, program_id)


def _require_int(value: object, column: str, program_id: str) -> int:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError as error:
        raise DatasetV2FormatError(
            f"{program_id}: column {column} is not numeric ({value!r})"
        ) from error
    if number != int(number):
        raise DatasetV2FormatError(
            f"{program_id}: column {column} must be an integer, got {value!r}"
        )
    return int(number)


def _feature_values_equal(column: str, saved: object, computed: float) -> bool:
    if column == "const_ratio":
        return round(float(saved), 10) == round(float(computed), 10)
    return int(float(saved)) == int(computed)
