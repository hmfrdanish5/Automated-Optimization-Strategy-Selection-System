"""Build the Review 1 CSV from hand-written samples plus generated programs."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

from dataset.examples import load_sample_programs
from dataset.generator import (
    EXPANDED_GENERATED_COUNT,
    FINAL_GENERATED_COUNT,
    GENERATED_PROGRAM_COUNT,
    GENERATION_SEED,
    generate_diverse_programs,
    generate_final_programs,
    generate_programs,
)
from dataset.labeler import DATASET_COLUMNS, label_row
from evaluation.label_policy import NO_IMPROVEMENT, STRATEGY_PRIORITY


DATASET_PATH = Path(__file__).resolve().parent / "dataset.csv"
EXPANDED_DATASET_PATH = Path(__file__).resolve().parent / "dataset_expanded.csv"
FINAL_DATASET_PATH = Path(__file__).resolve().parent / "dataset_final.csv"


def build_dataset(
    generated_count: int = GENERATED_PROGRAM_COUNT,
    seed: int = GENERATION_SEED,
) -> list[dict[str, object]]:
    """Label the hand-written samples, then `generated_count` generated programs.

    Rows are ordered hand-written first, then `gen_001`, `gen_002`, and so on.
    """
    rows = [label_row(program_id, program) for program_id, program in load_sample_programs()]
    for program_id, program, _category in generate_programs(generated_count, seed):
        rows.append(label_row(program_id, program))
    return rows


def write_dataset(
    path: Path | None = None,
    generated_count: int = GENERATED_PROGRAM_COUNT,
    seed: int = GENERATION_SEED,
) -> list[dict[str, object]]:
    """Write the dataset CSV and return the same rows."""
    rows = build_dataset(generated_count, seed)
    destination = path or DATASET_PATH
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_expanded_dataset(
    generated_count: int = EXPANDED_GENERATED_COUNT,
    seed: int = GENERATION_SEED,
) -> list[dict[str, object]]:
    """Label the hand-written samples and the wider generated set.

    This does not replace dataset.csv. Rows are hand-written first, then
    `exp_001`, `exp_002`, and so on.
    """
    rows = [label_row(program_id, program) for program_id, program in load_sample_programs()]
    for program_id, program, _category in generate_diverse_programs(generated_count, seed):
        rows.append(label_row(program_id, program))
    return rows


def write_expanded_dataset(
    path: Path | None = None,
    generated_count: int = EXPANDED_GENERATED_COUNT,
    seed: int = GENERATION_SEED,
) -> list[dict[str, object]]:
    """Write the expanded CSV and return the same rows."""
    rows = build_expanded_dataset(generated_count, seed)
    destination = path or EXPANDED_DATASET_PATH
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_final_dataset(
    generated_count: int = FINAL_GENERATED_COUNT,
    seed: int = GENERATION_SEED,
) -> list[dict[str, object]]:
    """Label the hand-written samples and the final generated set.

    This does not replace dataset.csv or dataset_expanded.csv.
    """
    rows = [label_row(program_id, program) for program_id, program in load_sample_programs()]
    for program_id, program, _category in generate_final_programs(generated_count, seed):
        rows.append(label_row(program_id, program))
    return rows


def write_final_dataset(
    path: Path | None = None,
    generated_count: int = FINAL_GENERATED_COUNT,
    seed: int = GENERATION_SEED,
) -> list[dict[str, object]]:
    """Write dataset_final.csv and return the same rows."""
    rows = build_final_dataset(generated_count, seed)
    destination = path or FINAL_DATASET_PATH
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def distribution_report(rows: list[dict[str, object]]) -> str:
    """Count labels and warn when a class is missing or larger than half the data."""
    counts: Counter[str] = Counter(str(row["best_strategy"]) for row in rows)
    total = len(rows)
    lines = [f"total samples: {total}"]
    for strategy in (*STRATEGY_PRIORITY, NO_IMPROVEMENT):
        lines.append(f"  {strategy}: {counts.get(strategy, 0)}")
    class_counts = [counts.get(strategy, 0) for strategy in (*STRATEGY_PRIORITY, NO_IMPROVEMENT)]
    lines.append(f"minimum class count: {min(class_counts) if class_counts else 0}")
    lines.append(f"maximum class count: {max(class_counts) if class_counts else 0}")
    unexpected = sorted(set(counts) - set(STRATEGY_PRIORITY) - {NO_IMPROVEMENT})
    for name in unexpected:
        lines.append(f"  {name}: {counts[name]}")

    warnings: list[str] = []
    for strategy in STRATEGY_PRIORITY:
        if counts.get(strategy, 0) == 0:
            warnings.append(f"WARNING: {strategy} has no samples.")
    if counts.get(NO_IMPROVEMENT, 0) == 0:
        warnings.append("WARNING: no_improvement has no samples.")
    for name, count in sorted(counts.items()):
        if total and count / total > 0.5:
            warnings.append(
                f"WARNING: {name} is {count} of {total} samples "
                f"({count / total:.0%}), more than half the dataset."
            )
    if not warnings:
        warnings.append("No class is missing, and none is larger than half the dataset.")
    return "\n".join(lines + [""] + warnings)


_STRATEGY_TITLES = (
    ("cf", "Constant Folding"),
    ("cp", "Constant Propagation"),
    ("alg", "Algebraic Simplification"),
    ("cse", "Common Subexpression Elimination"),
    ("dce", "Dead Code Elimination"),
    ("sr", "Strength Reduction"),
)


def format_gain_table(rows: list[dict[str, object]]) -> str:
    """One row per program: five gains per strategy, then the chosen label.

    Each cell is instruction/weighted-cost/operation/temporary/code-size gain.
    """
    headers = ("program", "CF", "CP", "ALG", "CSE", "DCE", "SR", "best_strategy")
    prefixes = ("cf", "cp", "alg", "cse", "dce", "sr")
    table = ["  ".join(f"{header:<18}" for header in headers)]
    for row in rows:
        cells = [f"{str(row['program_id']):<18}"]
        for prefix in prefixes:
            cell = (
                f"{row[prefix + '_instruction_gain']}/"
                f"{row[prefix + '_weighted_cost_gain']}/"
                f"{row[prefix + '_operation_gain']}/"
                f"{row[prefix + '_temp_gain']}/"
                f"{row[prefix + '_code_size_gain']}"
            )
            cells.append(f"{cell:<18}")
        cells.append(str(row["best_strategy"]))
        table.append("  ".join(cells))
    return "\n".join(table)


def format_label_details(row: dict[str, object]) -> str:
    """Spell out every strategy's gains for one program."""
    lines = [f"Program: {row['program_id']}", ""]
    for prefix, title in _STRATEGY_TITLES:
        lines.append(f"{title}:")
        lines.append(f"    instruction gain = {row[prefix + '_instruction_gain']}")
        lines.append(f"    weighted cost gain = {row[prefix + '_weighted_cost_gain']}")
        lines.append(f"    operation gain = {row[prefix + '_operation_gain']}")
        lines.append(f"    temp gain = {row[prefix + '_temp_gain']}")
        lines.append(f"    code size gain = {row[prefix + '_code_size_gain']}")
        lines.append("")
    lines.append(f"BEST: {row['best_strategy']}")
    return "\n".join(lines)
