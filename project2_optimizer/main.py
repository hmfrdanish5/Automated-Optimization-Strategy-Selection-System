"""Show features and independent optimizer results for the hand-written samples.

Run this from the project2_optimizer directory:

    python main.py

Each optimizer is applied to the original program. The passes are not chained.
"""

from __future__ import annotations

from dataset.dataset_builder import (
    distribution_report,
    format_gain_table,
    format_label_details,
    write_dataset,
)
from dataset.examples import load_sample_programs
from dataset.generator import GENERATION_SEED, generate_programs
from dataset.labeler import label_row
from evaluation.metrics import instruction_gain
from features.extractor import FEATURE_COLUMNS, extract_features
from optimizers.registry import STRATEGY_NAMES, apply_strategy


def main() -> None:
    programs = load_sample_programs()
    for program_id, program in programs:
        features = extract_features(program)
        print(f"=== {program_id} ===")
        print(program.to_tac())
        print("features:")
        for name in FEATURE_COLUMNS:
            value = features[name]
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")
        print("optimizers, each starting from the original TAC:")
        for strategy in STRATEGY_NAMES:
            optimized = apply_strategy(program, strategy)
            gain = instruction_gain(program, optimized)
            print(f"  {strategy}: gain {gain}")
            if optimized.to_tac() != program.to_tac():
                for line in optimized.to_tac().splitlines():
                    print(f"    {line}")
        print()
    print_dataset_report()


def print_dataset_report() -> None:
    """Label the review dataset and print the gain table and class counts."""
    rows = write_dataset()
    handwritten_ids = {program_id for program_id, _program in load_sample_programs()}
    handwritten = [row for row in rows if row["program_id"] in handwritten_ids]
    print("Hand-written samples, gains as instruction/weighted/operation/temporary/code-size:")
    print(format_gain_table(handwritten))
    print()
    for row in handwritten:
        print(format_label_details(row))
        print()
    print(distribution_report(rows))
    print()
    print(f"Generated examples (seed {GENERATION_SEED}):")
    for program_id, program, category in generate_programs(count=4, seed=GENERATION_SEED):
        row = label_row(program_id, program)
        print(f"--- {program_id} ({category}) -> {row['best_strategy']} ---")
        print(program.to_tac())
        print()


if __name__ == "__main__":
    main()
