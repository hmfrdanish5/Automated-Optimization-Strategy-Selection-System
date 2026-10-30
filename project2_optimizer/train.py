"""Train the decision tree and random forest, then print both evaluations.

Run this from the project2_optimizer directory:

    python train.py

The saved models are the strict feature-grouped fit. dataset.csv is left
as the 57-row Review-1 set. The expanded rows are written to
dataset/dataset_expanded.csv.
"""

from __future__ import annotations

from dataset.dataset_builder import write_expanded_dataset
from selection.config import STRICT_SPLIT
from selection.pipeline import (
    format_expanded_report,
    format_prediction_demo,
    load_expanded_examples,
    train_expanded_evaluation,
)


def main() -> None:
    write_expanded_dataset()
    examples = load_expanded_examples()
    results = train_expanded_evaluation(examples)
    print(format_expanded_report(results))
    print()
    print(format_prediction_demo(results[STRICT_SPLIT]))


if __name__ == "__main__":
    main()
