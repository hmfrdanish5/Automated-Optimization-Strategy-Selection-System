"""Run Review-2 holdout, CV, and ablation experiments."""

from __future__ import annotations

from evaluation.review2_experiment import run_review2_evaluation


def main() -> None:
    print(run_review2_evaluation())


if __name__ == "__main__":
    main()
