"""Evaluate the selectors using holdout, grouped cross-validation, and feature ablation."""

from __future__ import annotations

from evaluation.experiment import run_model_evaluation


def main() -> None:
    print(run_model_evaluation())


if __name__ == "__main__":
    main()
