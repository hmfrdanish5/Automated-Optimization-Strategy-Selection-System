"""Search controls, fair rule baseline, required plots and unseen benchmark.

Uses the saved production models for the unseen benchmark; run
`python train_models.py` first if models/ is empty. Writes outputs/search/.
"""

from __future__ import annotations

from evaluation.search_experiment import run_search_evaluation


def main() -> None:
    print(run_search_evaluation())


if __name__ == "__main__":
    main()
