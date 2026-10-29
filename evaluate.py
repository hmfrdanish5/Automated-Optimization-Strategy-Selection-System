"""Run the final dataset evaluation and write outputs/.

From the project2_optimizer directory:

    python evaluate.py

This trains holdout models for the reported scores, then trains separate
production models on the full final dataset. It does not train logistic
regression or a neural network.
"""

from __future__ import annotations

from evaluation.final_experiment import run_final_evaluation


if __name__ == "__main__":
    run_final_evaluation()
