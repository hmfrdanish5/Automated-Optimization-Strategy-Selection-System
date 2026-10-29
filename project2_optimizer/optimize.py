"""Run the iterative engine on hand-written TAC programs.

From the project2_optimizer directory:

    python optimize.py

Loads saved production models (Review-2 when present). This script does not train.
"""

from __future__ import annotations

from dataset.examples import load_sample_programs
from engine.loop import optimize_program
from engine.report import format_comparison, format_run
from engine.selectors import RankingSelector, runtime_selectors
from tac.parser import parse_tac

DEMO_IDS = (
    "constant_heavy",
    "algebraic_heavy",
    "cse_heavy",
    "dead_code_heavy",
    "mixed",
)

SELECTOR_KEYS = (
    "rule_based",
    "decision_tree",
    "random_forest",
    "logistic_regression",
    "mlp",
)

FALLBACK_TAC = """t1 = left + right
spare = 4
print t1
"""


def main() -> None:
    programs = dict(load_sample_programs())
    originals = {program_id: programs[program_id].to_tac() for program_id in DEMO_IDS}
    selectors = runtime_selectors()
    runs = []
    for program_id in DEMO_IDS:
        for name in SELECTOR_KEYS:
            if name not in selectors:
                raise RuntimeError(f"selector {name} is not available; run train_review2.py first")
            runs.append(optimize_program(programs[program_id], selectors[name], program_id))
    print(format_comparison(runs))
    print()
    showcase = next(
        run
        for run in runs
        if run.program_id == "mixed" and run.selector_type == "random_forest"
    )
    print(format_run(showcase))
    print()
    print(format_run(_fallback_run()))
    print()
    print("Original TAC unchanged:")
    for program_id in DEMO_IDS:
        same = programs[program_id].to_tac() == originals[program_id]
        print(f"    {program_id}: {same}")


def _fallback_run():
    """Constant folding is ranked first and does nothing. Dead code is second."""
    probabilities = {
        "constant_folding": 0.70,
        "constant_propagation": 0.02,
        "algebraic_simplification": 0.02,
        "common_subexpression_elimination": 0.04,
        "dead_code_elimination": 0.20,
        "strength_reduction": 0.01,
        "no_improvement": 0.01,
    }
    selector = RankingSelector("fallback_ranking", probabilities)
    return optimize_program(parse_tac(FALLBACK_TAC), selector, "fallback_dead_code")


if __name__ == "__main__":
    main()
