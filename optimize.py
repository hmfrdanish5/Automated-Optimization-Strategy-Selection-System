"""Run the iterative engine on hand-written TAC programs.

From the project2_optimizer directory:

    python optimize.py
    python optimize.py --early-stop 0.7
    python optimize.py --source examples/source_program.txt
    python optimize.py --source examples/source_program.txt --early-stop 0.7

``--source`` compiles one restricted C-like program with source_frontend and
optimizes the generated TAC with every selector, including Rule-Based v2.
``--early-stop`` stops a run when the selector predicts no_improvement with at
least that confidence.

Loads the saved production classifiers. This script does not train models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dataset.examples import load_sample_programs
from engine.loop import optimize_program
from engine.report import format_comparison, format_run
from engine.selectors import RankingSelector, RuleSelectorV2, runtime_selectors
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the iterative optimizer on sample or source programs.")
    parser.add_argument("--source", type=Path, help="restricted C-like source file to compile and optimize")
    parser.add_argument(
        "--early-stop",
        type=float,
        default=None,
        metavar="CONFIDENCE",
        help="stop when the selector predicts no_improvement with at least this confidence (0 to 1)",
    )
    args = parser.parse_args(argv)
    if args.early_stop is not None and not 0.0 <= args.early_stop <= 1.0:
        parser.error("--early-stop must be between 0 and 1")
    if args.source is not None:
        run_source(args.source, args.early_stop)
        return
    run_samples(args.early_stop)


def run_source(path: Path, early_stop: float | None = None) -> None:
    """Compile one source program, then optimize its TAC with every selector."""
    from source_frontend.compiler import compile_source

    result = compile_source(path.read_text(encoding="utf-8"))
    if not result.ok:
        for error in result.errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Generated TAC:")
    print(result.program.to_tac())
    print()
    selectors = {**runtime_selectors(), "rule_based_v2": RuleSelectorV2()}
    runs = [
        optimize_program(result.program, selectors[name], path.stem, stop_on_no_improvement=early_stop)
        for name in (*SELECTOR_KEYS, "rule_based_v2")
    ]
    print(format_comparison(runs))
    print()
    showcase = next(run for run in runs if run.selector_type == "random_forest")
    print(format_run(showcase))


def run_samples(early_stop: float | None = None) -> None:
    programs = dict(load_sample_programs())
    originals = {program_id: programs[program_id].to_tac() for program_id in DEMO_IDS}
    selectors = runtime_selectors()
    runs = []
    for program_id in DEMO_IDS:
        for name in SELECTOR_KEYS:
            if name not in selectors:
                raise RuntimeError(f"selector {name} is not available; run train_models.py first")
            runs.append(
                optimize_program(programs[program_id], selectors[name], program_id, stop_on_no_improvement=early_stop)
            )
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
