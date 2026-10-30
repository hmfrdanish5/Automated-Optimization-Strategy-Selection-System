"""Optional engine controls: early stop on `no_improvement`, per-round budget, full feature records."""

from __future__ import annotations

import unittest

from engine.loop import (
    STOP_BUDGET_EXHAUSTED,
    STOP_NO_BENEFIT,
    STOP_PREDICTED_NO_IMPROVEMENT,
    optimize_program,
)
from engine.report import format_run
from engine.selectors import RankingSelector, RuleSelector, RuleSelectorV2
from features.extended_extractor import FEATURE_COLUMNS_EXTENDED as FEATURE_COLUMNS_V2
from tac.parser import parse_tac

CLEAN = "t1 = a + b\nprint t1\n"
DEAD_ONLY = "t1 = a + b\nspare = 4\nprint t1\n"


def _ranking(**scores):
    base = {
        "constant_folding": 0.0,
        "constant_propagation": 0.0,
        "algebraic_simplification": 0.0,
        "common_subexpression_elimination": 0.0,
        "dead_code_elimination": 0.0,
        "strength_reduction": 0.0,
        "no_improvement": 0.0,
    }
    base.update(scores)
    return RankingSelector("test_ranking", base)


class EarlyStopTests(unittest.TestCase):
    def test_default_runs_the_final_sweep(self) -> None:
        run = optimize_program(parse_tac(CLEAN), _ranking(no_improvement=0.95))
        self.assertEqual(run.stopped_because, STOP_NO_BENEFIT)
        self.assertEqual(run.optimizer_evaluations_total, 6)

    def test_confident_no_improvement_stops_without_trying(self) -> None:
        run = optimize_program(parse_tac(CLEAN), _ranking(no_improvement=0.95), stop_on_no_improvement=0.9)
        self.assertEqual(run.stopped_because, STOP_PREDICTED_NO_IMPROVEMENT)
        self.assertEqual(run.optimizer_evaluations_total, 0)
        self.assertEqual(run.history[-1].attempts, ())
        self.assertIn("No optimizer tried", format_run(run))

    def test_low_confidence_does_not_stop(self) -> None:
        run = optimize_program(parse_tac(CLEAN), _ranking(no_improvement=0.6), stop_on_no_improvement=0.9)
        self.assertEqual(run.stopped_because, STOP_NO_BENEFIT)
        self.assertEqual(run.optimizer_evaluations_total, 6)

    def test_rule_selector_stops_when_it_sees_no_opportunity(self) -> None:
        run = optimize_program(parse_tac(CLEAN), RuleSelector(), stop_on_no_improvement=0.5)
        self.assertEqual(run.stopped_because, STOP_PREDICTED_NO_IMPROVEMENT)

    def test_early_stop_keeps_the_gain_on_a_dead_store(self) -> None:
        full = optimize_program(parse_tac(DEAD_ONLY), RuleSelectorV2())
        early = optimize_program(parse_tac(DEAD_ONLY), RuleSelectorV2(), stop_on_no_improvement=0.5)
        self.assertEqual(early.total_instruction_gain, full.total_instruction_gain)
        self.assertLess(early.optimizer_evaluations_total, full.optimizer_evaluations_total)

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            optimize_program(parse_tac(CLEAN), RuleSelector(), stop_on_no_improvement=1.5)


class BudgetTests(unittest.TestCase):
    def test_top_one_budget_tries_only_the_first_candidate(self) -> None:
        # Constant folding is ranked first but cannot help; the dead store needs DCE.
        selector = _ranking(constant_folding=0.9, dead_code_elimination=0.1)
        run = optimize_program(parse_tac(DEAD_ONLY), selector, max_attempts_per_iteration=1)
        self.assertEqual(run.optimizer_evaluations_total, 1)
        self.assertEqual(run.total_instruction_gain, 0)
        self.assertEqual(run.stopped_because, STOP_BUDGET_EXHAUSTED)

    def test_budget_of_two_reaches_the_fix(self) -> None:
        selector = _ranking(constant_folding=0.9, dead_code_elimination=0.1)
        run = optimize_program(parse_tac(DEAD_ONLY), selector, max_attempts_per_iteration=2)
        self.assertEqual(run.total_instruction_gain, 1)

    def test_invalid_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            optimize_program(parse_tac(CLEAN), RuleSelector(), max_attempts_per_iteration=0)


class FeatureRecordTests(unittest.TestCase):
    def test_eleven_feature_selector_records_every_input(self) -> None:
        program = parse_tac("i = 0\nL1:\nt1 = i * 2\nprint t1\ni = i + 1\nif i < 4 goto L1\nprint i\n")
        run = optimize_program(program, RuleSelectorV2())
        self.assertEqual(list(run.history[0].features), list(FEATURE_COLUMNS_V2))
        self.assertIn("num_mul_by_two_in_loops", format_run(run))


if __name__ == "__main__":
    unittest.main()
