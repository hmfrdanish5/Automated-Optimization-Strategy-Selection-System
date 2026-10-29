"""Tests for the iterative optimization engine."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.loop import (
    MAX_ITERATIONS,
    STOP_ITERATION_LIMIT,
    STOP_NO_BENEFIT,
    STOP_REPEATED_STATE,
    optimize_program,
)
from engine.selectors import RankingSelector, RuleSelector, saved_selectors
from features.extractor import FEATURE_COLUMNS, extract_features
from tac.parser import parse_tac


DEAD_AND_FOLD = """t1 = 10 + 20
spare = 1
print t1
"""

FALLBACK_TAC = """t1 = left + right
spare = 4
print t1
"""

STRAIGHT = """t1 = a + b
t2 = t1 * c
print t2
"""


class EngineTests(unittest.TestCase):
    def test_rule_selector_accepts_a_real_improvement(self) -> None:
        program = parse_tac(DEAD_AND_FOLD)
        original = program.to_tac()
        run = optimize_program(program, RuleSelector(), "fold_and_dead")
        self.assertEqual(program.to_tac(), original)
        self.assertIn("constant_folding", run.strategy_sequence)
        self.assertIn("dead_code_elimination", run.strategy_sequence)
        self.assertGreater(run.total_instruction_gain, 0)
        self.assertGreater(run.total_weighted_cost_gain, 0)
        self.assertEqual(run.stopped_because, STOP_NO_BENEFIT)
        self.assertLessEqual(run.successful_iterations, MAX_ITERATIONS)

    def test_saved_models_run_and_repeat(self) -> None:
        program = parse_tac(DEAD_AND_FOLD)
        original = program.to_tac()
        selectors = saved_selectors()
        for name in ("decision_tree", "random_forest"):
            first = optimize_program(program, selectors[name], "models")
            second = optimize_program(program, selectors[name], "models")
            self.assertEqual(program.to_tac(), original)
            self.assertEqual(first.strategy_sequence, second.strategy_sequence)
            self.assertEqual(first.final_program.to_tac(), second.final_program.to_tac())
            self.assertGreaterEqual(first.successful_iterations, 1)
            probabilities = first.history[0].candidate_probabilities
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
            self.assertAlmostEqual(
                first.history[0].confidence,
                probabilities[first.history[0].selected_strategy],
            )

    def test_ranking_follows_probabilities_and_skips_a_useless_top_choice(self) -> None:
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
        features = extract_features(parse_tac(FALLBACK_TAC))
        advice = selector.advise(features)
        self.assertEqual(advice.ranking[0], "constant_folding")
        self.assertEqual(advice.try_order[0], "constant_folding")
        self.assertIn("dead_code_elimination", advice.try_order)
        self.assertNotIn("no_improvement", advice.try_order)
        self.assertAlmostEqual(advice.confidence, 0.70)

        program = parse_tac(FALLBACK_TAC)
        original = program.to_tac()
        run = optimize_program(program, selector, "fallback")
        self.assertEqual(program.to_tac(), original)
        first = run.history[0]
        self.assertEqual(first.attempts[0].strategy, "constant_folding")
        self.assertFalse(first.attempts[0].accepted)
        self.assertEqual(first.attempts[0].outcome, "unchanged TAC")
        accepted = next(attempt for attempt in first.attempts if attempt.accepted)
        self.assertEqual(accepted.strategy, "dead_code_elimination")
        self.assertEqual(run.strategy_sequence[0], "dead_code_elimination")
        self.assertGreater(run.total_instruction_gain, 0)
        self.assertGreater(run.candidate_attempts, 0)
        self.assertEqual(run.optimizer_evaluations_total, run.candidate_attempts)
        self.assertEqual(
            run.successful_candidate_attempts + run.failed_candidate_attempts,
            run.candidate_attempts,
        )
        self.assertEqual(run.successful_candidate_attempts, run.successful_iterations)

    def test_no_improvement_prediction_does_not_stop_the_search(self) -> None:
        probabilities = {name: 0.0 for name in (
            "constant_folding",
            "constant_propagation",
            "algebraic_simplification",
            "common_subexpression_elimination",
            "dead_code_elimination",
            "strength_reduction",
            "no_improvement",
        )}
        probabilities["no_improvement"] = 0.80
        probabilities["dead_code_elimination"] = 0.20
        program = parse_tac(FALLBACK_TAC)
        run = optimize_program(program, RankingSelector("none_first", probabilities), "none_first")
        self.assertEqual(run.history[0].selected_strategy, "no_improvement")
        self.assertEqual(run.strategy_sequence[0], "dead_code_elimination")

    def test_straight_line_converges_without_a_change(self) -> None:
        program = parse_tac(STRAIGHT)
        run = optimize_program(program, RuleSelector(), "straight")
        self.assertEqual(run.strategy_sequence, ())
        self.assertEqual(run.successful_iterations, 0)
        self.assertEqual(run.total_instruction_gain, 0)
        self.assertEqual(run.stopped_because, STOP_NO_BENEFIT)
        self.assertEqual(run.final_program.to_tac(), program.to_tac())

    def test_iteration_limit_stops_a_program_that_can_still_improve(self) -> None:
        program = parse_tac(DEAD_AND_FOLD)
        limited = optimize_program(program, RuleSelector(), "limited", max_iterations=1)
        full = optimize_program(program, RuleSelector(), "limited")
        self.assertEqual(limited.stopped_because, STOP_ITERATION_LIMIT)
        self.assertEqual(limited.successful_iterations, 1)
        self.assertGreater(full.successful_iterations, limited.successful_iterations)

    def test_repeated_state_is_rejected(self) -> None:
        program = parse_tac("t1 = 10 + 20\nprint t1\nprint extra")
        shorter = parse_tac("print t1")
        calls = {"n": 0}

        def apply(_current, _strategy):
            calls["n"] += 1
            if calls["n"] == 1:
                return shorter.copy()
            return program.copy()

        probabilities = {name: 0.0 for name in (
            "constant_folding",
            "constant_propagation",
            "algebraic_simplification",
            "common_subexpression_elimination",
            "dead_code_elimination",
            "strength_reduction",
            "no_improvement",
        )}
        probabilities["dead_code_elimination"] = 1.0
        with patch("engine.loop.improves_on_original", return_value=True):
            run = optimize_program(
                program,
                RankingSelector("repeat", probabilities),
                "repeat",
                apply=apply,
            )
        self.assertEqual(run.successful_iterations, 1)
        self.assertEqual(run.stopped_because, STOP_REPEATED_STATE)
        self.assertTrue(any(attempt.outcome == STOP_REPEATED_STATE for attempt in run.history[-1].attempts))

    def test_history_has_the_recorded_fields(self) -> None:
        program = parse_tac(FALLBACK_TAC)
        run = optimize_program(program, RuleSelector(), "history")
        row = run.history_rows()[0]
        expected = {
            "iteration",
            "features",
            "strategy_ranking",
            "candidate_probabilities",
            "selected_strategy",
            "confidence",
            "instruction_count_before",
            "instruction_count_after",
            "instruction_gain",
            "weighted_cost_gain",
            "operation_gain",
            "temporary_gain",
            "code_size_gain",
            "accepted",
            "attempts",
        }
        self.assertTrue(expected <= set(row))
        self.assertEqual(set(row["features"]), set(FEATURE_COLUMNS))
        summary = run.summary()
        self.assertEqual(summary["program_id"], "history")
        self.assertEqual(summary["selector_type"], "rule_based")

    def test_single_useful_optimization(self) -> None:
        program = parse_tac("spare = 1\nprint 2\n")
        original = program.to_tac()
        run = optimize_program(program, RuleSelector(), "one_dead")
        self.assertEqual(program.to_tac(), original)
        self.assertEqual(run.strategy_sequence, ("dead_code_elimination",))
        self.assertEqual(run.successful_iterations, 1)
        self.assertEqual(run.total_instruction_gain, 1)

    def test_top_ranked_optimizer_is_accepted_when_it_improves(self) -> None:
        probabilities = {name: 0.01 for name in (
            "constant_folding",
            "constant_propagation",
            "algebraic_simplification",
            "common_subexpression_elimination",
            "dead_code_elimination",
            "strength_reduction",
            "no_improvement",
        )}
        probabilities["dead_code_elimination"] = 0.9
        program = parse_tac(FALLBACK_TAC)
        original = program.to_tac()
        run = optimize_program(program, RankingSelector("top_wins", probabilities), "top_wins")
        self.assertEqual(program.to_tac(), original)
        self.assertEqual(run.history[0].attempts[0].strategy, "dead_code_elimination")
        self.assertTrue(run.history[0].attempts[0].accepted)
        self.assertEqual(run.strategy_sequence[0], "dead_code_elimination")

    def test_multi_step_sequence_applies_more_than_one_pass(self) -> None:
        program = parse_tac(DEAD_AND_FOLD)
        original = program.to_tac()
        run = optimize_program(program, RuleSelector(), "multi")
        self.assertGreaterEqual(len(run.strategy_sequence), 2)
        self.assertEqual(program.to_tac(), original)

    def test_two_rule_runs_match(self) -> None:
        program = parse_tac(DEAD_AND_FOLD)
        first = optimize_program(program, RuleSelector(), "same")
        second = optimize_program(program, RuleSelector(), "same")
        self.assertEqual(first.strategy_sequence, second.strategy_sequence)
        self.assertEqual(first.final_program.to_tac(), second.final_program.to_tac())


if __name__ == "__main__":
    unittest.main()
