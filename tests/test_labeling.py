"""Tests for metrics, label comparison, and dataset rows."""

from __future__ import annotations

from collections import Counter
import tempfile
import unittest
from pathlib import Path

from dataset.dataset_builder import build_dataset, distribution_report, write_dataset
from dataset.generator import GENERATION_SEED, generate_programs
from dataset.labeler import DATASET_COLUMNS, label_program, label_row, training_features
from evaluation.comparison import StrategyScore, choose_best_strategy, compare_strategy_results
from evaluation.label_policy import LABEL_CRITERIA, NO_IMPROVEMENT
from evaluation.metrics import (
    arithmetic_operation_count,
    code_size,
    instruction_count,
    operation_gain,
    temporary_variable_count,
    weighted_cost_gain,
    weighted_operation_cost,
)
from evaluation.operation_cost import OPERATION_COST
from features.extractor import FEATURE_COLUMNS, num_tmp_vars
from optimizers import constant_folding
from optimizers import dead_code_elimination
from optimizers import strength_reduction
from tac.parser import parse_tac


def _score(
    strategy: str,
    instruction: int = 0,
    weighted: int = 0,
    operation: int = 0,
    temporary: int = 0,
    size: int = 0,
) -> StrategyScore:
    return StrategyScore(strategy, instruction, weighted, operation, temporary, size)


class MetricTests(unittest.TestCase):
    def test_counts_and_folding_gains(self) -> None:
        original = parse_tac("t1 = 10 + 20\nt2 = - a\nprint t1\n")
        self.assertEqual(instruction_count(original), 3)
        self.assertEqual(arithmetic_operation_count(original), 2)
        self.assertEqual(temporary_variable_count(original), 2)
        self.assertEqual(temporary_variable_count(original), num_tmp_vars(original))
        self.assertEqual(code_size(original), len(original.to_tac()))

        folded = constant_folding.optimize(original)
        self.assertEqual(operation_gain(original, folded), 1)
        self.assertEqual(weighted_cost_gain(original, folded), OPERATION_COST["+"])
        self.assertEqual(instruction_count(original) - instruction_count(folded), 0)
        self.assertGreater(code_size(original) - code_size(folded), 0)

    def test_relative_operator_weights(self) -> None:
        self.assertEqual(OPERATION_COST["+"], 1)
        self.assertEqual(OPERATION_COST["-"], 1)
        self.assertEqual(OPERATION_COST["*"], 2)
        self.assertEqual(OPERATION_COST["/"], 4)
        self.assertEqual(OPERATION_COST["%"], 4)
        program = parse_tac(
            "\n".join(
                [
                    "t1 = a + b",
                    "t2 = a - b",
                    "t3 = a * b",
                    "t4 = a / b",
                    "t5 = a % b",
                    "print t1",
                    "L1:",
                    "goto L1",
                ]
            )
        )
        self.assertEqual(weighted_operation_cost(program), 1 + 1 + 2 + 4 + 4)
        self.assertEqual(arithmetic_operation_count(program), 5)

    def test_non_arithmetic_statements_have_no_operation_cost(self) -> None:
        program = parse_tac("x = 1\nprint x\nL1:\ngoto L1\n")
        self.assertEqual(weighted_operation_cost(program), 0)
        self.assertEqual(arithmetic_operation_count(program), 0)

    def test_zero_division_is_still_an_arithmetic_operation(self) -> None:
        program = parse_tac("t1 = 8 / 0\nprint t1\n")
        self.assertEqual(arithmetic_operation_count(program), 1)
        self.assertEqual(weighted_operation_cost(program), OPERATION_COST["/"])
        self.assertEqual(constant_folding.optimize(program).to_tac(), program.to_tac())


class ComparisonTests(unittest.TestCase):
    def test_policy_lists_instruction_count_first(self) -> None:
        self.assertEqual(
            LABEL_CRITERIA,
            (
                "instruction_gain",
                "weighted_cost_gain",
                "operation_gain",
                "temporary_gain",
                "code_size_gain",
            ),
        )

    def test_instruction_gain_outranks_weighted_cost(self) -> None:
        shorter = _score("dead_code_elimination", instruction=1, weighted=0)
        cheaper = _score("strength_reduction", instruction=0, weighted=5)
        self.assertGreater(compare_strategy_results(shorter, cheaper), 0)
        self.assertEqual(choose_best_strategy([cheaper, shorter]), "dead_code_elimination")

    def test_later_metrics_break_ties_in_order(self) -> None:
        lower_cost = _score("strength_reduction", weighted=2, operation=0)
        fewer_ops = _score("common_subexpression_elimination", weighted=1, operation=5)
        self.assertGreater(compare_strategy_results(lower_cost, fewer_ops), 0)

        fewer_operations = _score("algebraic_simplification", operation=2, temporary=0)
        fewer_temps = _score("constant_propagation", operation=1, temporary=5)
        self.assertGreater(compare_strategy_results(fewer_operations, fewer_temps), 0)

        temps = _score("common_subexpression_elimination", temporary=2, size=0)
        size = _score("constant_propagation", temporary=1, size=50)
        self.assertGreater(compare_strategy_results(temps, size), 0)

        larger_text_drop = _score("algebraic_simplification", size=9)
        smaller_text_drop = _score("constant_folding", size=4)
        self.assertGreater(compare_strategy_results(larger_text_drop, smaller_text_drop), 0)

    def test_priority_breaks_a_complete_metric_tie(self) -> None:
        folding = _score("constant_folding", weighted=1, size=3)
        algebraic = _score("algebraic_simplification", weighted=1, size=3)
        self.assertGreater(compare_strategy_results(folding, algebraic), 0)
        self.assertEqual(choose_best_strategy([algebraic, folding]), "constant_folding")

    def test_all_zero_results_are_no_improvement(self) -> None:
        scores = [
            _score(name)
            for name in (
                "constant_folding",
                "constant_propagation",
                "algebraic_simplification",
                "common_subexpression_elimination",
                "dead_code_elimination",
                "strength_reduction",
            )
        ]
        self.assertEqual(choose_best_strategy(scores), NO_IMPROVEMENT)

    def test_a_negative_gain_does_not_become_the_label(self) -> None:
        scores = [
            _score("constant_folding"),
            _score("strength_reduction", size=-4),
        ]
        self.assertEqual(choose_best_strategy(scores), NO_IMPROVEMENT)


class LabelGenerationTests(unittest.TestCase):
    def test_each_gain_matches_that_optimizer_alone(self) -> None:
        program = parse_tac("t1 = 10 + 20\nx = 1\nprint t1\n")
        before = program.to_tac()
        labeled = label_program(program)
        self.assertEqual(program.to_tac(), before)
        folded = constant_folding.optimize(program.copy())
        folding_score = next(score for score in labeled.scores if score.strategy == "constant_folding")
        self.assertEqual(folding_score.operation_gain, operation_gain(program, folded))
        eliminated = dead_code_elimination.optimize(program.copy())
        dead_score = next(score for score in labeled.scores if score.strategy == "dead_code_elimination")
        self.assertEqual(dead_score.instruction_gain, instruction_count(program) - instruction_count(eliminated))
        self.assertNotEqual(folded.to_tac(), eliminated.to_tac())

    def test_dead_code_wins_when_it_deletes_and_cse_only_rewrites(self) -> None:
        program = parse_tac(
            "\n".join(
                [
                    "t1 = a + b",
                    "t2 = a + b",
                    "spare = 1",
                    "print t1",
                ]
            )
        )
        labeled = label_program(program)
        cse = next(score for score in labeled.scores if score.strategy == "common_subexpression_elimination")
        dead = next(score for score in labeled.scores if score.strategy == "dead_code_elimination")
        self.assertGreater(cse.operation_gain, 0)
        self.assertGreater(dead.instruction_gain, 0)
        self.assertEqual(labeled.best_strategy, "dead_code_elimination")

    def test_single_opportunity_labels(self) -> None:
        cases = {
            "t1 = 10 + 20\nprint t1\n": "constant_folding",
            "t1 = a + 0\nprint t1\n": "algebraic_simplification",
            "t1 = a + b\nt2 = a + b\nprint t1\nprint t2\n": "common_subexpression_elimination",
            "x = 1\ny = 2\nprint y\n": "dead_code_elimination",
            "t1 = a + b\nprint t1\n": NO_IMPROVEMENT,
        }
        for source, expected in cases.items():
            self.assertEqual(label_program(parse_tac(source)).best_strategy, expected, source)

    def test_multiply_by_two_in_a_loop_is_strength_reduction(self) -> None:
        for assignment in ("t1 = i * 2", "t1 = 2 * i"):
            source = "\n".join(
                [
                    "i = 0",
                    "L1:",
                    assignment,
                    "i = i + 1",
                    "if i < 10 goto L1",
                    "print t1",
                    "print i",
                ]
            )
            original = parse_tac(source)
            reduced = strength_reduction.optimize(original)
            self.assertEqual(instruction_count(original), instruction_count(reduced))
            self.assertEqual(operation_gain(original, reduced), 0)
            self.assertEqual(weighted_cost_gain(original, reduced), OPERATION_COST["*"] - OPERATION_COST["+"])
            self.assertIn("t1 = i + i", reduced.to_tac())
            labeled = label_program(original)
            strength = next(score for score in labeled.scores if score.strategy == "strength_reduction")
            self.assertEqual(strength.instruction_gain, 0)
            self.assertEqual(strength.weighted_cost_gain, 1)
            self.assertEqual(labeled.best_strategy, "strength_reduction", assignment)

    def test_weighted_cost_outranks_propagation_code_size(self) -> None:
        source = "\n".join(
            [
                "i = 0",
                "limit = 8",
                "L1:",
                "t1 = i * 2",
                "i = i + 1",
                "if i < limit goto L1",
                "print t1",
            ]
        )
        labeled = label_program(parse_tac(source))
        propagation = next(score for score in labeled.scores if score.strategy == "constant_propagation")
        strength = next(score for score in labeled.scores if score.strategy == "strength_reduction")
        self.assertGreater(propagation.code_size_gain, 0)
        self.assertEqual(propagation.weighted_cost_gain, 0)
        self.assertGreater(strength.weighted_cost_gain, 0)
        self.assertEqual(labeled.best_strategy, "strength_reduction")

    def test_propagation_still_wins_when_no_operator_gets_cheaper(self) -> None:
        source = "value = 8\nt1 = value + 3\nprint t1\nprint value\n"
        labeled = label_program(parse_tac(source))
        propagation = next(score for score in labeled.scores if score.strategy == "constant_propagation")
        self.assertEqual(propagation.weighted_cost_gain, 0)
        self.assertGreater(propagation.code_size_gain, 0)
        self.assertEqual(labeled.best_strategy, "constant_propagation")

    def test_repeated_multiply_is_not_forced_to_strength_reduction(self) -> None:
        source = "\n".join(
            [
                "i = 0",
                "L1:",
                "t1 = i * 2",
                "t2 = i * 2",
                "i = i + 1",
                "if i < 6 goto L1",
                "print t1",
                "print t2",
            ]
        )
        labeled = label_program(parse_tac(source))
        self.assertEqual(labeled.best_strategy, "common_subexpression_elimination")

    def test_hand_written_labels_come_from_measurement(self) -> None:
        expected = {
            "constant_heavy": "constant_folding",
            "propagation_ready": NO_IMPROVEMENT,
            "algebraic_heavy": "algebraic_simplification",
            "cse_heavy": "common_subexpression_elimination",
            "dead_code_heavy": "dead_code_elimination",
            "loop_heavy": "strength_reduction",
            "mixed": "dead_code_elimination",
            "straight_line": NO_IMPROVEMENT,
            "expression_redefined": NO_IMPROVEMENT,
        }
        rows = build_dataset(generated_count=0)
        self.assertEqual(
            {str(row["program_id"]): row["best_strategy"] for row in rows},
            expected,
        )


class DatasetRowTests(unittest.TestCase):
    def test_row_columns_and_feature_leakage(self) -> None:
        row = label_row("example", parse_tac("t1 = 10 + 20\nprint t1\n"))
        self.assertEqual(list(row), DATASET_COLUMNS)
        features = training_features(row)
        self.assertEqual(list(features), FEATURE_COLUMNS)
        for column in features:
            self.assertNotIn("gain", column)
            self.assertNotIn("best", column)
        self.assertNotIn("best_strategy", features)
        self.assertNotIn("cf_instruction_gain", features)
        self.assertEqual(row["best_strategy"], "constant_folding")
        self.assertEqual(row["best_primary_gain"], 0)
        self.assertGreater(row["best_secondary_gain"], 0)

    def test_same_seed_reproduces_programs_and_labels(self) -> None:
        first = generate_programs(8, seed=GENERATION_SEED)
        second = generate_programs(8, seed=GENERATION_SEED)
        self.assertEqual([item[0] for item in first], [item[0] for item in second])
        self.assertEqual([item[1].to_tac() for item in first], [item[1].to_tac() for item in second])
        self.assertEqual(
            [label_program(item[1]).best_strategy for item in first],
            [label_program(item[1]).best_strategy for item in second],
        )
        texts = [item[1].to_tac() for item in first]
        self.assertEqual(len(texts), len(set(texts)))

    def test_dataset_csv_round_trip_keeps_diagnostics_out_of_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            rows = write_dataset(path, generated_count=7, seed=1)
        self.assertEqual(len(rows), 16)
        strategies = {row["best_strategy"] for row in rows}
        allowed = {
            "constant_folding",
            "constant_propagation",
            "algebraic_simplification",
            "common_subexpression_elimination",
            "dead_code_elimination",
            "strength_reduction",
            NO_IMPROVEMENT,
        }
        self.assertTrue(strategies <= allowed)
        for row in rows:
            self.assertEqual(set(training_features(row)), set(FEATURE_COLUMNS))
        report = distribution_report(rows)
        self.assertIn("total samples: 16", report)
        self.assertIn("minimum class count:", report)

    def test_a_missing_class_is_reported(self) -> None:
        report = distribution_report(
            [{"best_strategy": "constant_folding"}, {"best_strategy": NO_IMPROVEMENT}]
        )
        self.assertIn("WARNING: strength_reduction has no samples", report)

    def test_full_review_dataset_contains_every_optimizer_class(self) -> None:
        rows = build_dataset()
        self.assertEqual(len(rows), 57)
        counts = Counter(str(row["best_strategy"]) for row in rows)
        for strategy in (
            "constant_folding",
            "constant_propagation",
            "algebraic_simplification",
            "common_subexpression_elimination",
            "dead_code_elimination",
            "strength_reduction",
        ):
            self.assertGreaterEqual(counts[strategy], 1, strategy)
        self.assertGreaterEqual(counts[NO_IMPROVEMENT], 1)
        self.assertLessEqual(max(counts.values()) / len(rows), 0.5)


if __name__ == "__main__":
    unittest.main()
