"""Tests for the six optimizers and instruction-count gain.

Each pass is applied to the original program. These tests do not chain passes.
"""

from __future__ import annotations

import unittest

from dataset.examples import load_sample_programs
from evaluation.metrics import instruction_gain, reduction_percentage
from features.extractor import num_dead_stmts
from optimizers import algebraic_simplification
from optimizers import common_subexpression_elimination
from optimizers import constant_folding
from optimizers import constant_propagation
from optimizers import dead_code_elimination
from optimizers import strength_reduction
from optimizers.constant_folding import _div_toward_zero
from optimizers.registry import STRATEGY_NAMES, apply_strategy
from tac.parser import parse_tac
from tac.program import TACProgram


def _optimize(source: str, optimize) -> tuple[TACProgram, TACProgram, str]:
    """Return the original program, the optimized program, and the original text."""
    original = parse_tac(source)
    before = original.to_tac()
    optimized = optimize(original)
    return original, optimized, before


class OptimizerSafetyTests(unittest.TestCase):
    def test_strategy_names_are_the_six_required_names(self) -> None:
        self.assertEqual(
            STRATEGY_NAMES,
            (
                "constant_folding",
                "constant_propagation",
                "algebraic_simplification",
                "common_subexpression_elimination",
                "dead_code_elimination",
                "strength_reduction",
            ),
        )

    def test_every_strategy_preserves_the_input_and_parses_again(self) -> None:
        for program_id, program in load_sample_programs():
            before = program.to_tac()
            for name in STRATEGY_NAMES:
                optimized = apply_strategy(program, name)
                self.assertEqual(program.to_tac(), before, f"{program_id} / {name}")
                self.assertIsNot(optimized, program)
                self.assertEqual(parse_tac(optimized.to_tac()).to_tac(), optimized.to_tac())

    def test_unknown_strategy_is_rejected(self) -> None:
        program = parse_tac("print 1\n")
        with self.assertRaises(KeyError):
            apply_strategy(program, "inline")

    def test_dead_code_gain_matches_the_dead_statement_feature(self) -> None:
        for program_id, program in load_sample_programs():
            optimized = dead_code_elimination.optimize(program)
            self.assertEqual(
                instruction_gain(program, optimized),
                num_dead_stmts(program),
                program_id,
            )

    def test_empty_program_has_zero_reduction_percentage(self) -> None:
        empty = parse_tac("")
        self.assertEqual(reduction_percentage(empty, empty), 0.0)

    def test_division_truncates_toward_zero(self) -> None:
        self.assertEqual(_div_toward_zero(7, 2), 3)
        self.assertEqual(_div_toward_zero(-5, 2), -2)
        self.assertEqual(_div_toward_zero(5, -2), -2)
        self.assertEqual(_div_toward_zero(-5, -2), 2)


class ConstantFoldingTests(unittest.TestCase):
    def test_folds_safe_arithmetic(self) -> None:
        original, optimized, before = _optimize(
            "\n".join(
                [
                    "t1 = 10 + 20",
                    "t2 = 5 * 4",
                    "t3 = 7 / 2",
                    "t4 = 7 % 2",
                    "t5 = 3 - 5",
                    "t6 = - 0",
                    "print t1",
                ]
            ),
            constant_folding.optimize,
        )
        self.assertEqual(original.to_tac(), before)
        self.assertEqual(
            optimized.to_tac(),
            "\n".join(
                [
                    "t1 = 30",
                    "t2 = 20",
                    "t3 = 3",
                    "t4 = 1",
                    "t5 = - 2",
                    "t6 = 0",
                    "print t1",
                ]
            ),
        )
        self.assertEqual(instruction_gain(original, optimized), 0)

    def test_leaves_non_literals_and_zero_division_alone(self) -> None:
        source = "\n".join(
            [
                "t1 = a + 20",
                "t2 = 8 / 0",
                "t3 = 8 % 0",
                "t4 = - a",
                "print t2",
            ]
        )
        original, optimized, before = _optimize(source, constant_folding.optimize)
        self.assertEqual(original.to_tac(), before)
        self.assertEqual(optimized.to_tac(), before)

    def test_does_not_propagate_a_folded_name(self) -> None:
        original, optimized, before = _optimize(
            "t1 = 10 + 20\nt2 = t1 + 5\nprint t2\n",
            constant_folding.optimize,
        )
        self.assertEqual(original.to_tac(), before)
        self.assertEqual(
            optimized.to_tac(),
            "t1 = 30\nt2 = t1 + 5\nprint t2",
        )

    def test_does_not_apply_algebraic_identities_or_dead_code_removal(self) -> None:
        source = "t1 = a + 0\nx = 1\nprint t1\n"
        _, optimized, before = _optimize(source, constant_folding.optimize)
        self.assertEqual(optimized.to_tac(), before)


class ConstantPropagationTests(unittest.TestCase):
    def test_substitutes_a_constant_without_folding_it(self) -> None:
        original, optimized, before = _optimize(
            "x = 10\ny = x + 5\nprint y\n",
            constant_propagation.optimize,
        )
        self.assertEqual(original.to_tac(), before)
        self.assertEqual(optimized.to_tac(), "x = 10\ny = 10 + 5\nprint y")
        self.assertNotIn("y = 15", optimized.to_tac())

    def test_respects_reassignment(self) -> None:
        _, optimized, before = _optimize(
            "x = 10\ny = x + 5\nx = 20\nz = x + 1\nprint z\n",
            constant_propagation.optimize,
        )
        self.assertEqual(
            optimized.to_tac(),
            "x = 10\ny = 10 + 5\nx = 20\nz = 20 + 1\nprint z",
        )
        self.assertNotEqual(optimized.to_tac(), before)

    def test_does_not_treat_a_binary_result_as_a_new_constant(self) -> None:
        _, optimized, _ = _optimize(
            "x = 10\ny = x + 5\nz = y + 1\nprint z\n",
            constant_propagation.optimize,
        )
        self.assertEqual(
            optimized.to_tac(),
            "x = 10\ny = 10 + 5\nz = y + 1\nprint z",
        )

    def test_conflicting_branch_values_are_not_substituted(self) -> None:
        _, optimized, before = _optimize(
            "\n".join(
                [
                    "x = 1",
                    "if a < b goto L1",
                    "x = 2",
                    "L1:",
                    "y = x + 3",
                    "print y",
                ]
            ),
            constant_propagation.optimize,
        )
        self.assertEqual(optimized.to_tac(), before)

    def test_same_constant_on_every_path_is_substituted(self) -> None:
        _, optimized, _ = _optimize(
            "\n".join(
                [
                    "x = 1",
                    "if a < b goto L1",
                    "y = x + 1",
                    "goto END",
                    "L1:",
                    "y = x + 2",
                    "END:",
                    "print y",
                ]
            ),
            constant_propagation.optimize,
        )
        self.assertEqual(
            optimized.to_tac(),
            "\n".join(
                [
                    "x = 1",
                    "if a < b goto L1",
                    "y = 1 + 1",
                    "goto END",
                    "L1:",
                    "y = 1 + 2",
                    "END:",
                    "print y",
                ]
            ),
        )

    def test_loop_kills_a_constant_that_the_back_edge_overwrites(self) -> None:
        source = "\n".join(
            [
                "x = 1",
                "L1:",
                "y = x + 2",
                "x = 3",
                "if a < b goto L1",
                "print y",
            ]
        )
        _, optimized, before = _optimize(source, constant_propagation.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_loop_keeps_a_constant_that_is_never_reassigned(self) -> None:
        _, optimized, _ = _optimize(
            "\n".join(
                [
                    "x = 2",
                    "L1:",
                    "y = x + 1",
                    "if a < b goto L1",
                    "print y",
                ]
            ),
            constant_propagation.optimize,
        )
        self.assertIn("y = 2 + 1", optimized.to_tac())

    def test_negative_constant_is_copied_but_not_placed_in_a_binary_operand(self) -> None:
        _, optimized, _ = _optimize(
            "x = - 10\ny = x + 1\nz = x\nprint x\n",
            constant_propagation.optimize,
        )
        self.assertEqual(
            optimized.to_tac(),
            "x = - 10\ny = x + 1\nz = - 10\nprint x",
        )

    def test_does_not_fold_or_delete_dead_code(self) -> None:
        source = "x = 10\ny = 1\nprint x\n"
        _, optimized, _ = _optimize(source, constant_propagation.optimize)
        self.assertEqual(optimized.to_tac(), "x = 10\ny = 1\nprint 10")
        self.assertIn("y = 1", optimized.to_tac())


class AlgebraicSimplificationTests(unittest.TestCase):
    def test_applies_the_documented_identities(self) -> None:
        _, optimized, _ = _optimize(
            "\n".join(
                [
                    "t1 = a + 0",
                    "t2 = 0 + b",
                    "t3 = c - 0",
                    "t4 = d * 1",
                    "t5 = 1 * e",
                    "t6 = f / 1",
                    "t7 = g * 0",
                    "t8 = 0 * h",
                    "print t1",
                ]
            ),
            algebraic_simplification.optimize,
        )
        self.assertEqual(
            optimized.to_tac(),
            "\n".join(
                [
                    "t1 = a",
                    "t2 = b",
                    "t3 = c",
                    "t4 = d",
                    "t5 = e",
                    "t6 = f",
                    "t7 = 0",
                    "t8 = 0",
                    "print t1",
                ]
            ),
        )

    def test_leaves_ordinary_arithmetic_unchanged(self) -> None:
        source = "t1 = a + b\nprint t1\n"
        _, optimized, before = _optimize(source, algebraic_simplification.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_does_not_rewrite_unsafe_or_unlisted_patterns(self) -> None:
        source = "t1 = 0 - a\nt2 = a / 0\nt3 = a - 1\nprint t1\n"
        _, optimized, before = _optimize(source, algebraic_simplification.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_does_not_fold_or_delete_the_assignment_that_fed_a_multiply(self) -> None:
        _, optimized, _ = _optimize(
            "a = 5\nt1 = 0 * a\nt2 = 10 + 20\nprint t1\n",
            algebraic_simplification.optimize,
        )
        self.assertEqual(
            optimized.to_tac(),
            "a = 5\nt1 = 0\nt2 = 10 + 20\nprint t1",
        )


class CommonSubexpressionEliminationTests(unittest.TestCase):
    def test_reuses_an_available_result(self) -> None:
        _, optimized, _ = _optimize(
            "t1 = a + b\nt2 = c * d\nt3 = a + b\nprint t1\nprint t3\n",
            common_subexpression_elimination.optimize,
        )
        self.assertEqual(
            optimized.to_tac(),
            "t1 = a + b\nt2 = c * d\nt3 = t1\nprint t1\nprint t3",
        )

    def test_does_nothing_when_there_is_no_repeated_expression(self) -> None:
        source = "t1 = a + b\nt2 = c * d\nprint t1\n"
        _, optimized, before = _optimize(source, common_subexpression_elimination.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_does_not_reuse_an_expression_after_reassignment(self) -> None:
        source = "t1 = a + b\na = 4\nt2 = a + b\nprint t1\nprint t2\n"
        _, optimized, before = _optimize(source, common_subexpression_elimination.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_does_not_treat_swapped_operands_as_the_same_expression(self) -> None:
        source = "t1 = a + b\nt2 = b + a\nprint t1\nprint t2\n"
        _, optimized, before = _optimize(source, common_subexpression_elimination.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_join_requires_the_same_result_variable_on_every_path(self) -> None:
        source = "\n".join(
            [
                "if c < d goto L1",
                "t1 = a + b",
                "goto END",
                "L1:",
                "t2 = a + b",
                "END:",
                "t3 = a + b",
                "print t3",
            ]
        )
        _, optimized, before = _optimize(source, common_subexpression_elimination.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_does_not_fold_constants(self) -> None:
        source = "t1 = 10 + 20\nprint t1\n"
        _, optimized, before = _optimize(source, common_subexpression_elimination.optimize)
        self.assertEqual(optimized.to_tac(), before)


class DeadCodeEliminationTests(unittest.TestCase):
    def test_removes_an_assignment_that_a_print_does_not_need(self) -> None:
        original, optimized, before = _optimize(
            "x = 10\ny = 20\nprint y\n",
            dead_code_elimination.optimize,
        )
        self.assertEqual(original.to_tac(), before)
        self.assertEqual(optimized.to_tac(), "y = 20\nprint y")
        self.assertEqual(instruction_gain(original, optimized), 1)
        self.assertAlmostEqual(reduction_percentage(original, optimized), 100 / 3)

    def test_removes_a_dependent_chain(self) -> None:
        original, optimized, _ = _optimize(
            "x = 10\nz = x + 1\nprint 0\n",
            dead_code_elimination.optimize,
        )
        self.assertEqual(optimized.to_tac(), "print 0")
        self.assertEqual(instruction_gain(original, optimized), 2)

    def test_keeps_a_live_straight_line_program(self) -> None:
        source = "t1 = a + b\nprint t1\n"
        _, optimized, before = _optimize(source, dead_code_elimination.optimize)
        self.assertEqual(optimized.to_tac(), before)
        self.assertEqual(instruction_gain(parse_tac(source), optimized), 0)

    def test_keeps_a_value_used_on_a_later_loop_iteration(self) -> None:
        source = "L1:\nprint x\nx = 1\nif a < b goto L1\n"
        _, optimized, before = _optimize(source, dead_code_elimination.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_keeps_an_assignment_that_a_branch_can_still_observe(self) -> None:
        source = "\n".join(
            [
                "x = 1",
                "if a < b goto L1",
                "x = 2",
                "L1:",
                "print x",
            ]
        )
        _, optimized, before = _optimize(source, dead_code_elimination.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_does_not_fold_or_simplify(self) -> None:
        _, optimized, _ = _optimize(
            "t1 = 10 + 20\nt2 = a + 0\nprint t2\n",
            dead_code_elimination.optimize,
        )
        self.assertEqual(optimized.to_tac(), "t2 = a + 0\nprint t2")


class StrengthReductionTests(unittest.TestCase):
    def test_rewrites_multiply_by_two_inside_a_loop(self) -> None:
        original, optimized, before = _optimize(
            "\n".join(
                [
                    "i = 0",
                    "n = 10",
                    "L1:",
                    "t1 = i * 2",
                    "t2 = 2 * i",
                    "i = i + 1",
                    "if i < n goto L1",
                    "print t1",
                ]
            ),
            strength_reduction.optimize,
        )
        self.assertEqual(original.to_tac(), before)
        self.assertEqual(
            optimized.to_tac(),
            "\n".join(
                [
                    "i = 0",
                    "n = 10",
                    "L1:",
                    "t1 = i + i",
                    "t2 = i + i",
                    "i = i + 1",
                    "if i < n goto L1",
                    "print t1",
                ]
            ),
        )
        self.assertEqual(instruction_gain(original, optimized), 0)

    def test_does_nothing_outside_a_loop(self) -> None:
        source = "t1 = a * 2\nprint t1\n"
        _, optimized, before = _optimize(source, strength_reduction.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_does_not_rewrite_unproven_or_overlapping_patterns(self) -> None:
        source = "\n".join(
            [
                "L1:",
                "t1 = i * 3",
                "t2 = i * 1",
                "t3 = i / 2",
                "t4 = 2 * 2",
                "if i < n goto L1",
                "print t1",
            ]
        )
        _, optimized, before = _optimize(source, strength_reduction.optimize)
        self.assertEqual(optimized.to_tac(), before)

    def test_does_not_remove_dead_code_around_the_loop(self) -> None:
        _, optimized, _ = _optimize(
            "dead = 1\nL1:\nt1 = i * 2\nif i < n goto L1\nprint t1\n",
            strength_reduction.optimize,
        )
        self.assertEqual(
            optimized.to_tac(),
            "dead = 1\nL1:\nt1 = i + i\nif i < n goto L1\nprint t1",
        )


if __name__ == "__main__":
    unittest.main()
