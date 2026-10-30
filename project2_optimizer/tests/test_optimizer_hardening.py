"""Adversarial checks for the six optimizers.

Each pass is applied once to the original program. Constant folding and
constant propagation stay separate. One application is a local fixed point:

- folding, algebra, and strength reduction rewrite each instruction in place
- propagation substitutes constants but does not evaluate the resulting add
- common-subexpression elimination rewrites a repeat into a copy, so the
  same binary opportunity is gone
- dead-code elimination repeats inside the pass until no dead assignment
  remains

A second call is therefore expected to reproduce the first result. That is
the current behavior, not an extra iteration policy.
"""

from __future__ import annotations

import unittest

from evaluation.metrics import instruction_gain, weighted_cost_gain
from optimizers.registry import STRATEGY_NAMES, apply_strategy
from tac.parser import parse_tac
from tac.program import TACProgram


def _once(source: str, name: str) -> tuple[TACProgram, TACProgram]:
    program = parse_tac(source)
    return program, apply_strategy(program, name)


class ConstantFoldingHardeningTests(unittest.TestCase):
    def test_positive_zero_and_arithmetic_fold_and_zero_division_does_not(self) -> None:
        source = "\n".join(
            [
                "t1 = 6 + 2",
                "t2 = 4 - 4",
                "t3 = 3 * 4",
                "t4 = 7 / 2",
                "t5 = 7 % 2",
                "t6 = - 9",
                "t7 = 3 - 8",
                "t8 = 8 / 0",
                "t9 = 8 % 0",
                "t10 = a + 1",
                "print t1",
            ]
        )
        original, optimized = _once(source, "constant_folding")
        self.assertEqual(
            optimized.to_tac(),
            "\n".join(
                [
                    "t1 = 8",
                    "t2 = 0",
                    "t3 = 12",
                    "t4 = 3",
                    "t5 = 1",
                    "t6 = - 9",
                    "t7 = - 5",
                    "t8 = 8 / 0",
                    "t9 = 8 % 0",
                    "t10 = a + 1",
                    "print t1",
                ]
            ),
        )
        self.assertIn("t8 = 8 / 0", original.to_tac())

    def test_folding_does_not_propagate_a_folded_result(self) -> None:
        _original, optimized = _once("t1 = 2 + 3\nt2 = t1 + 4\nprint t2\n", "constant_folding")
        self.assertIn("t1 = 5", optimized.to_tac())
        self.assertIn("t2 = t1 + 4", optimized.to_tac())

    def test_folding_on_both_sides_of_a_branch(self) -> None:
        source = "if a < b goto L1\nt1 = 9 * 3\ngoto END\nL1:\nt2 = 8 / 2\nEND:\nprint 0\n"
        _original, optimized = _once(source, "constant_folding")
        text = optimized.to_tac()
        self.assertIn("t1 = 27", text)
        self.assertIn("t2 = 4", text)
        self.assertIn("if a < b goto L1", text)

    def test_no_opportunity_is_unchanged(self) -> None:
        source = "t1 = a + b\nprint t1\n"
        original, optimized = _once(source, "constant_folding")
        self.assertEqual(optimized.to_tac(), original.to_tac())


class ConstantPropagationHardeningTests(unittest.TestCase):
    def test_propagation_does_not_fold_the_addition(self) -> None:
        _original, optimized = _once("x = 10\ny = x + 5\nprint y\n", "constant_propagation")
        self.assertEqual(optimized.to_tac(), "x = 10\ny = 10 + 5\nprint y")
        self.assertNotIn("y = 15", optimized.to_tac())

    def test_reassignment_uses_the_latest_constant(self) -> None:
        source = "x = 10\ny = x + 5\nx = 20\nz = x + 5\nprint y\nprint z\n"
        _original, optimized = _once(source, "constant_propagation")
        self.assertEqual(
            optimized.to_tac(),
            "x = 10\ny = 10 + 5\nx = 20\nz = 20 + 5\nprint y\nprint z",
        )

    def test_copies_of_a_constant_propagate(self) -> None:
        _original, optimized = _once("x = 10\ny = x\nz = y\nprint z\n", "constant_propagation")
        self.assertEqual(optimized.to_tac(), "x = 10\ny = 10\nz = 10\nprint 10")

    def test_negative_constant_is_not_placed_in_a_binary_operand(self) -> None:
        source = "x = - 10\ny = x + 1\nz = x\nprint x\n"
        _original, optimized = _once(source, "constant_propagation")
        text = optimized.to_tac()
        self.assertIn("y = x + 1", text)
        self.assertIn("z = - 10", text)
        self.assertIn("print x", text)
        self.assertNotIn("y = 15", text)

    def test_disagreeing_branch_values_are_not_propagated(self) -> None:
        source = "\n".join(
            [
                "if a < b goto L1",
                "x = 1",
                "goto L2",
                "L1:",
                "x = 2",
                "L2:",
                "y = x + 3",
                "print y",
            ]
        )
        _original, optimized = _once(source, "constant_propagation")
        self.assertIn("y = x + 3", optimized.to_tac())

    def test_agreeing_branch_values_are_propagated(self) -> None:
        source = "\n".join(
            [
                "if a < b goto L1",
                "x = 4",
                "goto L2",
                "L1:",
                "x = 4",
                "L2:",
                "y = x + 3",
                "print y",
            ]
        )
        _original, optimized = _once(source, "constant_propagation")
        self.assertIn("y = 4 + 3", optimized.to_tac())


class AlgebraicHardeningTests(unittest.TestCase):
    def test_every_implemented_identity(self) -> None:
        source = "\n".join(
            [
                "t1 = a + 0",
                "t2 = 0 + a",
                "t3 = a - 0",
                "t4 = a * 1",
                "t5 = 1 * a",
                "t6 = a / 1",
                "t7 = a * 0",
                "t8 = 0 * a",
                "print t1",
            ]
        )
        _original, optimized = _once(source, "algebraic_simplification")
        self.assertEqual(
            optimized.to_tac(),
            "\n".join(
                [
                    "t1 = a",
                    "t2 = a",
                    "t3 = a",
                    "t4 = a",
                    "t5 = a",
                    "t6 = a",
                    "t7 = 0",
                    "t8 = 0",
                    "print t1",
                ]
            ),
        )

    def test_unsupported_forms_stay(self) -> None:
        source = "t1 = 0 - x\nt2 = 10 + 20\nt3 = x / 0\nt4 = x * 3\nprint t1\n"
        original, optimized = _once(source, "algebraic_simplification")
        self.assertEqual(optimized.to_tac(), original.to_tac())


class CommonSubexpressionHardeningTests(unittest.TestCase):
    def test_repeats_reuse_the_latest_valid_temporary(self) -> None:
        _original, optimized = _once(
            "t1 = a + b\nt2 = a + b\nt3 = a + b\nprint t3\n",
            "common_subexpression_elimination",
        )
        text = optimized.to_tac()
        self.assertIn("t1 = a + b", text)
        self.assertIn("t2 = t1", text)
        self.assertIn("t3 = t2", text)

    def test_operand_reassignment_blocks_reuse(self) -> None:
        _original, optimized = _once(
            "t1 = a + b\na = 4\nt2 = a + b\nprint t1\nprint t2\n",
            "common_subexpression_elimination",
        )
        self.assertEqual(
            optimized.to_tac(),
            "t1 = a + b\na = 4\nt2 = a + b\nprint t1\nprint t2",
        )

    def test_unrelated_assignment_does_not_block_reuse(self) -> None:
        _original, optimized = _once(
            "t1 = a + b\nc = 1\nt2 = a + b\nprint t2\n",
            "common_subexpression_elimination",
        )
        self.assertIn("t2 = t1", optimized.to_tac())

    def test_addition_is_not_commutative(self) -> None:
        original, optimized = _once("t1 = a + b\nt2 = b + a\nprint t2\n", "common_subexpression_elimination")
        self.assertEqual(optimized.to_tac(), original.to_tac())

    def test_a_join_reuses_an_expression_only_when_every_path_agrees(self) -> None:
        blocked = "\n".join(
            [
                "if c < d goto L1",
                "t1 = a + b",
                "goto END",
                "L1:",
                "t1 = a + c",
                "END:",
                "t2 = a + b",
                "print t2",
            ]
        )
        _original, optimized = _once(blocked, "common_subexpression_elimination")
        self.assertIn("t2 = a + b", optimized.to_tac())

        available = "\n".join(
            [
                "t1 = a + b",
                "if c < d goto L1",
                "goto END",
                "L1:",
                "END:",
                "t2 = a + b",
                "print t2",
            ]
        )
        _original, optimized = _once(available, "common_subexpression_elimination")
        self.assertIn("t2 = t1", optimized.to_tac())


class DeadCodeHardeningTests(unittest.TestCase):
    def test_unused_chain_is_removed_and_a_second_pass_is_stable(self) -> None:
        source = "a = 1\nb = a + 2\nprint 7\n"
        original, once = _once(source, "dead_code_elimination")
        self.assertEqual(once.to_tac(), "print 7")
        twice = apply_strategy(once, "dead_code_elimination")
        self.assertEqual(once.to_tac(), twice.to_tac())
        self.assertEqual(original.to_tac(), "a = 1\nb = a + 2\nprint 7")

    def test_a_used_value_a_print_and_a_condition_stay(self) -> None:
        used = parse_tac("a = 1\nprint a\n")
        self.assertEqual(apply_strategy(used, "dead_code_elimination").to_tac(), used.to_tac())
        printed = parse_tac("print 5\n")
        self.assertEqual(apply_strategy(printed, "dead_code_elimination").to_tac(), "print 5")
        condition = parse_tac("x = 1\nif x < n goto L1\nprint 0\nL1:\nprint 1\n")
        optimized = apply_strategy(condition, "dead_code_elimination")
        self.assertIn("x = 1", optimized.to_tac())
        self.assertIn("print 0", optimized.to_tac())
        self.assertIn("print 1", optimized.to_tac())

    def test_control_flow_keeps_a_value_used_on_one_path(self) -> None:
        source = "x = 1\nif a < b goto L1\nprint x\ngoto END\nL1:\nprint 0\nEND:\n"
        _original, optimized = _once(source, "dead_code_elimination")
        self.assertIn("x = 1", optimized.to_tac())

    def test_an_unreachable_assignment_is_removed(self) -> None:
        _original, optimized = _once("goto L1\ndead = 1\nL1:\nprint 0\n", "dead_code_elimination")
        self.assertNotIn("dead = 1", optimized.to_tac())
        self.assertIn("print 0", optimized.to_tac())
        self.assertIn("goto L1", optimized.to_tac())


class StrengthReductionHardeningTests(unittest.TestCase):
    def test_loop_multiply_by_two_changes_cost_not_instruction_count(self) -> None:
        left = parse_tac("L1:\nt1 = i * 2\nif i < n goto L1\nprint t1\n")
        right = parse_tac("L1:\nt1 = 2 * i\nif i < n goto L1\nprint t1\n")
        for program in (left, right):
            optimized = apply_strategy(program, "strength_reduction")
            self.assertIn("t1 = i + i", optimized.to_tac())
            self.assertEqual(instruction_gain(program, optimized), 0)
            self.assertEqual(weighted_cost_gain(program, optimized), 1)

    def test_forms_outside_the_rule_stay(self) -> None:
        outside = parse_tac("t1 = i * 2\nprint t1\n")
        self.assertEqual(apply_strategy(outside, "strength_reduction").to_tac(), outside.to_tac())
        triple = parse_tac("L1:\nt1 = i * 3\nif i < n goto L1\nprint t1\n")
        self.assertEqual(apply_strategy(triple, "strength_reduction").to_tac(), triple.to_tac())
        square = parse_tac("L1:\nt1 = i * i\nif i < n goto L1\nprint t1\n")
        self.assertEqual(apply_strategy(square, "strength_reduction").to_tac(), square.to_tac())
        folded = parse_tac("L1:\nt1 = 2 * 2\nif i < n goto L1\nprint t1\n")
        self.assertEqual(apply_strategy(folded, "strength_reduction").to_tac(), folded.to_tac())


class ImmutabilityAndIdempotenceTests(unittest.TestCase):
    def test_every_optimizer_copies_the_program_and_is_idempotent(self) -> None:
        programs = [
            "t1 = 6 + 2\nt2 = a + 0\nt3 = a + b\nt4 = a + b\nspare = 1\nprint t4\n",
            "x = 10\ny = x + 5\nx = 20\nz = x + 5\nprint z\n",
            "i = 0\nL1:\nt1 = i * 2\ni = i + 1\nif i < n goto L1\nprint t1\n",
            "t1 = 8 / 0\nprint t1\n",
            "t1 = a + b\na = 4\nt2 = a + b\nprint t1\nprint t2\n",
            "print 1\n",
        ]
        for source in programs:
            program = parse_tac(source)
            before = program.to_tac()
            original_list = program.instructions
            for name in STRATEGY_NAMES:
                once = apply_strategy(program, name)
                twice = apply_strategy(once, name)
                self.assertEqual(program.to_tac(), before, name)
                self.assertIs(program.instructions, original_list, name)
                self.assertIsNot(once, program, name)
                self.assertIsNot(once.instructions, program.instructions, name)
                self.assertEqual(once.to_tac(), twice.to_tac(), name)


if __name__ == "__main__":
    unittest.main()
