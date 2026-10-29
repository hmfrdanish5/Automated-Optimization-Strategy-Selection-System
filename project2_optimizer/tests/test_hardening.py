"""Extra correctness checks for the six optimizers and the engine."""

from __future__ import annotations

import unittest

from evaluation.metrics import instruction_gain, weighted_cost_gain
from optimizers.registry import STRATEGY_NAMES, apply_strategy
from tac.interpreter import execute
from tac.parser import parse_tac


class AdversarialOptimizerTests(unittest.TestCase):
    def test_folding_covers_boundary_arithmetic_and_refuses_zero_division(self) -> None:
        program = parse_tac(
            "\n".join(
                [
                    "t1 = 0 + 4",
                    "t2 = 4 - 0",
                    "t3 = 0 * 4",
                    "t4 = 4 / 1",
                    "t5 = 4 % 1",
                    "t6 = - 7",
                    "t7 = 8 / 0",
                    "t8 = 8 % 0",
                    "print t1",
                ]
            )
        )
        optimized = apply_strategy(program, "constant_folding")
        self.assertEqual(
            optimized.to_tac(),
            "\n".join(
                [
                    "t1 = 4",
                    "t2 = 4",
                    "t3 = 0",
                    "t4 = 4",
                    "t5 = 0",
                    "t6 = - 7",
                    "t7 = 8 / 0",
                    "t8 = 8 % 0",
                    "print t1",
                ]
            ),
        )

    def test_cse_reuses_the_latest_equivalent_after_a_kill(self) -> None:
        optimized = apply_strategy(
            parse_tac("t1 = a + b\na = 4\nt2 = a + b\nt3 = a + b\nprint t3\n"),
            "common_subexpression_elimination",
        )
        self.assertIn("t2 = a + b", optimized.to_tac())
        self.assertIn("t3 = t2", optimized.to_tac())
        self.assertNotIn("t3 = t1", optimized.to_tac())

    def test_cse_ignores_an_unrelated_assignment(self) -> None:
        optimized = apply_strategy(
            parse_tac("t1 = a + b\nc = 1\nt2 = a + b\nprint t2\n"),
            "common_subexpression_elimination",
        )
        self.assertIn("t2 = t1", optimized.to_tac())

    def test_dead_code_keeps_a_condition_operand_and_is_idempotent_on_a_chain(self) -> None:
        source = "x = 1\nif x < n goto L1\nprint 0\nL1:\nprint 1\n"
        once = apply_strategy(parse_tac(source), "dead_code_elimination")
        self.assertIn("x = 1", once.to_tac())
        twice = apply_strategy(once, "dead_code_elimination")
        self.assertEqual(once.to_tac(), twice.to_tac())

    def test_strength_reduction_refuses_square_and_reports_cost_not_instruction_gain(self) -> None:
        square = parse_tac("L1:\nt1 = i * i\nif i < n goto L1\nprint t1\n")
        self.assertEqual(apply_strategy(square, "strength_reduction").to_tac(), square.to_tac())
        loop = parse_tac("L1:\nt1 = i * 2\nif i < n goto L1\nprint t1\n")
        optimized = apply_strategy(loop, "strength_reduction")
        self.assertEqual(instruction_gain(loop, optimized), 0)
        self.assertGreater(weighted_cost_gain(loop, optimized), 0)


class ImmutabilityAndIdempotenceTests(unittest.TestCase):
    def test_every_optimizer_returns_a_new_program_and_a_fixed_point(self) -> None:
        programs = [
            parse_tac("t1 = 6 + 2\nt2 = a + 0\nt3 = a + b\nt4 = a + b\nspare = 1\nprint t4\n"),
            parse_tac("i = 0\nL1:\nt1 = i * 2\ni = i + 1\nif i < n goto L1\nprint t1\n"),
            parse_tac("x = 3\ny = x + 1\nprint y\n"),
        ]
        for program in programs:
            before = program.to_tac()
            original_list = program.instructions
            for name in STRATEGY_NAMES:
                once = apply_strategy(program, name)
                twice = apply_strategy(once, name)
                self.assertEqual(program.to_tac(), before, name)
                self.assertIs(program.instructions, original_list)
                self.assertIsNot(once, program)
                self.assertIsNot(once.instructions, program.instructions)
                self.assertEqual(once.to_tac(), twice.to_tac(), name)


class SemanticPreservationTests(unittest.TestCase):
    def test_interpreter_matches_folding_and_branches(self) -> None:
        program = parse_tac("t1 = 7 / 2\nt2 = - 3\nprint t1\nprint t2\n")
        self.assertEqual(execute(program).prints, (3, -3))
        branched = parse_tac("if a < b goto L1\nprint 1\ngoto END\nL1:\nprint 2\nEND:\n")
        self.assertEqual(execute(branched, {"a": 1, "b": 4}).prints, (2,))
        self.assertEqual(execute(branched, {"a": 5, "b": 4}).prints, (1,))
        fault = parse_tac("t1 = 1 / 0\nprint t1\n")
        self.assertEqual(execute(fault).error, "division by zero")

    def test_each_optimizer_preserves_prints(self) -> None:
        cases = [
            ("t1 = 8 * 3\nprint t1\n", {}),
            ("x = 10\ny = x + 5\nprint y\n", {}),
            ("t1 = a + 0\nprint t1\n", {"a": 6}),
            ("t1 = a + b\nt2 = a + b\nprint t2\n", {"a": 2, "b": 9}),
            ("t1 = a + b\na = 4\nt2 = a + b\nprint t1\nprint t2\n", {"a": 2, "b": 9}),
            ("spare = 1\nprint 5\n", {}),
            ("i = 0\nL1:\nt1 = i * 2\ni = i + 1\nif i < n goto L1\nprint t1\n", {"n": 3}),
            ("t1 = 0 * a\nprint t1\n", {"a": 12}),
        ]
        for source, inputs in cases:
            original = parse_tac(source)
            expected = execute(original, inputs)
            for name in STRATEGY_NAMES:
                optimized = apply_strategy(original, name)
                self.assertEqual(execute(optimized, inputs), expected, f"{name} / {source}")


if __name__ == "__main__":
    unittest.main()
