"""Semantic preservation for the six optimizers.

`execute` is the observable semantics: printed integers, and an error string
when division or remainder by zero stops the program. A variable that the
program never assigns is an input. Integer division truncates toward zero.

These tests compare that observable result before and after each optimizer.
They do not treat a model prediction as the expected output.
"""

from __future__ import annotations

import unittest

from optimizers.registry import STRATEGY_NAMES, apply_strategy
from tac.interpreter import execute
from tac.parser import parse_tac


# (source, input maps). Each map is one run. An empty map means no inputs.
SEMANTIC_CASES = [
    ("t1 = 8 * 3\nprint t1\n", [{}]),
    ("t1 = 7 / 2\nprint t1\n", [{}]),
    ("t1 = 7 % 2\nprint t1\n", [{}]),
    ("t1 = 3 - 8\nprint t1\n", [{}]),
    ("t1 = 8 / 0\nprint t1\n", [{}]),
    ("t1 = 8 % 0\nprint t1\n", [{}]),
    ("t1 = a + b\nprint t1\n", [{"a": 1, "b": 2}, {"a": 0, "b": 0}]),
    ("x = 10\ny = x + 5\nprint y\n", [{}]),
    ("x = 10\ny = x + 5\nx = 20\nz = x + 5\nprint y\nprint z\n", [{}]),
    ("x = 10\ny = x\nz = y\nprint z\n", [{}]),
    ("x = - 4\ny = x + 1\nprint y\n", [{}]),
    ("t1 = a + 0\nprint t1\n", [{"a": 6}, {"a": 0}]),
    ("t1 = 0 * a\nprint t1\n", [{"a": 12}, {"a": 0}]),
    ("t1 = 0 - w\nprint t1\n", [{"w": 3}, {"w": 0}]),
    ("t1 = a + b\nt2 = a + b\nprint t2\n", [{"a": 2, "b": 9}, {"a": 0, "b": 1}]),
    (
        "t1 = a + b\na = 4\nt2 = a + b\nprint t1\nprint t2\n",
        [{"a": 2, "b": 9}, {"a": 4, "b": 1}],
    ),
    ("t1 = a + b\nt2 = b + a\nprint t1\nprint t2\n", [{"a": 2, "b": 5}]),
    ("spare = 1\nprint 5\n", [{}]),
    ("a = 1\nb = a + 2\nprint 7\n", [{}]),
    ("x = 1\nif x < n goto L1\nprint 0\nL1:\nprint 1\n", [{"n": 0}, {"n": 4}]),
    (
        "i = 0\nL1:\nt1 = i * 2\ni = i + 1\nif i < n goto L1\nprint t1\n",
        [{"n": 0}, {"n": 3}],
    ),
    (
        "i = 0\nL1:\nt1 = 2 * i\ni = i + 1\nif i < n goto L1\nprint t1\n",
        [{"n": 1}, {"n": 4}],
    ),
    ("t1 = z * 2\nprint t1\n", [{"z": 5}, {"z": 0}]),
    ("t1 = - 5\nt2 = t1 / 2\nprint t2\n", [{}]),
]


class InterpreterTests(unittest.TestCase):
    def test_division_truncates_toward_zero_and_zero_is_an_error(self) -> None:
        self.assertEqual(execute(parse_tac("t1 = 7 / 2\nprint t1\n")).prints, (3,))
        negative = parse_tac("t1 = - 5\nt2 = t1 / 2\nt3 = t1 % 2\nprint t2\nprint t3\n")
        self.assertEqual(execute(negative).prints, (-2, -1))
        fault = execute(parse_tac("t1 = 1 / 0\nprint t1\n"))
        self.assertEqual(fault.prints, ())
        self.assertEqual(fault.error, "division by zero")
        remainder = execute(parse_tac("print 1\nt1 = 4 % 0\n"))
        self.assertEqual(remainder.prints, (1,))
        self.assertEqual(remainder.error, "division by zero")

    def test_branches_and_inputs(self) -> None:
        program = parse_tac("if a < b goto L1\nprint 1\ngoto END\nL1:\nprint 2\nEND:\n")
        self.assertEqual(execute(program, {"a": 1, "b": 4}).prints, (2,))
        self.assertEqual(execute(program, {"a": 5, "b": 4}).prints, (1,))
        missing = execute(parse_tac("print a\n"), {})
        self.assertEqual(missing.error, "undefined 'a'")


class SemanticPreservationTests(unittest.TestCase):
    def test_each_optimizer_preserves_observable_output(self) -> None:
        checked = 0
        for source, input_sets in SEMANTIC_CASES:
            original = parse_tac(source)
            for inputs in input_sets:
                expected = execute(original, inputs)
                for name in STRATEGY_NAMES:
                    optimized = apply_strategy(original, name)
                    actual = execute(optimized, inputs)
                    self.assertEqual(actual, expected, f"{name} / {source} / {inputs}")
                    checked += 1
        self.assertEqual(checked, 6 * sum(len(item[1]) for item in SEMANTIC_CASES))

    def test_dead_code_keeps_unused_division_that_may_trap(self) -> None:
        source = "print 1\nprint 2\ndead = a / d\nprint 3\n"
        original = parse_tac(source)
        optimized = apply_strategy(original, "dead_code_elimination")
        self.assertIn("dead = a / d", optimized.to_tac())
        for inputs in ({"d": 0}, {"d": 2, "a": 6}):
            self.assertEqual(
                execute(optimized, inputs),
                execute(original, inputs),
                inputs,
            )

    def test_redefined_cse_and_zero_division_keep_different_observables(self) -> None:
        redefined = parse_tac("t1 = a + b\na = 4\nt2 = a + b\nprint t1\nprint t2\n")
        result = execute(redefined, {"a": 2, "b": 9})
        self.assertEqual(result.prints, (11, 13))
        optimized = apply_strategy(redefined, "common_subexpression_elimination")
        self.assertEqual(execute(optimized, {"a": 2, "b": 9}).prints, (11, 13))
        fault = parse_tac("t1 = 9 / 0\nprint t1\n")
        self.assertEqual(apply_strategy(fault, "constant_folding").to_tac(), fault.to_tac())
        self.assertEqual(execute(apply_strategy(fault, "constant_folding")).error, "division by zero")


if __name__ == "__main__":
    unittest.main()
