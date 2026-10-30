"""Semantic differential tests for source → TAC → optimizer pipeline."""

from __future__ import annotations

import unittest

from engine.loop import optimize_program
from engine.selectors import FixedOrderSelector
from optimizers.registry import STRATEGY_NAMES, apply_strategy
from source_frontend.compiler import compile_source
from tac.interpreter import execute


SOURCE_CASES = [
    "int a = 3;\nint b = 4;\nprint(a + b);\n",
    "int x = 10;\nwhile (x > 0) { x = x - 1; }\nprint(x);\n",
    """
    int a = 1;
    int b = 2;
    if (a != b) { print(a); } else { print(b); }
    """,
    """
    int a = 5;
    int b = 0;
    print(a);
    int dead = a / b;
    print(dead);
    """,
]


class SourceSemanticTests(unittest.TestCase):
    def test_each_optimizer_preserves_source_program_behavior(self) -> None:
        for index, source in enumerate(SOURCE_CASES):
            compiled = compile_source(source)
            self.assertTrue(compiled.ok, compiled.errors)
            original = compiled.program
            assert original is not None
            expected = execute(original)
            for strategy in STRATEGY_NAMES:
                with self.subTest(case=index, strategy=strategy):
                    optimized = apply_strategy(original, strategy)
                    actual = execute(optimized)
                    self.assertEqual(actual, expected, f"{strategy} case {index}")

    def test_iterative_engine_preserves_source_program_behavior(self) -> None:
        selector = FixedOrderSelector()
        source = """
        int i = 0;
        int acc = 0;
        while (i < 4) {
            acc = acc + i;
            i = i + 1;
        }
        print(acc);
        """
        compiled = compile_source(source)
        self.assertTrue(compiled.ok)
        original = compiled.program
        assert original is not None
        expected = execute(original)
        run = optimize_program(original, selector, program_id="source_sem")
        actual = execute(run.final_program)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
