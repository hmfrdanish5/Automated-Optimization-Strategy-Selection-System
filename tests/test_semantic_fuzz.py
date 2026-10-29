"""Deterministic differential fuzz tests for TAC semantic preservation.

The generator is intentionally small and bounded: every input program uses a
fixed set of input variables, contains ordinary arithmetic, reassignment,
branching, repeated expressions, a potentially trapping dead division, and a
small counted loop. A fixed RNG seed makes failures reproducible.

This is automated testing over a finite, generated corpus, not a formal proof
of equivalence for all TAC programs.
"""

from __future__ import annotations

import random
import unittest

from engine.loop import optimize_program
from engine.selectors import FixedOrderSelector
from optimizers.registry import STRATEGY_NAMES, apply_strategy
from tac.interpreter import execute
from tac.parser import parse_tac


SEED = 20261029
PROGRAM_COUNT = 80
RANDOM_INPUT_CASES_PER_PROGRAM = 5
BOUNDARY_INPUT_CASES = (
    # Boundary-style inputs cover zero divisors, negatives, equality, and loops.
    {"a": -3, "b": 0, "c": 2, "d": 0, "n": -1},
    {"a": 0, "b": 1, "c": -2, "d": 2, "n": 0},
    {"a": 2, "b": 2, "c": 3, "d": 1, "n": 1},
    {"a": 5, "b": -1, "c": 0, "d": -2, "n": 4},
    {"a": 1, "b": 4, "c": 2, "d": 3, "n": 6},
)


def _input_cases(rng: random.Random) -> tuple[dict[str, int], ...]:
    """Return boundary and seeded-random environments for one program."""
    generated = tuple(
        {
            "a": rng.randint(-5, 5),
            "b": rng.randint(-5, 5),
            "c": rng.randint(-5, 5),
            "d": rng.choice((-3, -2, -1, 0, 1, 2, 3)),
            # Keep test loops bounded while covering zero/negative/positive bounds.
            "n": rng.randint(-2, 6),
        }
        for _ in range(RANDOM_INPUT_CASES_PER_PROGRAM)
    )
    return BOUNDARY_INPUT_CASES + generated


def _generated_program(rng: random.Random, index: int) -> str:
    """Build one valid, terminating TAC program from a deterministic seed."""
    op1 = rng.choice(("+", "-", "*"))
    op2 = rng.choice(("+", "-", "*"))
    literal = rng.randint(0, 7)
    loop_op = rng.choice(("*", "+"))
    branch_op = rng.choice(("<", "<=", "==", "!="))
    # d is an input, so the dead division is an intentional trap when d == 0.
    # It occurs after two prints to make output-before-error observable too.
    return f"""# generated semantic case {index}
t1 = a {op1} b
t2 = t1 {op2} c
print t1
print t2
dead{index} = a / d
a = {literal}
if a {branch_op} b goto LEFT{index}
branch{index} = a + 2
goto JOIN{index}
LEFT{index}:
branch{index} = b + 2
JOIN{index}:
t3 = a + b
t4 = a + b
i = 0
LOOP{index}:
t5 = i {loop_op} 2
i = i + 1
if i < n goto LOOP{index}
print branch{index}
print t3
print t4
print t5
"""


class SemanticFuzzTests(unittest.TestCase):
    def test_seeded_generated_programs_preserve_behavior_for_each_pass(self) -> None:
        rng = random.Random(SEED)
        input_rng = random.Random(SEED + 100)
        for index in range(PROGRAM_COUNT):
            source = _generated_program(rng, index)
            original = parse_tac(source)
            original_text = original.to_tac()
            for inputs in _input_cases(input_rng):
                expected = execute(original, inputs)
                for strategy in STRATEGY_NAMES:
                    with self.subTest(seed=SEED, program=index, strategy=strategy, inputs=inputs):
                        optimized = apply_strategy(original, strategy)
                        actual = execute(optimized, inputs)
                        self.assertEqual(
                            actual,
                            expected,
                            f"strategy={strategy} program={index} inputs={inputs}",
                        )
                        self.assertEqual(original.to_tac(), original_text)

    def test_seeded_generated_programs_preserve_behavior_through_iterative_engine(self) -> None:
        rng = random.Random(SEED + 1)
        input_rng = random.Random(SEED + 200)
        selector = FixedOrderSelector()
        for index in range(24):
            source = _generated_program(rng, index + 1000)
            original = parse_tac(source)
            original_text = original.to_tac()
            for inputs in _input_cases(input_rng):
                with self.subTest(seed=SEED + 1, program=index, inputs=inputs):
                    run = optimize_program(original, selector, program_id=f"fuzz_{index}")
                    self.assertEqual(original.to_tac(), original_text)
                    self.assertEqual(
                        execute(run.final_program, inputs),
                        execute(original, inputs),
                        f"engine program={index} inputs={inputs}",
                    )


if __name__ == "__main__":
    unittest.main()
