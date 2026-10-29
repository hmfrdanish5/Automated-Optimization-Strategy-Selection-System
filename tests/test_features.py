"""Tests for the eight TAC feature definitions."""

from __future__ import annotations

import unittest

from dataset.examples import load_sample_programs
from features.extractor import (
    FEATURE_COLUMNS,
    extract_features,
    feature_vector,
    num_algebraic_ops,
    num_constants,
    num_dead_stmts,
    num_loop_ops,
    num_repeated_expr,
)
from tac.parser import parse_tac


def _features(source: str) -> dict[str, float]:
    return extract_features(parse_tac(source))


class FeatureDefinitionTests(unittest.TestCase):
    def test_column_order_is_fixed(self) -> None:
        self.assertEqual(
            FEATURE_COLUMNS,
            [
                "num_statements",
                "num_constants",
                "num_repeated_expr",
                "num_dead_stmts",
                "num_algebraic_ops",
                "num_loop_ops",
                "num_tmp_vars",
                "const_ratio",
            ],
        )

    def test_empty_program_has_zero_features(self) -> None:
        features = _features("")
        self.assertEqual(list(features), FEATURE_COLUMNS)
        self.assertEqual(feature_vector(parse_tac("")), [0.0] * 8)

    def test_statements_include_labels_jumps_and_prints(self) -> None:
        features = _features("L1:\ngoto L1\nprint 1\n")
        self.assertEqual(features["num_statements"], 3)

    def test_constants_count_only_safe_evaluable_arithmetic(self) -> None:
        source = "\n".join(
            [
                "t1 = 10 + 20",
                "t2 = 8 / 2",
                "t3 = 8 / 0",
                "t4 = 8 % 0",
                "t5 = - 10",
                "x = 10",
                "y = x + 5",
                "if 1 < 2 goto L1",
                "L1:",
                "print y",
            ]
        )
        self.assertEqual(num_constants(parse_tac(source)), 3)

    def test_repeated_expressions_count_copies_beyond_the_first(self) -> None:
        source = "\n".join(
            [
                "t1 = a + b",
                "t2 = a + b",
                "t3 = a + b",
                "t4 = b + a",
                "t5 = a + 00",
                "t6 = a + 0",
                "print t1",
            ]
        )
        # a + b has two extra copies. a + 0 and a + 00 are one extra pair.
        # b + a is a different expression.
        self.assertEqual(num_repeated_expr(parse_tac(source)), 3)

    def test_repeated_count_ignores_whether_cse_is_safe(self) -> None:
        source = "\n".join(
            [
                "t1 = a + b",
                "a = 4",
                "t2 = a + b",
                "print t1",
                "print t2",
            ]
        )
        self.assertEqual(_features(source)["num_repeated_expr"], 1)

    def test_direct_dead_assignment_and_live_print(self) -> None:
        features = _features("x = 10\ny = 20\nprint y\n")
        self.assertEqual(features["num_dead_stmts"], 1)

    def test_dead_chain_is_removed_across_iterations(self) -> None:
        features = _features("x = 10\nz = x + 1\nprint 0\n")
        self.assertEqual(features["num_dead_stmts"], 2)

    def test_reassignment_makes_the_earlier_value_dead(self) -> None:
        features = _features("x = 1\nx = 2\nprint x\n")
        self.assertEqual(features["num_dead_stmts"], 1)

    def test_loop_carried_value_is_live(self) -> None:
        source = "\n".join(
            [
                "L1:",
                "print x",
                "x = 1",
                "if a < b goto L1",
            ]
        )
        self.assertEqual(num_dead_stmts(parse_tac(source)), 0)

    def test_branch_can_keep_the_first_assignment_live(self) -> None:
        source = "\n".join(
            [
                "x = 1",
                "if a < b goto L1",
                "x = 2",
                "L1:",
                "print x",
            ]
        )
        self.assertEqual(num_dead_stmts(parse_tac(source)), 0)

    def test_algebraic_patterns_match_the_documented_list(self) -> None:
        source = "\n".join(
            [
                "t1 = a + 0",
                "t2 = 0 + a",
                "t3 = a - 0",
                "t4 = 0 - a",
                "t5 = a * 1",
                "t6 = 1 * a",
                "t7 = a / 1",
                "t8 = a * 0",
                "t9 = 0 * a",
                "t10 = a / 0",
                "print t1",
            ]
        )
        # 0 - a and a / 0 are intentionally not algebraic opportunities.
        self.assertEqual(num_algebraic_ops(parse_tac(source)), 8)

    def test_loop_count_includes_the_body_once(self) -> None:
        source = "\n".join(
            [
                "i = 0",
                "L1:",
                "t1 = i + 1",
                "if i < n goto L1",
                "goto END",
                "END:",
                "print t1",
            ]
        )
        # Body is L1, t1 = i + 1, and the conditional jump. goto END is forward.
        self.assertEqual(num_loop_ops(parse_tac(source)), 3)

    def test_nested_loops_do_not_count_an_instruction_twice(self) -> None:
        source = "\n".join(
            [
                "L1:",
                "L2:",
                "t1 = a + b",
                "if a < b goto L2",
                "if c < d goto L1",
                "print t1",
            ]
        )
        self.assertEqual(num_loop_ops(parse_tac(source)), 5)

    def test_temporaries_are_unique_t_names(self) -> None:
        features = _features("t1 = a + b\nt2 = t1\ntemp = t1\nprint t2\n")
        self.assertEqual(features["num_tmp_vars"], 2)

    def test_const_ratio_is_zero_when_there_are_no_statements(self) -> None:
        self.assertEqual(_features("")["const_ratio"], 0.0)

    def test_extraction_does_not_change_the_program(self) -> None:
        program = parse_tac("x = 1\nprint x\n")
        before = program.to_tac()
        extract_features(program)
        self.assertEqual(program.to_tac(), before)


class SampleProgramTests(unittest.TestCase):
    """Expected vectors were calculated from the feature rules, by hand."""

    def test_sample_ids_and_feature_vectors(self) -> None:
        expected = {
            "constant_heavy": {
                "num_statements": 6,
                "num_constants": 3,
                "num_repeated_expr": 0,
                "num_dead_stmts": 0,
                "num_algebraic_ops": 0,
                "num_loop_ops": 0,
                "num_tmp_vars": 4,
                "const_ratio": 0.5,
            },
            "propagation_ready": {
                "num_statements": 4,
                "num_constants": 0,
                "num_repeated_expr": 0,
                "num_dead_stmts": 0,
                "num_algebraic_ops": 0,
                "num_loop_ops": 0,
                "num_tmp_vars": 0,
                "const_ratio": 0.0,
            },
            "algebraic_heavy": {
                "num_statements": 12,
                "num_constants": 0,
                "num_repeated_expr": 0,
                "num_dead_stmts": 0,
                "num_algebraic_ops": 6,
                "num_loop_ops": 0,
                "num_tmp_vars": 6,
                "const_ratio": 0.0,
            },
            "cse_heavy": {
                "num_statements": 8,
                "num_constants": 0,
                "num_repeated_expr": 2,
                "num_dead_stmts": 0,
                "num_algebraic_ops": 0,
                "num_loop_ops": 0,
                "num_tmp_vars": 4,
                "const_ratio": 0.0,
            },
            "dead_code_heavy": {
                "num_statements": 4,
                "num_constants": 0,
                "num_repeated_expr": 0,
                "num_dead_stmts": 2,
                "num_algebraic_ops": 0,
                "num_loop_ops": 0,
                "num_tmp_vars": 0,
                "const_ratio": 0.0,
            },
            "loop_heavy": {
                "num_statements": 7,
                "num_constants": 0,
                "num_repeated_expr": 0,
                "num_dead_stmts": 0,
                "num_algebraic_ops": 0,
                "num_loop_ops": 4,
                "num_tmp_vars": 1,
                "const_ratio": 0.0,
            },
            "mixed": {
                "num_statements": 7,
                "num_constants": 1,
                "num_repeated_expr": 1,
                "num_dead_stmts": 3,
                "num_algebraic_ops": 1,
                "num_loop_ops": 0,
                "num_tmp_vars": 4,
                "const_ratio": 1 / 7,
            },
            "straight_line": {
                "num_statements": 3,
                "num_constants": 0,
                "num_repeated_expr": 0,
                "num_dead_stmts": 0,
                "num_algebraic_ops": 0,
                "num_loop_ops": 0,
                "num_tmp_vars": 2,
                "const_ratio": 0.0,
            },
            "expression_redefined": {
                "num_statements": 5,
                "num_constants": 0,
                "num_repeated_expr": 1,
                "num_dead_stmts": 0,
                "num_algebraic_ops": 0,
                "num_loop_ops": 0,
                "num_tmp_vars": 2,
                "const_ratio": 0.0,
            },
        }
        loaded = load_sample_programs()
        self.assertEqual([program_id for program_id, _ in loaded], list(expected))
        for program_id, program in loaded:
            features = extract_features(program)
            for name, value in expected[program_id].items():
                self.assertAlmostEqual(features[name], value, places=7, msg=f"{program_id}.{name}")


if __name__ == "__main__":
    unittest.main()
