"""The eleven-feature rule baseline follows the label policy's ordering."""

from __future__ import annotations

import unittest

from evaluation.label_policy import NO_IMPROVEMENT
from features.extended_extractor import FEATURE_COLUMNS_EXTENDED as FEATURE_COLUMNS_V2
from selection.rule_based import predict_rule_v2, rule_scores_v2


def features(**values):
    row = {name: 0 for name in FEATURE_COLUMNS_V2}
    row["num_statements"] = 6
    row.update(values)
    return row


class RuleV2Tests(unittest.TestCase):
    def test_dead_code_outranks_everything(self) -> None:
        self.assertEqual(
            predict_rule_v2(features(num_dead_stmts=1, num_constants=5, num_mul_by_two_in_loops=3)),
            "dead_code_elimination",
        )

    def test_strength_reduction_uses_the_extended_feature(self) -> None:
        self.assertEqual(
            predict_rule_v2(features(num_loop_ops=6, num_mul_by_two_in_loops=1)),
            "strength_reduction",
        )
        # A loop with no multiply by two is not a strength-reduction opportunity.
        self.assertEqual(predict_rule_v2(features(num_loop_ops=6)), NO_IMPROVEMENT)

    def test_constant_propagation_only_when_nothing_else_helps(self) -> None:
        self.assertEqual(predict_rule_v2(features(num_constant_propagation_uses=3)), "constant_propagation")
        self.assertEqual(
            predict_rule_v2(features(num_constant_propagation_uses=3, num_algebraic_ops=1)),
            "algebraic_simplification",
        )

    def test_safe_repeats_drive_cse(self) -> None:
        self.assertEqual(predict_rule_v2(features(num_safe_repeated_expr=2)), "common_subexpression_elimination")
        self.assertEqual(predict_rule_v2(features(num_repeated_expr=2)), NO_IMPROVEMENT)

    def test_missing_columns_raise(self) -> None:
        with self.assertRaises(KeyError):
            rule_scores_v2({"num_statements": 3})


if __name__ == "__main__":
    unittest.main()
