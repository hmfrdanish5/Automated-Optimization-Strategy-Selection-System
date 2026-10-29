"""Tests for the Review-2 extended feature schema."""

from __future__ import annotations

import unittest

from features.extended_extractor import FEATURE_COLUMNS_V2, extract_features_v2
from features.extractor import FEATURE_COLUMNS, extract_features
from tac.parser import parse_tac


class ExtendedFeatureTests(unittest.TestCase):
    def test_v2_includes_original_eight_in_same_order(self) -> None:
        program = parse_tac(
            "x = 1\n"
            "y = x + 2\n"
            "print y\n"
        )
        base = extract_features(program)
        extended = extract_features_v2(program)
        for column in FEATURE_COLUMNS:
            self.assertEqual(base[column], extended[column])
        self.assertEqual(list(extended.keys()), list(FEATURE_COLUMNS_V2))

    def test_v2_extra_features_are_non_negative_integers(self) -> None:
        program = parse_tac("i = 0\nloop:\n  i = i + 1\n  if i > 2 goto end\n  x = i * 2\nend:\nprint x\n")
        extended = extract_features_v2(program)
        for column in FEATURE_COLUMNS_V2[8:]:
            value = extended[column]
            self.assertGreaterEqual(value, 0)
            self.assertEqual(value, int(value))


if __name__ == "__main__":
    unittest.main()
