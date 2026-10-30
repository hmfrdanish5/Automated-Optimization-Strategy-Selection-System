"""Tests for pass-aware features in the 11-feature schema."""

from __future__ import annotations

import unittest

from features.extended_extractor import FEATURE_COLUMNS_EXTENDED, extract_extended_features
from features.extractor import FEATURE_COLUMNS, extract_features
from tac.parser import parse_tac


class ExtendedFeatureTests(unittest.TestCase):
    def test_11_includes_original_eight_in_same_order(self) -> None:
        program = parse_tac(
            "x = 1\n"
            "y = x + 2\n"
            "print y\n"
        )
        base = extract_features(program)
        extended = extract_extended_features(program)
        for column in FEATURE_COLUMNS:
            self.assertEqual(base[column], extended[column])
        self.assertEqual(list(extended.keys()), list(FEATURE_COLUMNS_EXTENDED))

    def test_11_extra_features_are_non_negative_integers(self) -> None:
        program = parse_tac("i = 0\nloop:\n  i = i + 1\n  if i > 2 goto end\n  x = i * 2\nend:\nprint x\n")
        extended = extract_extended_features(program)
        for column in FEATURE_COLUMNS_EXTENDED[8:]:
            value = extended[column]
            self.assertGreaterEqual(value, 0)
            self.assertEqual(value, int(value))


if __name__ == "__main__":
    unittest.main()
