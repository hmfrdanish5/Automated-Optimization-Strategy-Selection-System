"""Feature extraction for source-generated temporaries."""

from __future__ import annotations

import unittest

from features.extractor import extract_features, num_tmp_vars
from source_frontend.compiler import compile_source
from tac.parser import parse_tac


class TemporaryVariableFeatureTests(unittest.TestCase):
    def test_classic_t_names_are_counted(self) -> None:
        program = parse_tac("t1 = a + b\nt2 = t1\nprint t2\n")
        self.assertEqual(num_tmp_vars(program), 2)

    def test_src_tmp_names_are_counted(self) -> None:
        program = parse_tac("__src_tmp_1 = a + b\nc = __src_tmp_1\nprint c\n")
        self.assertEqual(num_tmp_vars(program), 1)

    def test_unique_temp_counted_once_when_reused(self) -> None:
        program = parse_tac(
            "__src_tmp_1 = a + b\n__src_tmp_2 = __src_tmp_1 + c\nprint __src_tmp_2\n"
        )
        self.assertEqual(num_tmp_vars(program), 2)

    def test_ordinary_identifiers_are_not_temporaries(self) -> None:
        program = parse_tac("total = a + b\ntemp = total\na = 1\nprint temp\n")
        self.assertEqual(num_tmp_vars(program), 0)

    def test_source_compile_to_features_counts_generated_temp(self) -> None:
        source = """
        int a = 10;
        int b = 20;
        int c = a + b * 2;
        print(c);
        """
        result = compile_source(source)
        self.assertTrue(result.ok, result.errors)
        features = extract_features(result.program)
        self.assertEqual(features["num_tmp_vars"], 2)
        self.assertIn("__src_tmp_1", result.tac_text)
        self.assertIn("__src_tmp_2", result.tac_text)


if __name__ == "__main__":
    unittest.main()
