"""Review-2 dataset, models, and classifier schema guards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dataset.v2_builder import build_dataset_v2, label_row_v2, v2_dataset_columns
from dataset.v2_loader import load_dataset_v2_rows, pair_v2_examples
from selection.review2_config import V2_DATASET_PATH
from features.extended_extractor import FEATURE_COLUMNS_V2
from features.extractor import FEATURE_COLUMNS
from selection.classifiers import decision_tree, load_model, logistic_regression, save_model
from tac.parser import parse_tac


class Review2DatasetTests(unittest.TestCase):
    def test_v2_builder_has_four_hundred_rows(self) -> None:
        rows = build_dataset_v2()
        self.assertEqual(len(rows), 400)
        self.assertEqual(set(rows[0].keys()), set(v2_dataset_columns()))
        for column in v2_dataset_columns():
            self.assertIn(column, rows[0])

    def test_label_row_v2_matches_eight_feature_extractor(self) -> None:
        program = parse_tac("a = 1 + 2\nprint a\n")
        row = label_row_v2("demo", program)
        for column in FEATURE_COLUMNS:
            self.assertIn(column, row)
        for column in FEATURE_COLUMNS_V2:
            self.assertIn(column, row)


class Review2EvaluationSeparationTests(unittest.TestCase):
    def test_evaluation_does_not_fit_production_models(self) -> None:
        from dataset.v2_builder import write_dataset_v2
        from evaluation.review2_experiment import run_review2_evaluation
        from selection.review2_config import V2_DATASET_PATH

        if not V2_DATASET_PATH.is_file():
            write_dataset_v2()
        with patch(
            "evaluation.review2_experiment.fit_review2_production",
            side_effect=AssertionError("production fit must not run during evaluation"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                summary = run_review2_evaluation(Path(directory))
        self.assertIn("Review-2 evaluation summary", summary)


class PersistedDatasetTests(unittest.TestCase):
    def test_pair_v2_examples_uses_csv_features(self) -> None:
        if not V2_DATASET_PATH.is_file():
            self.skipTest("dataset_v2.csv not generated yet")
        rows = load_dataset_v2_rows()
        examples = pair_v2_examples(rows, FEATURE_COLUMNS)
        self.assertEqual(examples[0].features["num_statements"], rows[0]["num_statements"])


class SchemaGuardTests(unittest.TestCase):
    def test_eight_feature_model_rejects_eleven_columns(self) -> None:
        program = parse_tac("a = 1\nprint a\n")
        row = label_row_v2("demo", program)
        frame = pd.DataFrame([row])
        model = decision_tree(list(FEATURE_COLUMNS)).fit(
            frame[list(FEATURE_COLUMNS)],
            pd.Series(["constant_folding"]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tree.pkl"
            save_model(model, path, {"feature_schema": "v1"})
            loaded = load_model(path)
            self.assertEqual(loaded.feature_columns, list(FEATURE_COLUMNS))
            with self.assertRaises(ValueError):
                loaded.predict(frame[list(FEATURE_COLUMNS_V2)])


if __name__ == "__main__":
    unittest.main()
