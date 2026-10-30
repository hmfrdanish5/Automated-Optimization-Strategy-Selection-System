"""Dataset generation, model evaluation, and classifier schema guards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dataset.extended_builder import build_dataset, label_row_extended, dataset_columns
from dataset.loader import load_dataset_rows, pair_dataset_examples
from selection.model_config import OPTIMIZATION_DATASET_PATH
from features.extended_extractor import FEATURE_COLUMNS_EXTENDED
from features.extractor import FEATURE_COLUMNS
from selection.classifiers import decision_tree, load_model, logistic_regression, save_model
from tac.parser import parse_tac


class DatasetBuilderTests(unittest.TestCase):
    def test_11_builder_has_four_hundred_rows(self) -> None:
        rows = build_dataset()
        self.assertEqual(len(rows), 400)
        self.assertEqual(set(rows[0].keys()), set(dataset_columns()))
        for column in dataset_columns():
            self.assertIn(column, rows[0])

    def test_label_row_extended_matches_eight_feature_extractor(self) -> None:
        program = parse_tac("a = 1 + 2\nprint a\n")
        row = label_row_extended("demo", program)
        for column in FEATURE_COLUMNS:
            self.assertIn(column, row)
        for column in FEATURE_COLUMNS_EXTENDED:
            self.assertIn(column, row)


class EvaluationSeparationTests(unittest.TestCase):
    def test_evaluation_does_not_fit_production_models(self) -> None:
        from dataset.extended_builder import write_dataset
        from evaluation.experiment import run_model_evaluation
        from selection.model_config import OPTIMIZATION_DATASET_PATH

        if not OPTIMIZATION_DATASET_PATH.is_file():
            write_dataset()
        with patch(
            "evaluation.experiment.fit_production_models",
            side_effect=AssertionError("production fit must not run during evaluation"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                summary = run_model_evaluation(Path(directory))
        self.assertIn("Selector evaluation summary", summary)


class PersistedDatasetTests(unittest.TestCase):
    def test_pair_dataset_examples_uses_csv_features(self) -> None:
        if not OPTIMIZATION_DATASET_PATH.is_file():
            self.skipTest("optimization_dataset.csv not generated yet")
        rows = load_dataset_rows()
        examples = pair_dataset_examples(rows, FEATURE_COLUMNS)
        self.assertEqual(examples[0].features["num_statements"], rows[0]["num_statements"])


class SchemaGuardTests(unittest.TestCase):
    def test_eight_feature_model_rejects_eleven_columns(self) -> None:
        program = parse_tac("a = 1\nprint a\n")
        row = label_row_extended("demo", program)
        frame = pd.DataFrame([row])
        model = decision_tree(list(FEATURE_COLUMNS)).fit(
            frame[list(FEATURE_COLUMNS)],
            pd.Series(["constant_folding"]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tree.pkl"
            save_model(model, path, {"feature_schema": "8_features"})
            loaded = load_model(path)
            self.assertEqual(loaded.feature_columns, list(FEATURE_COLUMNS))
            with self.assertRaises(ValueError):
                loaded.predict(frame[list(FEATURE_COLUMNS_EXTENDED)])


if __name__ == "__main__":
    unittest.main()
