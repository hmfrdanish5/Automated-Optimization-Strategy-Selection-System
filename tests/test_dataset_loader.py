"""Persisted optimization_dataset.csv loading and validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset.extended_builder import write_dataset
from dataset.loader import (
    DatasetFormatError,
    DatasetMismatchError,
    load_dataset_rows,
    pair_dataset_examples,
    programs_for_dataset,
    validate_row_against_program,
)
from dataset.extended_builder import dataset_columns
from evaluation.experiment import run_model_evaluation
from features.extended_extractor import FEATURE_COLUMNS_EXTENDED
from features.extractor import FEATURE_COLUMNS
from selection.model_config import OPTIMIZATION_DATASET_PATH


class DatasetLoaderTests(unittest.TestCase):
    def test_load_reads_persisted_csv(self) -> None:
        if not OPTIMIZATION_DATASET_PATH.is_file():
            write_dataset()
        rows_disk = load_dataset_rows()
        self.assertEqual(len(rows_disk), 400)
        examples = pair_dataset_examples(rows_disk, FEATURE_COLUMNS)
        self.assertEqual(len(examples), 400)
        self.assertEqual(examples[0].program_id, str(rows_disk[0]["program_id"]))

    def test_evaluation_does_not_rebuild_dataset(self) -> None:
        if not OPTIMIZATION_DATASET_PATH.is_file():
            write_dataset()
        with patch("dataset.extended_builder.build_dataset", side_effect=AssertionError("must not rebuild")):
            with tempfile.TemporaryDirectory() as directory:
                summary = run_model_evaluation(Path(directory))
        self.assertIn("Selector evaluation summary", summary)

    def test_train_reads_back_written_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimization_dataset.csv"
            write_dataset(path)
            rows = load_dataset_rows(path)
            self.assertEqual(len(rows), 400)
            original_schema = pair_dataset_examples(rows, FEATURE_COLUMNS)
            extended_schema = pair_dataset_examples(rows, FEATURE_COLUMNS_EXTENDED)
            self.assertEqual([example.program_id for example in original_schema], [example.program_id for example in extended_schema])


class DatasetValidationTests(unittest.TestCase):
    def test_rejects_fractional_integer_column(self) -> None:
        if not OPTIMIZATION_DATASET_PATH.is_file():
            write_dataset()
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.csv"
            text = OPTIMIZATION_DATASET_PATH.read_text(encoding="utf-8").splitlines()
            header = text[0]
            first_data = text[1].split(",")
            # num_statements is column index 1 after program_id
            first_data[1] = "3.5"
            bad.write_text("\n".join([header, ",".join(first_data)] + text[2:]), encoding="utf-8")
            with self.assertRaises(DatasetFormatError):
                load_dataset_rows(bad)

    def test_rejects_corrupted_gain_field(self) -> None:
        if not OPTIMIZATION_DATASET_PATH.is_file():
            write_dataset()
        rows = load_dataset_rows()
        program_id, program = programs_for_dataset()[0]
        row = dict(rows[0])
        gain_col = next(c for c in dataset_columns() if c.endswith("instruction_gain"))
        row[gain_col] = int(row[gain_col]) + 999
        with self.assertRaises(DatasetMismatchError):
            validate_row_against_program(row, program)


if __name__ == "__main__":
    unittest.main()
