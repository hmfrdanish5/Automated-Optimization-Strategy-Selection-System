"""Persisted dataset_v2.csv loading and validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset.v2_builder import write_dataset_v2
from dataset.v2_loader import (
    DatasetV2FormatError,
    DatasetV2MismatchError,
    load_dataset_v2_rows,
    pair_v2_examples,
    programs_for_dataset_v2,
    validate_row_against_program,
)
from dataset.v2_builder import v2_dataset_columns
from evaluation.review2_experiment import run_review2_evaluation
from features.extended_extractor import FEATURE_COLUMNS_V2
from features.extractor import FEATURE_COLUMNS
from selection.review2_config import V2_DATASET_PATH


class DatasetV2LoaderTests(unittest.TestCase):
    def test_load_reads_persisted_csv(self) -> None:
        if not V2_DATASET_PATH.is_file():
            write_dataset_v2()
        rows_disk = load_dataset_v2_rows()
        self.assertEqual(len(rows_disk), 400)
        examples = pair_v2_examples(rows_disk, FEATURE_COLUMNS)
        self.assertEqual(len(examples), 400)
        self.assertEqual(examples[0].program_id, str(rows_disk[0]["program_id"]))

    def test_evaluation_does_not_rebuild_dataset(self) -> None:
        if not V2_DATASET_PATH.is_file():
            write_dataset_v2()
        with patch("dataset.v2_builder.build_dataset_v2", side_effect=AssertionError("must not rebuild")):
            with tempfile.TemporaryDirectory() as directory:
                summary = run_review2_evaluation(Path(directory))
        self.assertIn("Review-2 evaluation summary", summary)

    def test_train_reads_back_written_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset_v2.csv"
            write_dataset_v2(path)
            rows = load_dataset_v2_rows(path)
            self.assertEqual(len(rows), 400)
            v1 = pair_v2_examples(rows, FEATURE_COLUMNS)
            v2 = pair_v2_examples(rows, FEATURE_COLUMNS_V2)
            self.assertEqual([example.program_id for example in v1], [example.program_id for example in v2])


class DatasetV2ValidationTests(unittest.TestCase):
    def test_rejects_fractional_integer_column(self) -> None:
        if not V2_DATASET_PATH.is_file():
            write_dataset_v2()
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.csv"
            text = V2_DATASET_PATH.read_text(encoding="utf-8").splitlines()
            header = text[0]
            first_data = text[1].split(",")
            # num_statements is column index 1 after program_id
            first_data[1] = "3.5"
            bad.write_text("\n".join([header, ",".join(first_data)] + text[2:]), encoding="utf-8")
            with self.assertRaises(DatasetV2FormatError):
                load_dataset_v2_rows(bad)

    def test_rejects_corrupted_gain_field(self) -> None:
        if not V2_DATASET_PATH.is_file():
            write_dataset_v2()
        rows = load_dataset_v2_rows()
        program_id, program = programs_for_dataset_v2()[0]
        row = dict(rows[0])
        gain_col = next(c for c in v2_dataset_columns() if c.endswith("instruction_gain"))
        row[gain_col] = int(row[gain_col]) + 999
        with self.assertRaises(DatasetV2MismatchError):
            validate_row_against_program(row, program)


if __name__ == "__main__":
    unittest.main()
