"""Grouped splits, duplicate feature vectors, and the expanded dataset."""

from __future__ import annotations

import unittest

from dataset.dataset_builder import DATASET_PATH, build_expanded_dataset
from dataset.diversity import diversity_report, feature_signature, format_diversity_report
from dataset.labeler import label_program
from dataset.examples import load_sample_programs
from dataset.generator import EXPANDED_GENERATED_COUNT, GENERATION_SEED, generate_diverse_programs
from features.extractor import FEATURE_COLUMNS, extract_features
from selection.config import REVIEW_SPLIT, STRICT_SPLIT
from selection.pipeline import load_dataset_frame, load_labeled_examples
from selection.splitting import split_examples
from tac.parser import parse_tac


class GroupedSplitTests(unittest.TestCase):
    def test_identical_feature_vectors_stay_together(self) -> None:
        program = parse_tac("print x")
        shared = {column: 0 for column in FEATURE_COLUMNS}
        other = {column: 0 for column in FEATURE_COLUMNS}
        other["num_statements"] = 4
        examples = []
        for index in range(6):
            features = dict(shared if index < 4 else other)
            label = "dead_code_elimination" if index < 4 else "constant_folding"
            examples.append(_example(f"p{index}", program, features, label))
        _train, test, info = split_examples(examples, STRICT_SPLIT)
        test_ids = {example.program_id for example in test}
        shared_ids = {f"p{index}" for index in range(4)}
        self.assertTrue(test_ids.isdisjoint(shared_ids) or shared_ids <= test_ids)
        self.assertEqual(info.overlap_rows, 0)
        self.assertIn("strict feature-grouped split", info.explanation)

    def test_grouped_split_is_deterministic(self) -> None:
        examples = load_labeled_examples()
        first_train, first_test, _info = split_examples(examples, STRICT_SPLIT)
        second_train, second_test, _info = split_examples(examples, STRICT_SPLIT)
        self.assertEqual(
            [example.program_id for example in first_train],
            [example.program_id for example in second_train],
        )
        self.assertEqual(
            [example.program_id for example in first_test],
            [example.program_id for example in second_test],
        )

    def test_review_split_reports_overlap_and_strict_split_has_none(self) -> None:
        examples = load_labeled_examples()
        _train, _test, review = split_examples(examples, REVIEW_SPLIT)
        _strict_train, _strict_test, strict = split_examples(examples, STRICT_SPLIT)
        self.assertEqual(review.overlap_rows, 7)
        self.assertEqual(strict.overlap_rows, 0)
        self.assertIn("MODE A", review.explanation)
        self.assertIn("MODE B", strict.explanation)


class DiversityTests(unittest.TestCase):
    def test_review_dataset_duplicate_counts(self) -> None:
        frame = load_dataset_frame()
        report = diversity_report(frame.to_dict(orient="records"))
        self.assertEqual(report.row_count, 57)
        self.assertEqual(report.unique_feature_vectors, 34)
        self.assertEqual(report.duplicate_rows, 23)
        text = format_diversity_report(report)
        self.assertIn("Unique feature vectors: 34", text)
        self.assertIn("Duplicate feature rows: 23", text)
        self.assertEqual(len(report.unique_values), len(FEATURE_COLUMNS))
        self.assertIn("num_loop_ops", report.ranges)

    def test_conflicting_labels_are_visible(self) -> None:
        features = {column: 1 for column in FEATURE_COLUMNS}
        rows = [
            {"program_id": "a", "best_strategy": "constant_folding", **features},
            {"program_id": "b", "best_strategy": "dead_code_elimination", **features},
        ]
        report = diversity_report(rows)
        self.assertEqual(len(report.inconsistent_signatures), 1)
        text = format_diversity_report(report)
        self.assertIn("WARNING", text)
        self.assertIn("Feature representation is not sufficient to uniquely determine the label for this signature.", text)

    def test_same_features_can_measure_different_labels(self) -> None:
        killed = parse_tac("t1 = a + b\na = 4\nt2 = a + b\nprint t1\nprint t2\n")
        repeated = parse_tac("t1 = left + right\nt2 = left + right\nprint t1\nprint t2\nprint 0\n")
        rows = []
        for program_id, program in (("expression_redefined", killed), ("safe_repeated_add", repeated)):
            features = extract_features(program)
            labeled = label_program(program)
            rows.append(
                {
                    "program_id": program_id,
                    "best_strategy": labeled.best_strategy,
                    **features,
                }
            )
        self.assertEqual(feature_signature(rows[0]), feature_signature(rows[1]))
        self.assertNotEqual(rows[0]["best_strategy"], rows[1]["best_strategy"])
        report = diversity_report(rows)
        self.assertEqual(len(report.inconsistent_signatures), 1)
        text = format_diversity_report(report)
        self.assertIn("WARNING", text)
        self.assertIn(
            "Feature representation is not sufficient to uniquely determine the label for this signature.",
            text,
        )


class ExpandedDatasetTests(unittest.TestCase):
    def test_diverse_programs_are_deterministic_and_varied(self) -> None:
        first = generate_diverse_programs()
        second = generate_diverse_programs()
        self.assertEqual(
            [program.to_tac() for _program_id, program, _category in first],
            [program.to_tac() for _program_id, program, _category in second],
        )
        self.assertEqual(len(first), EXPANDED_GENERATED_COUNT)
        signatures = {
            feature_signature(extract_features(program))
            for _program_id, program, _category in first
        }
        self.assertGreaterEqual(len(signatures), 80)
        total = len(load_sample_programs()) + len(first)
        self.assertGreaterEqual(total, 100)
        self.assertLessEqual(total, 120)

    def test_expanded_labels_are_measured_and_consistent(self) -> None:
        rows = build_expanded_dataset()
        self.assertGreaterEqual(len(rows), 100)
        self.assertLessEqual(len(rows), 120)
        self.assertNotEqual(DATASET_PATH.name, "dataset_expanded.csv")
        report = diversity_report(rows)
        self.assertGreaterEqual(report.unique_feature_vectors, 70)
        text = format_diversity_report(report)
        if report.inconsistent_signatures:
            self.assertIn("WARNING", text)
        else:
            self.assertIn("consistent labels", text)
        self.assertIn("best_strategy", rows[0])
        self.assertTrue(rows[0]["program_id"])
        self.assertEqual(GENERATION_SEED, 42)


def _example(program_id, program, features, label):
    from selection.splitting import LabeledExample

    return LabeledExample(program_id, program, features, label)


if __name__ == "__main__":
    unittest.main()
