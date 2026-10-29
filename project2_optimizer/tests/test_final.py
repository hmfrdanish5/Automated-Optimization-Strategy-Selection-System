"""Tests for the final dataset evaluation pieces.

The full 200-row experiment is run by evaluate.py. These tests check the
grouping, metrics, and plot writers without repeating that run.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from dataset.dataset_builder import DATASET_PATH, EXPANDED_DATASET_PATH, FINAL_DATASET_PATH
from dataset.diversity import diversity_report, feature_signature
from dataset.generator import GENERATION_SEED, generate_final_programs
from evaluation.classification import CLASS_ORDER, confusion, metric_bundle
from evaluation.final_experiment import cv_statistics, mean_gains, pearson
from evaluation.folds import grouped_folds
from evaluation.plots import instruction_gains_by_strategy, save_feature_importance, save_gain_boxplot
from features.extractor import FEATURE_COLUMNS, extract_features
from selection.splitting import LabeledExample, split_examples
from selection.config import STRICT_SPLIT
from tac.parser import parse_tac


class FinalDatasetTests(unittest.TestCase):
    def test_final_generator_is_deterministic_and_does_not_replace_old_files(self) -> None:
        first = generate_final_programs(12, GENERATION_SEED)
        second = generate_final_programs(12, GENERATION_SEED)
        self.assertEqual(
            [program.to_tac() for _pid, program, _category in first],
            [program.to_tac() for _pid, program, _category in second],
        )
        self.assertTrue(str(first[0][0]).startswith("fin_"))
        self.assertNotEqual(FINAL_DATASET_PATH, DATASET_PATH)
        self.assertNotEqual(FINAL_DATASET_PATH, EXPANDED_DATASET_PATH)
        signatures = {
            feature_signature(extract_features(program))
            for _pid, program, _category in generate_final_programs()
        }
        self.assertGreaterEqual(len(signatures) / 191, 0.85)

    def test_conflicting_signatures_are_counted(self) -> None:
        features = {column: 1 for column in FEATURE_COLUMNS}
        rows = [
            {"program_id": "a", "best_strategy": "constant_folding", **features},
            {"program_id": "b", "best_strategy": "dead_code_elimination", **features},
        ]
        report = diversity_report(rows)
        self.assertEqual(len(report.inconsistent_signatures), 1)
        self.assertEqual(report.duplicate_rows, 1)
        self.assertEqual(report.unique_feature_vectors, 1)


class GroupEvaluationTests(unittest.TestCase):
    def test_strict_split_and_grouped_folds_keep_signatures_together(self) -> None:
        examples = _varied_examples()
        _train, test, info = split_examples(examples, STRICT_SPLIT)
        self.assertEqual(info.overlap_rows, 0)
        method, folds = grouped_folds(examples, n_splits=5)
        self.assertIn("Fold", method)
        self.assertEqual(len(folds), 5)
        groups = [feature_signature(example.features) for example in examples]
        for train_index, test_index in folds:
            train_groups = {groups[index] for index in train_index}
            self.assertTrue(all(groups[index] not in train_groups for index in test_index))

    def test_metric_bundle_and_confusion_use_the_class_order(self) -> None:
        actual = ["constant_folding", "dead_code_elimination"]
        predicted = ["constant_folding", "constant_folding"]
        bundle = metric_bundle(actual, predicted)
        self.assertEqual(bundle["accuracy"], 0.5)
        matrix = confusion(actual, predicted)
        self.assertEqual(matrix.shape, (len(CLASS_ORDER), len(CLASS_ORDER)))
        self.assertEqual(int(matrix[0, 0]), 1)

    def test_cv_statistics_and_mean_gains(self) -> None:
        rows = [
            {"selector": "rule_based", "fold": 1, "accuracy": 0.5, "macro_f1": 0.4},
            {"selector": "rule_based", "fold": 2, "accuracy": 1.0, "macro_f1": 0.8},
        ]
        summary = cv_statistics(rows)
        mean = next(row for row in summary if row["fold"] == "mean")
        self.assertAlmostEqual(mean["accuracy"], 0.75)
        gains = mean_gains(
            [
                {"selector": "rule_based", "instruction_gain": 2, "weighted_cost_gain": 4},
                {"selector": "rule_based", "instruction_gain": 0, "weighted_cost_gain": 2},
            ]
        )
        self.assertEqual(gains["rule_based"]["instruction_gain"], 1)
        self.assertIsNone(pearson([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(pearson([1, 2, 3], [1, 2, 3]), 1.0)


class PlotTests(unittest.TestCase):
    def test_plot_writers_create_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_feature_importance(
                {"num_dead_stmts": 0.4, "num_constants": 0.1},
                root / "importance.png",
            )
            save_gain_boxplot(
                {"constant_folding": [0, 0, 1], "dead_code_elimination": [2, 3]},
                root / "box.png",
            )
            self.assertTrue((root / "importance.png").stat().st_size > 0)
            self.assertTrue((root / "box.png").stat().st_size > 0)
        gains = instruction_gains_by_strategy(
            [
                {
                    "cf_instruction_gain": 0,
                    "cp_instruction_gain": 0,
                    "alg_instruction_gain": 1,
                    "cse_instruction_gain": 0,
                    "dce_instruction_gain": 2,
                    "sr_instruction_gain": 0,
                }
            ]
        )
        self.assertEqual(gains["dead_code_elimination"], [2])
        self.assertEqual(gains["strength_reduction"], [0])


def _varied_examples() -> list[LabeledExample]:
    program = parse_tac("print x")
    labels = list(CLASS_ORDER)
    examples = []
    for index in range(20):
        features = {column: 0 for column in FEATURE_COLUMNS}
        features["num_statements"] = index + 1
        features["const_ratio"] = 0.0
        # Two rows share a signature so grouping has something to keep together.
        if index == 19:
            features["num_statements"] = 1
        examples.append(
            LabeledExample(f"p{index}", program, features, labels[index % len(labels)])
        )
    return examples


if __name__ == "__main__":
    unittest.main()
