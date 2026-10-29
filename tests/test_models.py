"""Tests for feature selection, training, probabilities, and model files."""

from __future__ import annotations

from collections import Counter
import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
import pandas as pd

from evaluation.classification import CLASS_ORDER, accuracy, confusion
from evaluation.label_policy import NO_IMPROVEMENT
from features.extractor import FEATURE_COLUMNS
from optimizers.registry import STRATEGY_NAMES
from selection.classifiers import decision_tree, load_model, random_forest, save_model
from selection.config import MODEL_RANDOM_STATE, TEST_SIZE
from selection.pipeline import (
    feature_matrix,
    load_dataset_frame,
    load_labeled_examples,
    split_examples,
    target_labels,
    train_review_models,
)
from selection.rule_based import predict_rule, rule_scores


class DatasetSelectionTests(unittest.TestCase):
    def test_frame_has_features_and_the_label_column(self) -> None:
        frame = load_dataset_frame()
        self.assertEqual(len(frame), 57)
        features = feature_matrix(frame)
        labels = target_labels(frame)
        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertEqual(labels.name, "best_strategy")
        self.assertNotIn("best_strategy", features.columns)
        for column in features.columns:
            self.assertNotIn("gain", column)
            self.assertNotIn("weighted", column)

    def test_examples_follow_the_csv_ids(self) -> None:
        frame = load_dataset_frame()
        examples = load_labeled_examples(frame)
        self.assertEqual([example.program_id for example in examples], list(frame["program_id"]))
        self.assertEqual(list(examples[0].features), FEATURE_COLUMNS)


class RuleTests(unittest.TestCase):
    def test_dead_statements_score_dead_code_elimination(self) -> None:
        features = {column: 0.0 for column in FEATURE_COLUMNS}
        features["num_statements"] = 4
        features["num_dead_stmts"] = 2
        self.assertEqual(predict_rule(features), "dead_code_elimination")

    def test_zero_features_are_no_improvement(self) -> None:
        features = {column: 0.0 for column in FEATURE_COLUMNS}
        self.assertEqual(predict_rule(features), "no_improvement")
        self.assertTrue(all(score == 0 for score in rule_scores(features).values()))

    def test_equal_scores_use_priority_order(self) -> None:
        features = {column: 0.0 for column in FEATURE_COLUMNS}
        features["num_statements"] = 2
        features["num_constants"] = 2
        features["num_dead_stmts"] = 2
        self.assertEqual(rule_scores(features)["constant_folding"], rule_scores(features)["dead_code_elimination"])
        self.assertEqual(predict_rule(features), "constant_folding")


class TrainingTests(unittest.TestCase):
    def test_split_sizes_and_stratification_are_reported(self) -> None:
        examples = load_labeled_examples()
        train, test, info = split_examples(examples)
        self.assertEqual(len(train) + len(test), len(examples))
        self.assertGreater(len(test), 0)
        self.assertLess(len(test), len(train))
        self.assertAlmostEqual(len(test) / len(examples), TEST_SIZE, delta=0.05)
        self.assertIn("stratified", info.explanation)
        if info.stratified:
            train_counts = Counter(example.label for example in train)
            test_counts = Counter(example.label for example in test)
            for label, count in Counter(example.label for example in examples).items():
                if count >= 2:
                    self.assertGreaterEqual(train_counts[label] + test_counts[label], count)

    def test_models_predict_probabilities_and_reload(self) -> None:
        examples = load_labeled_examples()
        train, test, _info = split_examples(examples)
        features = pd.DataFrame([example.features for example in train], columns=FEATURE_COLUMNS)
        labels = pd.Series([example.label for example in train])
        test_features = pd.DataFrame([example.features for example in test], columns=FEATURE_COLUMNS)

        first_forest = random_forest().fit(features, labels)
        second_forest = random_forest().fit(features, labels)
        self.assertEqual(first_forest.predict(test_features), second_forest.predict(test_features))
        self.assertEqual(first_forest.estimator.random_state, MODEL_RANDOM_STATE)

        tree = decision_tree().fit(features, labels)
        probabilities = first_forest.predict_proba(test_features)
        self.assertEqual(len(probabilities), len(test))
        for row, predicted in zip(probabilities, first_forest.predict(test_features)):
            self.assertAlmostEqual(sum(row.values()), 1.0, places=6)
            self.assertEqual(predicted, max(row, key=row.get))
            self.assertIn(predicted, row)

        one = first_forest.predict_one(test[0].features)
        self.assertAlmostEqual(one.confidence, one.probabilities[one.strategy])
        self.assertIn("num_statements", tree.export_tree_text())

        matrix = confusion(
            [example.label for example in test],
            first_forest.predict(test_features),
        )
        self.assertEqual(matrix.shape, (len(CLASS_ORDER), len(CLASS_ORDER)))
        self.assertGreaterEqual(accuracy([example.label for example in test], first_forest.predict(test_features)), 0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forest.pkl"
            save_model(first_forest, path)
            reloaded = load_model(path)
        self.assertEqual(reloaded.predict(test_features), first_forest.predict(test_features))

    def test_prediction_requires_feature_column_order(self) -> None:
        examples = load_labeled_examples()
        train, _test, _info = split_examples(examples)
        features = pd.DataFrame([example.features for example in train], columns=FEATURE_COLUMNS)
        labels = pd.Series([example.label for example in train])
        forest = random_forest().fit(features, labels)
        shuffled = features[list(reversed(FEATURE_COLUMNS))]
        with self.assertRaises(ValueError):
            forest.predict(shuffled)
        probabilities = forest.predict_proba(features.iloc[:1])
        self.assertAlmostEqual(sum(probabilities[0].values()), 1.0, places=6)
        self.assertEqual(forest.predict(features.iloc[:1])[0], max(probabilities[0], key=probabilities[0].get))
        self.assertTrue(set(probabilities[0]) <= set(labels.unique()) | {"no_improvement"})

    def test_loading_a_different_sklearn_version_warns(self) -> None:
        examples = load_labeled_examples()
        features = pd.DataFrame([example.features for example in examples], columns=FEATURE_COLUMNS)
        labels = pd.Series([example.label for example in examples])
        forest = random_forest().fit(features, labels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forest.pkl"
            save_model(forest, path)
            payload = __import__("joblib").load(path)
            payload["sklearn_version"] = "0.0.0"
            __import__("joblib").dump(payload, path)
            with self.assertWarns(UserWarning):
                load_model(path)

    def test_both_models_stay_inside_the_known_classes_and_round_trip(self) -> None:
        """A unique probability maximum must match predict.

        scikit-learn breaks probability ties with its own class order. This
        test checks argmax only when one class is strictly highest.
        """
        known = set(STRATEGY_NAMES) | {NO_IMPROVEMENT}
        examples = load_labeled_examples()
        train, test, _info = split_examples(examples)
        features = pd.DataFrame([example.features for example in train], columns=FEATURE_COLUMNS)
        labels = pd.Series([example.label for example in train])
        test_features = pd.DataFrame([example.features for example in test], columns=FEATURE_COLUMNS)
        with tempfile.TemporaryDirectory() as directory:
            for factory in (decision_tree, random_forest):
                first = factory().fit(features, labels)
                second = factory().fit(features, labels)
                self.assertEqual(first.predict(test_features), second.predict(test_features))
                predictions = first.predict(test_features)
                probabilities = first.predict_proba(test_features)
                for predicted, row in zip(predictions, probabilities):
                    self.assertIn(predicted, known)
                    self.assertTrue(set(row) <= known)
                    self.assertAlmostEqual(sum(row.values()), 1.0, places=6)
                    winners = [name for name, value in row.items() if value == max(row.values())]
                    if len(winners) == 1:
                        self.assertEqual(predicted, winners[0])
                path = Path(directory) / f"{first.name}.pkl"
                save_model(first, path)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reloaded = load_model(path)
                self.assertFalse(
                    any("scikit-learn" in str(item.message) for item in caught)
                )
                self.assertEqual(reloaded.predict(test_features), predictions)

    def test_a_missing_or_different_sklearn_version_warns(self) -> None:
        examples = load_labeled_examples()
        features = pd.DataFrame([example.features for example in examples], columns=FEATURE_COLUMNS)
        labels = pd.Series([example.label for example in examples])
        forest = random_forest().fit(features, labels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forest.pkl"
            save_model(forest, path)
            payload = joblib.load(path)
            self.assertEqual(payload["sklearn_version"], __import__("sklearn").__version__)
            payload.pop("sklearn_version")
            joblib.dump(payload, path)
            with warnings.catch_warnings(record=True) as missing:
                warnings.simplefilter("always")
                load_model(path)
                load_model(path)
            missing_messages = [str(item.message) for item in missing if issubclass(item.category, UserWarning)]
            self.assertEqual(len(missing_messages), 1)
            self.assertIn("no recorded scikit-learn version", missing_messages[0])
            payload["sklearn_version"] = "0.0.0"
            joblib.dump(payload, path)
            with warnings.catch_warnings(record=True) as mismatched:
                warnings.simplefilter("always")
                load_model(path)
            mismatched_messages = [str(item.message) for item in mismatched if issubclass(item.category, UserWarning)]
            self.assertTrue(any("0.0.0" in message for message in mismatched_messages))

    def test_review_training_uses_only_the_eight_features(self) -> None:
        result = train_review_models(save=False)
        self.assertEqual(len(result["examples"]), 57)
        self.assertTrue(result["split"].stratified or result["split"].explanation)
        self.assertEqual(len(result["practical"]), len(result["test"]))
        leaked = [column for column in FEATURE_COLUMNS if "gain" in column or column == "best_strategy"]
        self.assertEqual(leaked, [])
        self.assertIn("no_improvement", CLASS_ORDER)


if __name__ == "__main__":
    unittest.main()
