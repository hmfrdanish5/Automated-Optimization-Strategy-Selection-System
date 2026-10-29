"""Tests for the Streamlit demo adapter (not the Streamlit renderer)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.selectors import runtime_selectors
from selection.config import PRODUCTION_FOREST_PATH, PRODUCTION_TREE_PATH
from selection.review2_config import (
    REVIEW2_PRODUCTION_FOREST_V2,
    REVIEW2_PRODUCTION_LOGISTIC_V2,
    REVIEW2_PRODUCTION_MLP_V2,
    REVIEW2_PRODUCTION_TREE_V2,
)
from ui.demo_service import SELECTOR_ORDER
from tac.parser import parse_tac
from selection.classifiers import StrategyClassifier
from tac.interpreter import execute
from ui.demo_service import (
    CLASS_ORDER,
    DEMO_SAMPLE_IDS,
    DemoError,
    class_score_rows,
    compare_all_selectors,
    ensure_production_models,
    iteration_history_rows,
    load_demo_samples,
    load_random_tac,
    run_demo_optimization,
)
from ui.random_tac import generate_random_tac


ROOT = Path(__file__).resolve().parents[1]


class DemoServiceTests(unittest.TestCase):
    def test_app_module_imports(self) -> None:
        spec = importlib.util.spec_from_file_location("streamlit_app", ROOT / "app.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "main"))

    def test_random_tac_is_reproducible_and_observable(self) -> None:
        first = generate_random_tac(7)
        self.assertEqual(load_random_tac(7), first)
        self.assertNotEqual(generate_random_tac(7), generate_random_tac(8))
        seen = set()
        for seed in range(24):
            text = generate_random_tac(seed)
            seen.add(text)
            program = parse_tac(text)
            self.assertIn("print", text)
            result = execute(program)
            self.assertIsNone(result.error)
            self.assertGreater(len(result.prints), 0)
        self.assertGreater(len(seen), 1)

    def test_demo_samples_load(self) -> None:
        samples = load_demo_samples()
        for program_id in DEMO_SAMPLE_IDS:
            self.assertIn(program_id, samples.by_id)
            self.assertTrue(samples.by_id[program_id].strip())

    def test_empty_tac_raises_demo_error(self) -> None:
        with self.assertRaises(DemoError):
            run_demo_optimization("   \n", "rule_based")

    def test_malformed_tac_raises_demo_error(self) -> None:
        with self.assertRaises(DemoError):
            run_demo_optimization("not valid tac !!!", "rule_based")

    @unittest.skipUnless(
        PRODUCTION_TREE_PATH.is_file() and PRODUCTION_FOREST_PATH.is_file(),
        "production models required",
    )
    def test_optimization_uses_production_selectors(self) -> None:
        samples = load_demo_samples()
        source = samples.get("dead_code_heavy")
        result = run_demo_optimization(source, "random_forest")
        self.assertTrue(result.uses_production_models)
        self.assertEqual(result.original_tac, parse_tac(source).to_tac())
        self.assertGreaterEqual(result.run.candidate_attempts, 1)
        self.assertTrue(iteration_history_rows(result.run))

    @unittest.skipUnless(
        PRODUCTION_TREE_PATH.is_file() and PRODUCTION_FOREST_PATH.is_file(),
        "production models required",
    )
    def test_selector_keys_map_to_runtime_selectors(self) -> None:
        samples = load_demo_samples()
        source = samples.get("constant_heavy")
        for key in ("rule_based", "decision_tree", "random_forest"):
            result = run_demo_optimization(source, key)
            self.assertEqual(result.selector_key, key)
        selectors = runtime_selectors()
        self.assertIn("decision_tree", selectors)
        self.assertIn("random_forest", selectors)

    @unittest.skipUnless(
        (
            PRODUCTION_TREE_PATH.is_file() and PRODUCTION_FOREST_PATH.is_file()
        )
        or (
            REVIEW2_PRODUCTION_TREE_V2.is_file()
            and REVIEW2_PRODUCTION_FOREST_V2.is_file()
            and REVIEW2_PRODUCTION_LOGISTIC_V2.is_file()
            and REVIEW2_PRODUCTION_MLP_V2.is_file()
        ),
        "production models required",
    )
    def test_compare_runs_all_selectors_from_the_same_original(self) -> None:
        samples = load_demo_samples()
        cases = {
            "no_print": "t1 = b * c\nt2 = a + t1\nt3 = t2 - d\nx = t3\n",
            "with_print": "t1 = b * c\nt2 = a + t1\nt3 = t2 - d\nx = t3\nprint x\n",
            "cse": samples.get("cse_heavy"),
            "dead": samples.get("dead_code_heavy"),
            "loop": samples.get("loop_heavy"),
            "random": generate_random_tac(3),
        }
        for name, source in cases.items():
            with self.subTest(name):
                with patch.object(StrategyClassifier, "fit", side_effect=AssertionError("must not train")):
                    comparison = compare_all_selectors(source)
                self.assertTrue(comparison.uses_production_models)
                self.assertEqual(comparison.original_tac, parse_tac(source).to_tac())
                self.assertEqual(set(comparison.by_selector), set(SELECTOR_ORDER))
                originals = {item.original_tac for item in comparison.by_selector.values()}
                self.assertEqual(originals, {comparison.original_tac})
                for key, item in comparison.by_selector.items():
                    self.assertIn("no_improvement", item.initial_advice.probabilities)
                    strategies = {row["Strategy"] for row in class_score_rows(item.initial_advice, key)}
                    self.assertEqual(len(strategies), len(CLASS_ORDER))

    def test_empty_and_malformed_comparison(self) -> None:
        with self.assertRaises(DemoError):
            compare_all_selectors("  ")
        with self.assertRaises(DemoError):
            compare_all_selectors("goto nowhere\n")

    def test_ensure_production_models_matches_files(self) -> None:
        review1 = PRODUCTION_TREE_PATH.is_file() and PRODUCTION_FOREST_PATH.is_file()
        review2 = (
            REVIEW2_PRODUCTION_TREE_V2.is_file()
            and REVIEW2_PRODUCTION_FOREST_V2.is_file()
            and REVIEW2_PRODUCTION_LOGISTIC_V2.is_file()
            and REVIEW2_PRODUCTION_MLP_V2.is_file()
        )
        self.assertEqual(ensure_production_models(), review1 or review2)


if __name__ == "__main__":
    unittest.main()
