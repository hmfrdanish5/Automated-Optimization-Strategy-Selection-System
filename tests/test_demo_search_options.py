"""Rule-Based v2 and early stop in the demo service and the command-line runner."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import unittest

from engine.loop import STOP_PREDICTED_NO_IMPROVEMENT
from ui.demo_service import SELECTOR_ORDER, compare_all_selectors, ensure_production_models

ROOT = Path(__file__).resolve().parents[1]
TAC = "t1 = a * 2\nspare = 4\nprint t1\n"


@unittest.skipUnless(ensure_production_models(), "production models not trained")
class DemoSearchOptionTests(unittest.TestCase):
    def test_rule_based_v2_is_compared_on_eleven_features(self) -> None:
        comparison = compare_all_selectors(TAC)
        self.assertIn("rule_based_v2", SELECTOR_ORDER)
        item = comparison.by_selector["rule_based_v2"]
        self.assertEqual(item.feature_schema, "11 features (extended)")
        self.assertEqual(item.selector_label, "Rule-Based v2")

    def test_early_stop_reaches_the_same_code_with_fewer_runs(self) -> None:
        full = compare_all_selectors(TAC)
        early = compare_all_selectors(TAC, stop_on_no_improvement=0.7)
        for key in SELECTOR_ORDER:
            self.assertLessEqual(
                early.by_selector[key].run.optimizer_evaluations_total,
                full.by_selector[key].run.optimizer_evaluations_total,
            )
        stopped = {key for key in SELECTOR_ORDER if early.by_selector[key].run.stopped_because == STOP_PREDICTED_NO_IMPROVEMENT}
        self.assertTrue(stopped, "at least one selector should stop early on this program")
        for key in stopped:
            self.assertEqual(early.by_selector[key].optimized_tac, full.by_selector[key].optimized_tac)


@unittest.skipUnless(ensure_production_models(), "production models not trained")
class CommandLineTests(unittest.TestCase):
    def test_source_option_compiles_and_compares_every_selector(self) -> None:
        import optimize

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            optimize.main(["--source", str(ROOT / "examples" / "source_program.txt"), "--early-stop", "0.7"])
        text = output.getvalue()
        self.assertIn("Generated TAC:", text)
        self.assertIn("rule_based_v2", text)
        self.assertIn("unused = 99", text)

    def test_early_stop_must_be_a_confidence(self) -> None:
        import optimize

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            optimize.main(["--early-stop", "1.5"])


if __name__ == "__main__":
    unittest.main()
