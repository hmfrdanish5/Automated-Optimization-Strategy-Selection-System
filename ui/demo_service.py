"""Thin adapter between the Streamlit demo and the Review-2 backend.

Compares five selectors on the same TAC using saved production models (Review-2
when present, else Review-1). Does not train or modify datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from dataset.examples import SAMPLE_FILE, load_sample_sources
from tac.instruction import InstructionType
from ui.random_tac import generate_random_tac
from engine.loop import (
    STOP_ITERATION_LIMIT,
    STOP_NO_BENEFIT,
    STOP_REPEATED_STATE,
    OptimizationRun,
    optimize_program,
)
from engine.selectors import ModelSelector, SelectorAdvice, runtime_selectors
from evaluation.label_policy import NO_IMPROVEMENT
from features.extended_extractor import FEATURE_COLUMNS_V2, extract_features_v2
from features.extractor import FEATURE_COLUMNS, extract_features
from optimizers.registry import STRATEGY_NAMES
from selection.config import PRODUCTION_FOREST_PATH, PRODUCTION_TREE_PATH
from selection.review2_config import (
    REVIEW2_PRODUCTION_FOREST_V2,
    REVIEW2_PRODUCTION_LOGISTIC_V2,
    REVIEW2_PRODUCTION_MLP_V2,
    REVIEW2_PRODUCTION_TREE_V2,
)
from tac.parser import parse_tac
from tac.program import TACProgram

logger = logging.getLogger(__name__)

DEMO_SAMPLE_IDS = (
    "constant_heavy",
    "algebraic_heavy",
    "cse_heavy",
    "dead_code_heavy",
    "mixed",
    "loop_heavy",
)

SELECTOR_LABELS = {
    "rule_based": "Rule-Based",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "mlp": "MLP",
}

SELECTOR_ORDER = (
    "rule_based",
    "decision_tree",
    "random_forest",
    "logistic_regression",
    "mlp",
)

STRATEGY_LABELS = {
    "constant_folding": "Constant folding",
    "constant_propagation": "Constant propagation",
    "algebraic_simplification": "Algebraic simplification",
    "common_subexpression_elimination": "Common subexpression elimination",
    "dead_code_elimination": "Dead code elimination",
    "strength_reduction": "Strength reduction",
    "no_improvement": "No improvement",
}

OBSERVABILITY_NOTE = (
    "Assignments whose results are never printed or used by observable control flow "
    "may be removed as dead code. Add `print x` if you want the final value of `x` to be observable."
)

CLASS_ORDER = (*STRATEGY_NAMES, NO_IMPROVEMENT)

_STOP_MESSAGES = {
    STOP_NO_BENEFIT: "Optimization converged — no beneficial strategy remains.",
    STOP_ITERATION_LIMIT: "Maximum iteration limit reached.",
    STOP_REPEATED_STATE: "Repeated TAC state detected.",
}


class DemoError(Exception):
    """User-facing demo failure with a short message."""


class ModelArtifactsMissing(DemoError):
    """Production model files are not on disk."""


@dataclass(frozen=True)
class DemoSamples:
    """Named TAC sources for the demo sidebar."""

    by_id: dict[str, str]

    def get(self, program_id: str) -> str:
        if program_id not in self.by_id:
            raise DemoError(f"Sample program '{program_id}' was not found.")
        return self.by_id[program_id]


@dataclass(frozen=True)
class SelectorComparison:
    """Five independent engine runs on one original program."""

    submitted_tac: str
    original_tac: str
    features: dict[str, float]
    has_print: bool
    by_selector: dict[str, DemoOptimizationResult]
    uses_production_models: bool


@dataclass(frozen=True)
class DemoOptimizationResult:
    """Everything the UI needs from one Optimize click."""

    original_tac: str
    optimized_tac: str
    features: dict[str, float]
    feature_schema: str
    selector_key: str
    selector_label: str
    initial_advice: SelectorAdvice
    run: OptimizationRun
    stop_message: str
    uses_production_models: bool


def load_random_tac(seed: int | None = None) -> str:
    """Demo TAC with at least one print. Not written to a dataset."""
    return generate_random_tac(seed)


def program_has_print(program: TACProgram) -> bool:
    return any(instruction.instruction_type is InstructionType.PRINT for instruction in program.instructions)


def strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def load_demo_samples(path: Path | None = None) -> DemoSamples:
    """Load hand-written samples from examples/sample_programs.txt."""
    sources = dict(load_sample_sources(path or SAMPLE_FILE))
    missing = [program_id for program_id in DEMO_SAMPLE_IDS if program_id not in sources]
    if missing:
        raise DemoError(f"Missing demo samples in the sample file: {', '.join(missing)}")
    return DemoSamples({program_id: sources[program_id] for program_id in DEMO_SAMPLE_IDS})


def ensure_production_models() -> bool:
    """Return True when runtime selectors can load production pickles."""
    review2 = all(
        path.is_file()
        for path in (
            REVIEW2_PRODUCTION_TREE_V2,
            REVIEW2_PRODUCTION_FOREST_V2,
            REVIEW2_PRODUCTION_LOGISTIC_V2,
            REVIEW2_PRODUCTION_MLP_V2,
        )
    )
    review1 = PRODUCTION_TREE_PATH.is_file() and PRODUCTION_FOREST_PATH.is_file()
    return review2 or review1


def get_runtime_selectors():
    """Same selector map as optimize.py (production models when present)."""
    if not ensure_production_models():
        raise ModelArtifactsMissing(
            "Production models were not found. Run python train_review2.py or "
            "python evaluate.py to create model artifacts under models/."
        )
    return runtime_selectors()


def selector_feature_bundle(program: TACProgram, selector) -> tuple[dict[str, float], str]:
    """Features and schema label for display (8-feature rule baseline or ML schema)."""
    if hasattr(selector, "prepare_features"):
        features = selector.prepare_features(program)
        schema = "11 features (extended)" if len(features) == len(FEATURE_COLUMNS_V2) else "8 features (original)"
        return features, schema
    features = extract_features(program)
    return {column: features[column] for column in FEATURE_COLUMNS}, "8 features (original)"


def parse_tac_text(source: str) -> TACProgram:
    """Parse TAC or raise DemoError with a friendly message."""
    text = source.strip()
    if not text:
        raise DemoError("TAC input is empty. Enter at least one instruction per line.")
    try:
        return parse_tac(text)
    except Exception as error:
        logger.debug("TAC parse failed: %s", error)
        raise DemoError(f"Could not parse TAC: {error}") from error


def compare_all_selectors(source: str) -> SelectorComparison:
    """Run rule, decision tree, and random forest from the same original TAC.

    Each call uses ``optimize_program``, which copies its input. The parsed
    program is checked afterward and must match the text that was submitted.
    """
    program = parse_tac_text(source)
    submitted = program.to_tac()
    features = extract_features(program)
    features_v2 = extract_features_v2(program)
    selectors = get_runtime_selectors()
    by_selector: dict[str, DemoOptimizationResult] = {}
    for key in SELECTOR_ORDER:
        by_selector[key] = _optimize_copy(program, selectors[key], key, features, features_v2)
    if program.to_tac() != submitted:
        raise DemoError("Internal error: the backend changed the input program.")
    return SelectorComparison(
        submitted_tac=submitted,
        original_tac=submitted,
        features={column: features[column] for column in FEATURE_COLUMNS},
        has_print=program_has_print(program),
        by_selector=by_selector,
        uses_production_models=ensure_production_models(),
    )


def comparison_table_rows(comparison: SelectorComparison) -> list[dict[str, object]]:
    """One summary row per selector, using measured engine fields."""
    rows = []
    for key in SELECTOR_ORDER:
        item = comparison.by_selector[key]
        advice = item.initial_advice
        top = advice.confidence
        if key == "rule_based":
            top_text = f"{top:.4f} (rule score)"
        else:
            top_text = f"{top * 100:.1f}% (class probability)"
        rows.append(
            {
                "Selector": item.selector_label,
                "First predicted strategy": strategy_label(advice.selected_strategy),
                "Top score or probability": top_text,
                "Final instructions": item.run.final_instruction_count,
                "Instruction gain": item.run.total_instruction_gain,
                "Weighted-cost gain": item.run.total_weighted_cost_gain,
                "Successful iterations": item.run.successful_iterations,
                "Candidate attempts": item.run.candidate_attempts,
                "Convergence": item.stop_message,
            }
        )
    return rows


def class_score_rows(advice: SelectorAdvice, selector_key: str) -> list[dict[str, object]]:
    """All seven classes, highest score first. Includes no_improvement."""
    ranked = list(advice.ranking)
    for name in CLASS_ORDER:
        if name not in ranked:
            ranked.append(name)
    rows = []
    best = ranked[0] if ranked else ""
    for name in ranked:
        value = float(advice.probabilities.get(name, 0.0))
        row: dict[str, object] = {
            "Strategy": strategy_label(name),
            "Top prediction": "yes" if name == advice.selected_strategy else "",
        }
        if selector_key == "rule_based":
            row["Rule score"] = round(value, 4)
        else:
            row["Probability (%)"] = round(value * 100.0, 1)
        if name == best and name != advice.selected_strategy:
            row["Top prediction"] = ""
        rows.append(row)
    return rows


def horizontal_score_figure(advice: SelectorAdvice, selector_key: str):
    """Horizontal bars. ML values are percentages from 0 to 100."""
    import matplotlib.pyplot as plt

    names = []
    values = []
    for name in advice.ranking:
        names.append(strategy_label(name))
        raw = float(advice.probabilities.get(name, 0.0))
        values.append(raw if selector_key == "rule_based" else raw * 100.0)
    # Highest score at the top of the chart.
    names = list(reversed(names))
    values = list(reversed(values))
    figure, axis = plt.subplots(figsize=(7.2, 3.1))
    axis.barh(names, values, color="#3d6f99")
    if selector_key == "rule_based":
        axis.set_xlabel("Rule score (not a probability)")
    else:
        axis.set_xlim(0, 100)
        axis.set_xlabel("Class probability (%)")
    axis.set_ylabel("")
    figure.tight_layout()
    return figure


def run_demo_optimization(source: str, selector_key: str) -> DemoOptimizationResult:
    """Parse, advise, and optimize without mutating the parsed program."""
    if selector_key not in SELECTOR_LABELS:
        raise DemoError(f"Unknown selector '{selector_key}'.")
    program = parse_tac_text(source)
    original_tac = program.to_tac()
    selectors = get_runtime_selectors()
    selector = selectors[selector_key]
    features, schema = selector_feature_bundle(program, selector)
    initial_advice = selector.advise(features)
    try:
        run = optimize_program(program, selector, program_id="demo")
    except Exception as error:
        logger.exception("Optimization failed")
        raise DemoError(f"Optimization failed: {error}") from error
    if program.to_tac() != original_tac:
        raise DemoError("Internal error: the backend changed the input program.")
    return DemoOptimizationResult(
        original_tac=original_tac,
        optimized_tac=run.final_program.to_tac(),
        features=features,
        feature_schema=schema,
        selector_key=selector_key,
        selector_label=SELECTOR_LABELS[selector_key],
        initial_advice=initial_advice,
        run=run,
        stop_message=convergence_message(run.stopped_because),
        uses_production_models=ensure_production_models(),
    )


def convergence_message(stopped_because: str) -> str:
    """Map engine stop codes to demo text."""
    return _STOP_MESSAGES.get(stopped_because, stopped_because)


def iteration_history_rows(run: OptimizationRun, selector_key: str = "") -> list[dict[str, object]]:
    """One row per optimizer attempt recorded by the engine."""
    score_name = "Rule score" if selector_key == "rule_based" else "Class probability (%)"
    rows: list[dict[str, object]] = []
    for record in run.history:
        for attempt in record.attempts:
            raw = float(record.candidate_probabilities.get(attempt.strategy, 0.0))
            shown = round(raw, 4) if selector_key == "rule_based" else round(raw * 100.0, 1)
            rows.append(
                {
                    "Iteration": record.iteration,
                    "Strategy attempted": strategy_label(attempt.strategy),
                    score_name: shown,
                    "Before": attempt.instruction_count_before,
                    "After": attempt.instruction_count_after,
                    "Instruction gain": attempt.instruction_gain,
                    "Weighted-cost gain": attempt.weighted_cost_gain,
                    "Status": "Accepted" if attempt.accepted else "Rejected",
                }
            )
    return rows


def _optimize_copy(
    program,
    selector,
    selector_key: str,
    features_v1: dict[str, float],
    features_v2: dict[str, float],
) -> DemoOptimizationResult:
    before = program.to_tac()
    if isinstance(selector, ModelSelector):
        features, schema = selector_feature_bundle(program, selector)
    else:
        features = {column: features_v1[column] for column in FEATURE_COLUMNS}
        schema = "8 features (original)"
    advice = selector.advise(features)
    try:
        run = optimize_program(program, selector, program_id="demo")
    except Exception as error:
        logger.exception("Optimization failed")
        raise DemoError(f"Optimization failed: {error}") from error
    if program.to_tac() != before:
        raise DemoError("Internal error: the backend changed the input program.")
    return DemoOptimizationResult(
        original_tac=before,
        optimized_tac=run.final_program.to_tac(),
        features=features,
        feature_schema=schema,
        selector_key=selector_key,
        selector_label=SELECTOR_LABELS[selector_key],
        initial_advice=advice,
        run=run,
        stop_message=convergence_message(run.stopped_because),
        uses_production_models=ensure_production_models(),
    )


def ranking_rows(advice: SelectorAdvice, selector_key: str) -> list[dict[str, object]]:
    """Strategy ranking for the expandable panel."""
    if selector_key == "rule_based":
        label = "Score"
    else:
        label = "Probability"
    rows = []
    for name in advice.ranking:
        rows.append(
            {
                "Strategy": name,
                label: round(float(advice.probabilities.get(name, 0.0)), 4),
            }
        )
    return rows


def probability_chart_frame(advice: SelectorAdvice):
    """Data frame for a horizontal bar chart (ML selectors only)."""
    import pandas as pd

    rows = [
        {"strategy": name, "probability": float(advice.probabilities.get(name, 0.0))}
        for name in CLASS_ORDER
    ]
    return pd.DataFrame(rows).sort_values("probability", ascending=True)
