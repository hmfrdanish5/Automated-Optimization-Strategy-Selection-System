"""Streamlit views for saved model-evaluation and optimization artifacts."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "outputs" / "evaluation"


def _missing(path: Path) -> bool:
    if not path.is_file():
        st.warning(f"Artifact not available: `{path.relative_to(ROOT)}`")
        return True
    return False


def render_model_performance_tab() -> None:
    st.markdown("### Saved model-performance results")
    st.caption(
        "Frozen offline evaluation on the 400-program dataset. Results are loaded from saved "
        "artifacts and are not recomputed when this tab opens."
    )
    holdout_path = EVALUATION / "holdout_metrics.csv"
    if _missing(holdout_path):
        return
    holdout = pd.read_csv(holdout_path)
    if "feature_schema" not in holdout.columns:
        st.warning("The holdout metrics file is outdated. Run `python evaluate_models.py` to regenerate it.")
        return
    ml = holdout[holdout["selector"] != "rule_based"].copy()
    if ml.empty:
        st.info("No machine-learning holdout rows are available.")
        return
    st.markdown("#### Strict holdout macro F1")
    pivot = ml.pivot(index="selector", columns="feature_schema", values="macro_f1")
    st.bar_chart(pivot)
    st.caption(
        "The original schema has eight features; the extended schema has eleven. "
        "The Rule-Based selector is shown separately because it uses the original eight features."
    )
    rule = holdout[holdout["selector"] == "rule_based"]
    if not rule.empty:
        st.dataframe(rule[["feature_schema", "accuracy", "macro_f1", "weighted_f1"]], hide_index=True, use_container_width=True)

    cm_path = EVALUATION / "confusion_matrix_random_forest_11_features.png"
    if not _missing(cm_path):
        st.markdown("#### Random Forest confusion matrix (11 features)")
        st.image(str(cm_path), use_container_width=True)
    cm_csv = EVALUATION / "confusion_matrix_random_forest_11_features.csv"
    if cm_csv.is_file():
        with st.expander("Confusion matrix values"):
            st.dataframe(pd.read_csv(cm_csv, index_col=0), use_container_width=True)

    cv_path = EVALUATION / "cross_validation_metrics.csv"
    if not _missing(cv_path):
        cv = pd.read_csv(cv_path)
        if "schema" in cv.columns:
            summary = cv.groupby(["selector", "schema"], as_index=False)["macro_f1"].agg(mean="mean", std="std")
            st.markdown("#### Grouped cross-validation")
            st.dataframe(summary, hide_index=True, use_container_width=True)

    ablation_path = EVALUATION / "feature_ablation.csv"
    if not _missing(ablation_path):
        ablation = pd.read_csv(ablation_path)
        st.markdown("#### Eight-feature vs eleven-feature ablation")
        st.dataframe(ablation.pivot(index="model", columns="schema", values="macro_f1"), use_container_width=True)
        st.caption("Both schemas are evaluated on the same grouped holdout programs.")

    st.markdown("#### Per-class metrics")
    for _, row in ml.iterrows():
        schema = row.get("feature_schema", "feature schema")
        with st.expander(f"{row['selector']} · {schema} · macro F1 {row['macro_f1']:.3f}"):
            try:
                per_class = ast.literal_eval(str(row["per_class"]))
            except (SyntaxError, ValueError):
                st.write("Per-class metrics could not be parsed from the saved artifact.")
                continue
            rows = [
                {"class": name, "precision": stats.get("precision"), "recall": stats.get("recall"),
                 "F1": stats.get("f1-score"), "support": stats.get("support")}
                for name, stats in per_class.items()
                if name not in {"accuracy", "macro avg", "weighted avg"}
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_optimization_analytics_tab() -> None:
    st.markdown("### Optimization analytics")
    st.caption("Saved offline results from the deterministic dataset and model evaluation.")
    importance = EVALUATION / "random_forest_feature_importance_11_features.png"
    if not _missing(importance):
        st.markdown("#### Random Forest feature importance")
        st.image(str(importance), use_container_width=True)
        st.caption("Mean decrease in impurity; this is a model diagnostic, not a causal effect.")
    coefficients = EVALUATION / "logistic_coefficients_11_features.csv"
    if coefficients.is_file():
        with st.expander("Logistic Regression coefficients"):
            st.dataframe(pd.read_csv(coefficients), hide_index=True, use_container_width=True)
    results = EVALUATION / "optimization_results.csv"
    if not _missing(results):
        frame = pd.read_csv(results)
        st.markdown("#### Measured optimization results")
        st.dataframe(frame.head(60), hide_index=True, use_container_width=True)
        st.caption("These are saved offline measurements; live program results appear in Optimize Program.")
    summary = EVALUATION / "summary.txt"
    if summary.is_file():
        with st.expander("Evaluation summary"):
            st.code(summary.read_text(encoding="utf-8"), language="text")


def render_semantic_correctness_tab() -> None:
    st.markdown("### Semantic preservation approach")
    st.markdown(
        """
**Conservative optimizer rules** are intended to preserve observable behavior. For example,
constant folding avoids unsafe division, and dead-code elimination retains unused `/` or `%`
operations when removing them could hide a division-by-zero error. These safeguards are not a
formal proof of equivalence.

**Differential testing** executes original and optimized TAC with the project interpreter and
compares printed output, runtime errors, and termination under a bounded execution limit.

The deterministic fuzz tests cover 4,800 independent optimizer-pass comparisons plus 240
iterative-engine comparisons. Hand-written regression tests cover representative cases for each
pass, and source-frontend tests compare execution before and after optimization. Passing tests are
evidence for the tested programs and inputs, not a universal guarantee.
        """
    )
    st.markdown("#### Source-language pipeline testing")
    st.markdown(
        "Tests under `tests/test_source_*.py` cover lexing, parsing, AST construction, TAC generation, "
        "definite-assignment checks, and optimized-program behavior."
    )
