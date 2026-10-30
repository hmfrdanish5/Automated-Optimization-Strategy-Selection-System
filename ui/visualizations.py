"""Streamlit helpers for offline experiment visualizations."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REVIEW2 = OUTPUTS / "review2"


def _missing(path: Path) -> bool:
    if not path.is_file():
        st.warning(f"Artifact not available: `{path.relative_to(ROOT)}`")
        return True
    return False


def _caption(text: str) -> None:
    st.caption(text)


def render_model_performance_tab() -> None:
    st.markdown("### Review-2 classification artifacts (frozen offline results)")
    _caption(
        "Dataset: `dataset/dataset_v2.csv` (400 programs). "
        "This tab mixes **two holdout settings**: the five-selector table uses the "
        "eight-feature (v1) strict holdout in `holdout_metrics.csv`; the Random Forest "
        "v2 confusion matrix is a separate eleven-feature (v2) holdout run. "
        "Nothing here is recomputed live in the UI."
    )
    holdout_path = REVIEW2 / "holdout_metrics.csv"
    if _missing(holdout_path):
        return
    holdout = pd.read_csv(holdout_path)
    ml = holdout[holdout["selector"] != "rule_based"].copy()
    if ml.empty:
        st.info("No ML holdout rows found.")
        return

    st.markdown("#### Confusion matrix (Random Forest, eleven-feature v2 holdout)")
    cm_png = REVIEW2 / "confusion_matrix_random_forest_v2.png"
    if not _missing(cm_png):
        _caption(
            "Model: Random Forest · Feature schema: **eleven-feature v2** · "
            "Standalone holdout from `evaluate_five_selectors(..., tag=\"v2\")` "
            "(not the same run as the eight-feature five-selector table below)."
        )
        st.image(str(cm_png), use_container_width=True)
    cm_csv = REVIEW2 / "confusion_matrix_random_forest_v2.csv"
    if not _missing(cm_csv):
        matrix = pd.read_csv(cm_csv, index_col=0)
        with st.expander("Confusion matrix table (Random Forest v2)"):
            st.dataframe(matrix, use_container_width=True)

    st.markdown("#### Review-1 baseline confusion matrix (200-program era)")
    review1_cm = OUTPUTS / "confusion_matrix.png"
    if not _missing(review1_cm):
        _caption("Review-1 frozen artifact · Decision Tree / Random Forest training era")
        st.image(str(review1_cm), use_container_width=True)

    st.markdown("#### Holdout macro F1 — five selectors (eight-feature v1 schema)")
    chart = ml.set_index("selector")[["macro_f1", "weighted_f1"]]
    st.bar_chart(chart)
    _caption(
        "From `holdout_metrics.csv`: strict grouped holdout on `dataset_v2` with the "
        "**eight-feature (v1)** production models for Rule-Based, Decision Tree, Random Forest, "
        "Logistic Regression, and MLP."
    )

    st.markdown("#### Grouped cross-validation macro F1 (Review-2)")
    cv_path = REVIEW2 / "cross_validation_metrics.csv"
    if not _missing(cv_path):
        cv = pd.read_csv(cv_path)
        summary = (
            cv.groupby(["selector", "schema"], as_index=False)["macro_f1"]
            .mean()
            .pivot(index="selector", columns="schema", values="macro_f1")
        )
        st.dataframe(summary, use_container_width=True)
        _caption(
            "Evaluation: StratifiedGroupKFold on `dataset_v2.csv`. "
            "Columns v1/v2 name the eight- and eleven-feature schemas."
        )

    st.markdown("#### Eight vs eleven feature ablation (shared strict holdout)")
    ablation_path = REVIEW2 / "feature_ablation.csv"
    if not _missing(ablation_path):
        abl = pd.read_csv(ablation_path)
        pivot = abl.pivot(index="model", columns="schema", values="macro_f1")
        st.dataframe(pivot, use_container_width=True)
        _caption(
            "Shared strict holdout grouped by original **eight-feature program signatures** "
            "(`feature_ablation.csv`). Compares v1_8_features vs v2_11_features on the same programs."
        )

    st.markdown("#### Per-class precision / recall / F1 (eight-feature v1 holdout)")
    for _, row in ml.iterrows():
        with st.expander(f"{row['selector']} — macro F1 {row['macro_f1']:.3f}"):
            try:
                per_class = ast.literal_eval(str(row["per_class"]))
            except (SyntaxError, ValueError):
                st.write("Per-class metrics could not be parsed.")
                continue
            rows = []
            for name, stats in per_class.items():
                if name in {"accuracy", "macro avg", "weighted avg"}:
                    continue
                rows.append(
                    {
                        "class": name,
                        "precision": stats.get("precision"),
                        "recall": stats.get("recall"),
                        "f1": stats.get("f1-score"),
                        "support": stats.get("support"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_optimization_analytics_tab() -> None:
    st.markdown("### Offline optimization analytics (saved artifacts)")
    _caption("These charts summarize historical experiments — not the program currently edited in Tab 1.")

    st.markdown("#### Random Forest feature importance")
    for label, path in (
        ("Review-2 (11-feature)", REVIEW2 / "random_forest_feature_importance_v2.png"),
        ("Review-1 baseline", OUTPUTS / "random_forest_feature_importance.png"),
    ):
        if not _missing(path):
            st.image(str(path), use_container_width=True)
            _caption(f"{label} · mean decrease in impurity · not a causal effect")

    st.markdown("#### Confidence vs measured gain")
    scatter = OUTPUTS / "confidence_vs_gain.png"
    if not _missing(scatter):
        st.image(str(scatter), use_container_width=True)
        _caption("Review-1 era scatter from saved optimization/search logs.")

    st.markdown("#### Instruction reduction by strategy")
    box = OUTPUTS / "instruction_reduction_by_strategy.png"
    if not _missing(box):
        st.image(str(box), use_container_width=True)
        _caption("Review-1 dataset instruction-count gains per optimizer pass.")

    st.markdown("#### Strategy selection sequences (sample programs)")
    sequences = OUTPUTS / "strategy_sequences.png"
    if not _missing(sequences):
        st.image(str(sequences), use_container_width=True)

    st.markdown("#### Constant-feature count vs constant-folding gain")
    constants = OUTPUTS / "constants_vs_constant_folding.png"
    if not _missing(constants):
        st.image(str(constants), use_container_width=True)

    opt_csv = REVIEW2 / "optimization_results.csv"
    if opt_csv.is_file():
        with st.expander("Review-2 optimization_results.csv (first 50 rows)"):
            sample = pd.read_csv(opt_csv).head(50)
            st.dataframe(sample, use_container_width=True)
            _caption("Per-program holdout optimization traces — live UI runs are separate measurements.")


def render_semantic_correctness_tab() -> None:
    st.markdown("### Semantic preservation approach")
    st.markdown(
        """
**Conservative optimizer rules** (for example, refusing to fold division by a variable that
might be zero, and keeping unused `/` or `%` when the divisor may be zero during dead-code
elimination) are intended to preserve semantics but are **not** a formal proof.

**Offline differential testing** executes original and optimized TAC with the project
interpreter (10,000-step limit) and compares printed output, runtime errors, and termination.

**Deterministic fuzz** (`tests/test_semantic_fuzz.py`, seed `20261029`):
80 generated programs × 10 input environments × 6 optimizer passes (4,800 comparisons), plus
24 programs × 10 environments through the fixed-order iterative engine (240 comparisons).

**Hand-written regression** (`tests/test_semantics.py`) covers representative programs on each pass.

These counts reflect the latest verified full-suite run (**221** unit tests including fuzz subTests);
passing tests provide evidence for exercised cases only, not universal equivalence.
        """
    )
    st.markdown("#### Source-language pipeline testing")
    st.markdown(
        "Dedicated tests under `tests/test_source_*.py` compile restricted C-like programs to TAC, "
        "execute them, and compare optimized behavior using the same differential utilities."
    )
