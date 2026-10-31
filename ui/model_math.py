"""Mathematics reference for the Streamlit demo."""

from __future__ import annotations

import streamlit as st

from evaluation.operation_cost import OPERATION_COST, UNARY_MINUS_COST
from features.extended_extractor import FEATURE_COLUMNS_EXTENDED
from features.extractor import FEATURE_COLUMNS
from selection.search_config import DEFAULT_EARLY_STOP_THRESHOLD, RULE_WEIGHTS_V2
from selection.config import RULE_WEIGHTS
from selection.model_config import LOGISTIC_PARAMETERS, MLP_PARAMETERS


def render_mathematics_tab() -> None:
    st.markdown("### Mathematics reference")
    st.caption(
        "Notation matches offline training and live optimization. "
        "Rule scores are **not** probabilities; ML outputs are **not** guaranteed gains."
    )

    tab_features, tab_rules, tab_ml, tab_engine, tab_eval, tab_sem = st.tabs(
        [
            "Features",
            "Rule-Based",
            "ML selectors",
            "Optimization",
            "Offline evaluation",
            "Semantics",
        ]
    )

    with tab_features:
        _render_features()

    with tab_rules:
        _render_rule_based()

    with tab_ml:
        _render_ml_selectors()

    with tab_engine:
        _render_optimization_metrics()

    with tab_eval:
        _render_offline_evaluation()

    with tab_sem:
        _render_semantics()


def _render_features() -> None:
    st.markdown("**Program counts** (denominator **N** = `num_statements`)")
    rows = [{"Symbol / feature": name, "Meaning": _feature_role(name)} for name in FEATURE_COLUMNS]
    for name in FEATURE_COLUMNS_EXTENDED:
        if name not in FEATURE_COLUMNS:
            rows.append({"Symbol / feature": name, "Meaning": _feature_role(name)})
    # Tall enough to show every feature without an inner scroll bar.
    st.dataframe(rows, hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 4)
    st.latex(r"const\_ratio = N_c / N")
    st.caption("N_c is `num_constants`; N is `num_statements` (0 when empty).")


def _render_rule_based() -> None:
    st.markdown("**Rule-Based selector** (eight features, scaled by program size)")
    st.caption(
        "For each strategy, count features are divided by N before weights from "
        "`selection/config.py` are applied. `const_ratio` is already normalized."
    )
    st.latex(r"S_{CF} = \frac{N_c}{N} + 0.5 \cdot R_c")
    st.latex(r"S_{CP} = 0.2 \cdot R_c")
    st.latex(r"S_{ALG} = \frac{N_a}{N}, \quad S_{CSE} = \frac{N_r}{N}")
    st.latex(r"S_{DCE} = \frac{N_d}{N}, \quad S_{SR} = \frac{N_l}{N}")
    st.markdown(
        """
| Symbol | Feature column |
| --- | --- |
| N_c | `num_constants` |
| R_c | `const_ratio` |
| N_a | `num_algebraic_ops` |
| N_r | `num_repeated_expr` |
| N_d | `num_dead_stmts` |
| N_l | `num_loop_ops` |
        """
    )
    weight_rows = []
    for strategy, weights in RULE_WEIGHTS.items():
        weight_rows.append(
            {
                "Strategy": strategy,
                "Weighted features": ", ".join(f"{k}×{v}" for k, v in weights.items()),
            }
        )
    st.dataframe(weight_rows, hide_index=True, width="stretch")
    st.caption("Highest positive score wins; all zero → `no_improvement`.")

    st.markdown("**Rule-Based v2** (the same eleven features as the ML models, raw counts)")
    st.caption(
        "A fair rule baseline ordered like the label policy: dead code first (only DCE removes instructions), "
        "then the passes that save operation cost, and constant propagation last. Weights from "
        "`selection/search_config.py`."
    )
    st.latex(r"S^{v2}_s = \sum_f w_{s,f} \cdot f")
    v2_rows = [
        {"Strategy": strategy, "Weighted features": ", ".join(f"{k}×{v:g}" for k, v in weights.items())}
        for strategy, weights in RULE_WEIGHTS_V2.items()
    ]
    st.dataframe(v2_rows, hide_index=True, width="stretch")
    st.caption("All scores zero → `no_improvement`; ties use the fixed strategy priority.")


def _render_ml_selectors() -> None:
    st.markdown("**Decision Tree and Random Forest**")
    st.markdown(
        "Split nodes test **x_j ≤ t** on one feature; leaves predict a strategy class. "
        "Random Forest averages many trees (100 trees, fixed seed offline)."
    )
    st.markdown("**Logistic Regression**")
    st.latex(
        r"P(y=k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^\top \mathbf{z} + b_k)}"
        r"{\sum_j \exp(\mathbf{w}_j^\top \mathbf{z} + b_j)}"
    )
    st.latex(r"\mathbf{z} = \mathrm{StandardScale}(\mathbf{x})")
    st.caption(
        f"solver={LOGISTIC_PARAMETERS['solver']!r}, C={LOGISTIC_PARAMETERS['C']} (offline)."
    )
    st.markdown("**MLP**")
    st.latex(r"\mathbf{h} = \mathrm{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1)")
    st.latex(r"P(y=k \mid \mathbf{x}) = \mathrm{softmax}_k(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)")
    st.caption(
        f"Hidden {MLP_PARAMETERS['hidden_layer_sizes']}, "
        f"alpha={MLP_PARAMETERS['alpha']} (offline)."
    )
    st.markdown("**Feature vector x**")
    st.caption(
        "The Rule-Based selector uses eight features. Rule-Based v2 and the four production ML models use all "
        "eleven features."
    )


def _render_optimization_metrics() -> None:
    st.markdown("**Weighted arithmetic cost** (labels and secondary gains)")
    cost_table = [
        {"Operator": op, "Weight": w} for op, w in OPERATION_COST.items()
    ]
    cost_table.append({"Operator": "unary −", "Weight": UNARY_MINUS_COST})
    st.dataframe(cost_table, hide_index=True, width="stretch")
    st.caption("Assignments, prints, labels, jumps, and comparisons add **0** to C(P).")
    st.latex(
        r"C(P) = n_{+} + n_{-} + 2 n_{*} + 4 n_{/} + 4 n_{\%} + n_{unary}"
    )
    st.latex(r"G_I = I_o - I_n")
    st.caption(
        "I_o / I_n: instruction counts before and after a pass. "
        "The engine accepts a step only when measured gain is positive."
    )
    st.markdown("**Search controls** (optional; off by default)")
    st.latex(
        r"\text{early stop: } \hat{y} = \texttt{no\_improvement} \;\wedge\; P(\hat{y} \mid \mathbf{x}) \ge \tau"
    )
    st.latex(r"\text{budget: try only the top } k \text{ ranked strategies per round}")
    st.caption(
        f"The demo's stop-early switch uses τ = {DEFAULT_EARLY_STOP_THRESHOLD:g}. "
        "Rule scores are not probabilities, so for the rule selectors the top score plays the role of P."
    )


def _render_offline_evaluation() -> None:
    st.markdown("**Dataset labeling** (frozen CSV — not recomputed in the UI)")
    st.latex(r"s^* = \arg\max_s^{\mathrm{lex}} G(s)")
    st.caption(
        "G(s) is instruction-count gain from strategy s; lex tie-break uses fixed priority. "
        "Weighted-cost gain breaks remaining ties offline."
    )
    st.latex(r"MacroF1 = \frac{1}{7} \sum_{k=1}^{7} F1_k")
    st.caption(
        "Seven classes: six optimizers plus `no_improvement`. "
        "Holdout tables in **Model Performance** use saved CSV/PNG artifacts only."
    )


def _render_semantics() -> None:
    st.markdown("**Differential testing** (regression evidence, not proof)")
    st.latex(r"Obs(P, x) = Obs(\mathrm{Optimize}(P), x)")
    st.caption(
        "Obs includes printed integers, runtime errors, and termination under the TAC interpreter "
        "(10,000-step limit). Conservative optimizer rules and DCE trap safeguards aim to preserve "
        "Obs on tested inputs only."
    )


def _feature_role(name: str) -> str:
    roles = {
        "num_statements": "N — all instructions.",
        "num_constants": "N_c — foldable constant arithmetic.",
        "num_repeated_expr": "N_r — duplicate expressions (CSE hint).",
        "num_dead_stmts": "N_d — removable dead assignments.",
        "num_algebraic_ops": "N_a — algebraic identities.",
        "num_loop_ops": "N_l — instructions inside loops.",
        "num_tmp_vars": "Unique t* or __src_tmp_* names.",
        "const_ratio": "R_c = N_c / N.",
        "num_constant_propagation_uses": "Constant-propagation substitution opportunities.",
        "num_safe_repeated_expr": "Repeated expressions safe for common-subexpression elimination.",
        "num_mul_by_two_in_loops": "Multiplication-by-two sites inside loops.",
    }
    return roles.get(name, "Model input.")
