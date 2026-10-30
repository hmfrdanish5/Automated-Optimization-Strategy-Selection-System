"""Mathematics captions for ML selectors in the Streamlit demo."""

from __future__ import annotations

import streamlit as st

from features.extended_extractor import FEATURE_COLUMNS_V2
from features.extractor import FEATURE_COLUMNS
from selection.review2_config import LOGISTIC_PARAMETERS, MLP_PARAMETERS


def render_mathematics_tab() -> None:
    st.markdown("### How features enter each selector")
    st.caption(
        "Live optimization extracts an eight- or eleven-dimensional vector **x** from the "
        "current TAC (counts such as statements, dead code, loops). Rule-Based and Random Forest "
        "use related but separate scoring rules; this panel focuses on the three parametric models."
    )

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("**Feature vector x**")
        rows = [{"Feature": name, "Role": _feature_role(name)} for name in FEATURE_COLUMNS]
        for name in FEATURE_COLUMNS_V2:
            if name not in FEATURE_COLUMNS:
                rows.append({"Feature": name, "Role": _feature_role(name)})
        st.dataframe(rows, hide_index=True, use_container_width=True, height=420)

    with right:
        st.markdown("**Decision Tree**")
        st.markdown(
            """
At each node the tree compares one component **x_j** to a learned threshold **t**:
`if x_j ≤ t go left else right`. The leaf stores a class label (one of six optimizers or
`no_improvement`). **Every feature** can appear in splits; importance in offline plots reflects
how often a split reduced Gini impurity on the training set—not causal effect.
            """
        )

        st.markdown("**Logistic Regression**")
        st.latex(
            r"P(y=k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^\top \mathbf{z} + b_k)}"
            r"{\sum_j \exp(\mathbf{w}_j^\top \mathbf{z} + b_j)}, \quad "
            r"\mathbf{z} = \mathrm{StandardScale}(\mathbf{x})"
        )
        st.caption(
            f"Offline training used solver={LOGISTIC_PARAMETERS['solver']!r}, "
            f"C={LOGISTIC_PARAMETERS['C']}. Each **w_k** has one weight per feature in x."
        )

        st.markdown("**MLP**")
        st.latex(
            r"\mathbf{h} = \mathrm{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1), \quad "
            r"P(y=k \mid \mathbf{x}) = \mathrm{softmax}_k(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)"
        )
        st.caption(
            f"Saved architecture: hidden layers {MLP_PARAMETERS['hidden_layer_sizes']}, "
            f"alpha={MLP_PARAMETERS['alpha']}. Input **z** is the same standardized feature vector."
        )

    st.info(
        "Class probabilities rank candidates only. The engine still **measures** instruction and "
        "weighted-cost gains before accepting an optimization step."
    )


def _feature_role(name: str) -> str:
    roles = {
        "num_statements": "Size of the program (all instructions).",
        "num_constants": "Foldable constant arithmetic sites.",
        "num_repeated_expr": "Duplicate expression patterns (CSE hint).",
        "num_dead_stmts": "Assignments DCE could remove.",
        "num_algebraic_ops": "Algebraic identity opportunities.",
        "num_loop_ops": "Instructions inside loops.",
        "num_tmp_vars": "Compiler temporary names (t* or __src_tmp_*).",
        "const_ratio": "num_constants / num_statements.",
        "num_constant_propagation_uses": "Safe constant-propagation sites (11-feature schema).",
        "num_safe_repeated_expr": "CSE-safe repeated expressions (extended).",
        "num_mul_by_two_in_loops": "Loop ×2 sites for strength reduction (extended).",
    }
    return roles.get(name, "Model input dimension.")
