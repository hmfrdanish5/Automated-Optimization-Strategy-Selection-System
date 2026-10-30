
from __future__ import annotations

import html
import logging

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import streamlit as st

from selection.config import FOREST_PARAMETERS, MODEL_RANDOM_STATE
from features.extended_extractor import FEATURE_COLUMNS_V2
from ui.demo_service import (
    DEMO_SAMPLE_IDS,
    OBSERVABILITY_NOTE,
    SELECTOR_LABELS,
    SELECTOR_ORDER,
    DemoError,
    ModelArtifactsMissing,
    class_score_rows,
    compare_all_selectors,
    comparison_table_rows,
    compile_source_for_demo,
    horizontal_score_figure,
    iteration_history_rows,
    load_demo_samples,
    load_random_tac,
    load_random_source,
    strategy_label,
)
from ui.model_math import render_mathematics_tab
from ui.visualizations import (
    render_model_performance_tab,
    render_optimization_analytics_tab,
    render_semantic_correctness_tab,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLACEHOLDER = """t1 = b * c
t2 = a + t1
t3 = t2 - d
x = t3
print x"""

SOURCE_PLACEHOLDER = """int a = 10;
int b = 20;
int c = a + b * 2;
print(c);
"""

INPUT_MODES = ("Source Code", "Advanced TAC Input")

PANELS = (
    (
        "overview",
        "Overview",
        "Instruction counts, gains, and extracted features for all five selectors.",
    ),
    (
        "scores",
        "Comparison",
        "Pick one selector and inspect its predicted strategy and full ranking.",
    ),
    (
        "output",
        "TAC output",
        "Pick one selector and compare the original TAC with that selector's optimized TAC.",
    ),
    (
        "history",
        "History",
        "Pick one selector and review every optimizer attempt, including rejected ones.",
    ),
)

FEATURE_HELP = {
    "num_statements": "Every TAC instruction, including labels, jumps, and prints.",
    "num_constants": "Foldable constant arithmetic (not a plain copy like x = 10).",
    "num_repeated_expr": "Extra copies of the same written expression.",
    "num_dead_stmts": "Assignments the dead-code analysis can remove.",
    "num_algebraic_ops": "Identities such as x + 0 and x * 1.",
    "num_loop_ops": "Instructions inside a back-edge loop.",
    "num_tmp_vars": "Unique temporaries: t1, t2, … or __src_tmp_1 from the source compiler.",
    "const_ratio": "num_constants divided by num_statements.",
    "num_constant_propagation_uses": "Sites constant propagation could rewrite.",
    "num_safe_repeated_expr": "Repeated expressions safe for CSE.",
    "num_mul_by_two_in_loops": "Loop multiplies by two that strength reduction could rewrite.",
}


@st.cache_resource(show_spinner=False)
def cached_samples():
    return load_demo_samples()


def main() -> None:
    st.set_page_config(page_title="Optimization Strategy Selection", layout="wide")
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True
    if "editor_height" not in st.session_state:
        st.session_state.editor_height = 220
    _apply_layout(st.session_state.sidebar_open)
    _banner()

    samples = cached_samples()
    tab_optimize, tab_models, tab_analytics, tab_semantic = st.tabs(
        [
            "Optimize Program",
            "Model Performance",
            "Optimization Analytics",
            "Semantic Correctness",
        ]
    )
    with tab_optimize:
        toggle_col, _spacer = st.columns([0.12, 0.88])
        with toggle_col:
            label = "Hide program panel" if st.session_state.sidebar_open else "Show program panel"
            if st.button(label, use_container_width=True, key="sidebar_toggle"):
                st.session_state.sidebar_open = not st.session_state.sidebar_open
                st.rerun()
        if st.session_state.sidebar_open:
            with st.sidebar:
                _editor(samples)
        _results_panel()
    with tab_models:
        render_model_performance_tab()
    with tab_math:
        render_mathematics_tab()
    with tab_analytics:
        render_optimization_analytics_tab()
    with tab_semantic:
        render_semantic_correctness_tab()
    _about()


def _banner() -> None:
    st.markdown(
        """
        <div class="app-banner">
            Automated Optimization Strategy Selection
        </div>
        """,
        unsafe_allow_html=True,
    )


def _apply_layout(sidebar_open: bool) -> None:
    sidebar_css = ""
    if not sidebar_open:
        sidebar_css = """
        section[data-testid="stSidebar"] { display: none !important; }
        section.main > div.block-container { max-width: 1400px; }
        """
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff; }
        section.main > div.block-container {
            max-width: 1220px;
            padding-top: 0.8rem;
            padding-bottom: 1.2rem;
        }
        """
        + sidebar_css
        + """
        .app-banner {
            background: #6baed6;
            color: #ffffff;
            border-radius: 12px;
            padding: 0.95rem 1.25rem;
            margin: 0 0 1rem 0;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            box-shadow: 0 2px 6px rgba(107, 174, 214, 0.35);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #d5e4f0 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 2px rgba(30, 42, 54, 0.04);
        }
        div.stButton > button[kind="primary"] {
            background-color: #6baed6;
            border-color: #6baed6;
            color: #ffffff;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #4f9fd0;
            border-color: #4f9fd0;
            color: #ffffff;
        }
        div.stButton > button[kind="secondary"] {
            border-color: #6baed6;
            color: #2b6a94;
        }
        /* Hide Streamlit's built-in help icons so we use the custom tip-i only. */
        div[data-testid="stTooltipIcon"] { display: none !important; }
        .tip-row {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            margin: 0 0 0.35rem 0;
            font-weight: 600;
            color: #1e2a36;
        }
        .tip-i {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #e8f4fb;
            color: #2b6a94;
            font-family: Georgia, "Times New Roman", serif;
            font-style: italic;
            font-size: 11px;
            font-weight: 700;
            line-height: 1;
            cursor: help;
            flex: 0 0 auto;
        }
        .tip-i:hover {
            background: #2b6a94;
            color: #ffffff;
        }
        .tip-i .tip-bubble {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            left: 50%;
            top: calc(100% + 8px);
            transform: translateX(-50%);
            width: max-content;
            max-width: 260px;
            background: #1e2a36;
            color: #ffffff;
            font-family: sans-serif;
            font-style: normal;
            font-size: 12px;
            font-weight: 400;
            line-height: 1.35;
            padding: 0.55rem 0.7rem;
            border-radius: 10px;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.18);
            white-space: normal;
            text-align: left;
        }
        .tip-i .tip-bubble::before {
            content: "";
            position: absolute;
            left: 50%;
            top: -6px;
            transform: translateX(-50%);
            border-width: 0 6px 6px 6px;
            border-style: solid;
            border-color: transparent transparent #1e2a36 transparent;
        }
        .tip-i:hover .tip-bubble {
            visibility: visible;
            opacity: 1;
        }
        section.main > div.block-container div[data-testid="stVerticalBlockBorderWrapper"] {
            max-height: calc(100vh - 150px);
            overflow-y: auto;
            overflow-x: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _tip(label: str, text: str) -> None:
    """Label with a small circular i; details appear only when hovering the i."""
    safe_label = html.escape(label)
    safe_text = html.escape(text)
    st.markdown(
        f"""
        <div class="tip-row">
            <span>{safe_label}</span>
            <span class="tip-i" tabindex="0">i<span class="tip-bubble">{safe_text}</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _editor(samples) -> None:
    with st.container(border=True):
        if "input_mode" not in st.session_state:
            st.session_state["input_mode"] = INPUT_MODES[0]
        st.radio(
            "Input mode",
            INPUT_MODES,
            key="input_mode",
            help="Source Code compiles to TAC before optimization. Advanced TAC skips the source front end.",
        )
        _tip(
            "Program",
            "Edit source or TAC here. All five selectors start from the compiled or pasted program. "
            + OBSERVABILITY_NOTE,
        )
        if "source_input" not in st.session_state:
            st.session_state["source_input"] = ""
        if "tac_input" not in st.session_state:
            st.session_state["tac_input"] = ""
        st.slider(
            "Editor height",
            min_value=120,
            max_value=520,
            step=20,
            key="editor_height",
            help="Drag to give the TAC editor more or less vertical space.",
        )
        if st.session_state["input_mode"] == INPUT_MODES[0]:
            st.text_area(
                "Source",
                height=int(st.session_state.editor_height),
                placeholder=SOURCE_PLACEHOLDER,
                key="source_input",
                label_visibility="collapsed",
            )
        else:
            st.text_area(
                "TAC",
                height=int(st.session_state.editor_height),
                placeholder=PLACEHOLDER,
                key="tac_input",
                label_visibility="collapsed",
            )
        _tip(
            "Example",
            "Choose a saved program, then click Load Example. Selecting alone does not change the editor.",
        )
        example = st.selectbox(
            "Example",
            list(DEMO_SAMPLE_IDS),
            label_visibility="collapsed",
        )
        _tip(
            "Generate Random TAC",
            "Fills the editor with a small valid program that prints a result. Not added to the training set.",
        )
        st.button(
            "Generate Random TAC",
            on_click=_fill_random,
            use_container_width=True,
        )
        _tip(
            "Generate Random Source Code",
            "Fills Source Code mode with a small valid C-like program (not added to the training set).",
        )
        st.button(
            "Generate Random Source Code",
            on_click=_fill_random_source,
            use_container_width=True,
        )
        _tip(
            "Load Example",
            "Copies the selected TAC example into Advanced TAC mode.",
        )
        st.button(
            "Load Example",
            on_click=_fill_example,
            args=(samples, example),
            use_container_width=True,
        )
        _tip(
            "Compare All Selectors",
            "Compiles source (when selected) or parses TAC, then runs all five selectors "
            "on separate copies. Loads saved production models; does not train.",
        )
        if st.button(
            "Compare All Selectors",
            type="primary",
            use_container_width=True,
        ):
            _compare()
        _tip("Clear", "Empty the editor and hide the last comparison.")
        st.button(
            "Clear",
            on_click=_clear_editor,
            use_container_width=True,
        )


def _results_panel() -> None:
    _render_pipeline_flowchart(st.session_state.get("comparison") is not None)
    comparison = _current_comparison()
    with st.container(border=True):
        _panel_switcher()
        notice = st.session_state.get("notice")
        if notice:
            kind = st.session_state.get("notice_kind")
            if kind == "error":
                st.error(notice)
            elif kind == "info":
                st.info(notice)
            else:
                st.warning(notice)
        if comparison is None:
            st.markdown("**Awaiting a run**")
            st.caption(
                "Enter or generate source/TAC in the sidebar, then **Compare All Selectors**. "
                "Results appear below; use **Model Performance**, **Mathematics**, and "
                "**Optimization Analytics** tabs for saved experiments."
            )
            return
        if comparison.submitted_source:
            with st.expander("Source → TAC pipeline", expanded=True):
                st.markdown("**Original source**")
                st.code(comparison.submitted_source, language="c")
                st.markdown("**Generated TAC (validated)**")
                st.code(comparison.submitted_tac or "(empty)", language=None)
        panel = st.session_state.get("result_panel", "overview")
        if panel == "scores":
            _scores_tab(comparison)
        elif panel == "output":
            _output_tab(comparison)
        elif panel == "history":
            _history_tab(comparison)
        else:
            _overview_tab(comparison)


def _current_comparison():
    comparison = st.session_state.get("comparison")
    if comparison is None:
        return None
    mode = st.session_state.get("input_mode", INPUT_MODES[0])
    if mode == INPUT_MODES[0]:
        current = st.session_state.get("source_input", "").strip()
        expected = (comparison.submitted_source or "").strip()
        if current != expected:
            st.session_state["notice"] = (
                "The editor no longer matches the last comparison. Click Compare All Selectors to run again."
            )
            st.session_state["notice_kind"] = "info"
            return None
    else:
        current = st.session_state.get("tac_input", "").strip()
        if current != comparison.submitted_tac.strip():
            st.session_state["notice"] = (
                "The editor no longer matches the last comparison. Click Compare All Selectors to run this TAC."
            )
            st.session_state["notice_kind"] = "info"
            return None
    if st.session_state.get("notice_kind") == "info":
        st.session_state.pop("notice", None)
        st.session_state.pop("notice_kind", None)
    return comparison


def _panel_switcher() -> None:
    if "result_panel" not in st.session_state:
        st.session_state["result_panel"] = "overview"
    help_by_key = {key: text for key, _label, text in PANELS}
    active = st.session_state["result_panel"]
    _tip("Results", help_by_key.get(active, "Comparison results for the TAC on the left."))
    columns = st.columns(len(PANELS))
    for column, (key, label, _help_text) in zip(columns, PANELS):
        selected = active == key
        column.button(
            label,
            key=f"panel_{key}",
            type="primary" if selected else "secondary",
            use_container_width=True,
            on_click=_select_panel,
            args=(key,),
        )


def _selector_picker(state_key: str = "focus_selector") -> str:
    if state_key not in st.session_state:
        st.session_state[state_key] = "random_forest"
    _tip(
        "Selector",
        "Inspect one selector at a time. All five still ran on the same original TAC.",
    )
    labels = [SELECTOR_LABELS[key] for key in SELECTOR_ORDER]
    key_by_label = {SELECTOR_LABELS[key]: key for key in SELECTOR_ORDER}
    current = SELECTOR_LABELS.get(st.session_state[state_key], "Random Forest")
    chosen = st.radio(
        "Selector",
        labels,
        index=labels.index(current) if current in labels else 2,
        horizontal=True,
        label_visibility="collapsed",
        key=f"{state_key}_radio",
    )
    st.session_state[state_key] = key_by_label[chosen]
    return st.session_state[state_key]


def _select_panel(key: str) -> None:
    st.session_state["result_panel"] = key


def _fill_random() -> None:
    st.session_state["input_mode"] = INPUT_MODES[1]
    st.session_state["tac_input"] = load_random_tac()


def _fill_random_source() -> None:
    st.session_state["input_mode"] = INPUT_MODES[0]
    st.session_state["source_input"] = load_random_source()


def _render_pipeline_flowchart(ran_comparison: bool) -> None:
    status = "done" if ran_comparison else "ready"
    st.markdown("#### End-to-end pipeline")
    st.caption(
        "Source mode compiles through the front end; Advanced TAC skips straight to validation. "
        "After comparison, explore other tabs for frozen ML charts."
    )
    st.mermaid(
        f"""
        flowchart LR
            subgraph input [Your input]
                S[Source code]
                T[Advanced TAC]
            end
            subgraph frontend [source_frontend]
                L[Lexer]
                P[Parser / AST]
                G[TAC generator]
            end
            V[TAC parser validation]
            F[Feature extraction]
            SEL[Five selectors]
            ENG[Iterative optimizer]
            RES[Compare results]
            MP[Model Performance tab]
            MA[Mathematics tab]
            OA[Optimization Analytics tab]
            S --> L --> P --> G --> V
            T --> V
            V --> F --> SEL --> ENG --> RES
            RES --> MP
            RES --> MA
            RES --> OA
            classDef highlight fill:#6baed6,stroke:#2b6a94,color:#fff
            classDef idle fill:#eef4f8,stroke:#9fbcd0
            class {"RES,MP,MA,OA highlight" if status == "done" else "S,T highlight"}
        """
    )


def _fill_example(samples, example: str) -> None:
    st.session_state["input_mode"] = INPUT_MODES[1]
    st.session_state["tac_input"] = samples.get(example)


def _clear_editor() -> None:
    st.session_state["source_input"] = ""
    st.session_state["tac_input"] = ""
    st.session_state.pop("comparison", None)
    st.session_state.pop("notice", None)
    st.session_state.pop("notice_kind", None)


def _compare() -> None:
    st.session_state.pop("notice", None)
    st.session_state.pop("notice_kind", None)
    try:
        mode = st.session_state.get("input_mode", INPUT_MODES[0])
        if mode == INPUT_MODES[0]:
            source_text = st.session_state.get("source_input", "")
            compilation = compile_source_for_demo(source_text)
            tac_text = compilation.tac_text
            if "print" not in tac_text:
                st.session_state["notice"] = (
                    "Compiled TAC has no print statement. Dead-code elimination may remove every assignment. "
                    + OBSERVABILITY_NOTE
                )
                st.session_state["notice_kind"] = "warning"
            st.session_state["comparison"] = compare_all_selectors(
                tac_text,
                submitted_source=source_text.strip(),
            )
        else:
            program_text = st.session_state.get("tac_input", "").strip()
            if program_text and "print" not in program_text:
                st.session_state["notice"] = (
                    "This TAC has no print statement. Dead-code elimination may remove every assignment "
                    "and leave an empty program. " + OBSERVABILITY_NOTE
                )
                st.session_state["notice_kind"] = "warning"
            st.session_state["comparison"] = compare_all_selectors(program_text)
        st.session_state["result_panel"] = "overview"
        st.session_state["focus_selector"] = "random_forest"
    except (ModelArtifactsMissing, DemoError) as error:
        st.session_state.pop("comparison", None)
        st.session_state["notice"] = str(error)
        st.session_state["notice_kind"] = "error"
    except Exception as error:
        logger.exception("Unexpected demo error")
        st.session_state.pop("comparison", None)
        st.session_state["notice"] = f"An unexpected error occurred: {error}"
        st.session_state["notice_kind"] = "error"


def _overview_tab(comparison) -> None:
    _tip(
        "Overview",
        "Each selector started from the same TAC. Rule scores are not probabilities. "
        "A class probability is not a guarantee of optimization gain.",
    )
    cards = st.columns(len(SELECTOR_ORDER))
    for column, key in zip(cards, SELECTOR_ORDER):
        item = comparison.by_selector[key]
        run = item.run
        with column:
            with st.container(border=True):
                st.markdown(f"**{item.selector_label}**")
                st.metric("Instructions", f"{run.initial_instruction_count} → {run.final_instruction_count}")
                st.metric("Instruction gain", run.total_instruction_gain)
                st.metric("Weighted-cost gain", run.total_weighted_cost_gain)
                st.caption(
                    f"{run.successful_iterations} successful iterations, {run.candidate_attempts} attempts"
                )
    st.dataframe(pd.DataFrame(comparison_table_rows(comparison)), hide_index=True, use_container_width=True)
    _tip(
        "Features",
        "Eight core inputs extracted from the submitted TAC. Eleven-feature ML selectors "
        "also use the three extended Review-2 columns shown below when applicable.",
    )
    _feature_grid(comparison.features)
    extended_item = next(
        (
            comparison.by_selector[key]
            for key in SELECTOR_ORDER
            if len(comparison.by_selector[key].features) == len(FEATURE_COLUMNS_V2)
        ),
        None,
    )
    if extended_item is not None:
        extended = {
            name: extended_item.features[name]
            for name in FEATURE_COLUMNS_V2
            if name not in comparison.features
        }
        st.markdown("**Extended features (11-feature ML selectors)**")
        _feature_grid(extended)


def _feature_grid(features: dict[str, float]) -> None:
    names = list(features)
    if not names:
        return
    for start in range(0, len(names), 4):
        columns = st.columns(4)
        for column, name in zip(columns, names[start : start + 4]):
            value = features[name]
            shown = f"{float(value):.4f}" if name == "const_ratio" else str(int(value))
            with column:
                _tip(name, FEATURE_HELP.get(name, name))
                st.markdown(f"**{shown}**")


def _scores_tab(comparison) -> None:
    key = _selector_picker("focus_selector")
    item = comparison.by_selector[key]
    advice = item.initial_advice
    with st.container(border=True):
        if key == "rule_based":
            top = f"{advice.confidence:.4f} (rule score, not a probability)"
        else:
            top = f"{advice.confidence * 100:.1f}% class probability"
        st.markdown(f"**{item.selector_label}**")
        st.markdown(
            f"First predicted strategy: **{strategy_label(advice.selected_strategy)}** · top value {top}"
        )
        st.caption(f"Feature schema: {item.feature_schema}")
        _tip(
            "Scores / probabilities",
            "High values do not guarantee optimization gain. Rule-Based shows hand-written "
            "rule scores (not probabilities). Decision Tree, Random Forest, Logistic Regression, "
            "and MLP show softmax class probabilities for all seven labels, including "
            "no_improvement.",
        )
        figure = horizontal_score_figure(advice, key)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
        st.dataframe(
            pd.DataFrame(class_score_rows(advice, key)),
            hide_index=True,
            use_container_width=True,
        )


def _output_tab(comparison) -> None:
    key = _selector_picker("focus_selector")
    item = comparison.by_selector[key]
    run = item.run
    with st.container(border=True):
        st.markdown(f"**{item.selector_label}**")
        metrics = st.columns(4)
        metrics[0].metric("Initial", run.initial_instruction_count)
        metrics[1].metric("Final", run.final_instruction_count)
        metrics[2].metric("Instruction gain", run.total_instruction_gain)
        metrics[3].metric("Weighted-cost gain", run.total_weighted_cost_gain)
        sequence = " → ".join(strategy_label(name) for name in run.strategy_sequence) or "(none accepted)"
        st.caption(f"Strategy sequence: {sequence}. {item.stop_message}")
        left, right = st.columns(2)
        left.markdown("**Original TAC**")
        left.code(comparison.original_tac or "(empty)", language=None)
        optimized = item.optimized_tac.strip()
        right.markdown("**Optimized TAC**")
        if not optimized:
            right.info(
                "The optimized TAC is empty. That is the measured result: with no print or other "
                "observable use, dead-code elimination can remove every assignment."
            )
        right.code(optimized or "(empty program)", language=None)


def _history_tab(comparison) -> None:
    key = _selector_picker("focus_selector")
    item = comparison.by_selector[key]
    with st.container(border=True):
        st.markdown(f"**{item.selector_label}**")
        st.caption(item.stop_message)
        st.dataframe(
            pd.DataFrame(iteration_history_rows(item.run, key)),
            hide_index=True,
            use_container_width=True,
        )


def _about() -> None:
    with st.expander("About and model information"):
        st.markdown(
            f"""
This demo accepts **restricted C-like source** (default) or **Advanced TAC**. Source compiles through
`source_frontend/` (lexer → parser → AST → TAC generator → existing TAC parser) before optimization.
**This is not the missing external Project 1 parser.**

The UI ranks six optimizers, applies candidates in order, and repeats until convergence.
**No training happens here** — saved production artifacts load from `models/review2_production_*` when present.

**Review-2 dataset:** `dataset_v2.csv` (400 programs, generated offline).

**Selectors:** Rule-Based (8 features), Decision Tree, Random Forest, Logistic Regression,
and MLP. ML models use either the **original eight-feature schema** or the **extended
eleven-feature schema** depending on which production pickle was trained; the per-selector
panel shows which schema applies.

**Optimizers:** constant folding, constant propagation, algebraic simplification,
common subexpression elimination, dead code elimination, strength reduction.

Random Forest training config (offline): {FOREST_PARAMETERS.get("n_estimators", 100)} trees,
`random_state={MODEL_RANDOM_STATE}`. Environment: scikit-learn {sklearn.__version__}.
"""
        )


def _streamlit_active() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if _streamlit_active():
    main()
elif __name__ == "__main__":
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).resolve()
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]],
            cwd=str(app_path.parent),
        )
    )
