# Automated Optimization Strategy Selection

## Introduction

This system chooses which compiler optimization to try next on a **TAC**
(three-address code) program. For each step it extracts features, ranks the six
optimizers (plus a **`no_improvement`** class when no single pass wins under the
label policy), tries candidates in order, measures real gains, and repeats until
convergence.

**Review 2 (current):** five selectors : Rule-Based, Decision Tree, Random Forest,
Logistic Regression, and MLP : trained on `dataset/dataset_v2.csv` (400 programs).
ML models may use the **original eight features** or the **extended eleven-feature**
schema; the Rule-Based selector always uses eight features. The Streamlit demo and
`optimize.py` load **saved production pickles**; they do not train.

**Review 1 (frozen baseline):** two ML models (tree + forest) on 200 programs.
Artifacts and metrics are preserved; see `REVIEW_1_BASELINE.md`.

TAC is the primary input. A Project 1 source parser is **not** bundled; see
`integration/project1_adapter.py` for the expected hook.

## Architecture

```text
TAC program
    → extract features (8 for rules; 8 or 11 for ML, per saved model)
    → selector ranks six optimizers + no_improvement
    → try candidates in order until one measurably improves the program
    → re-extract features and repeat
    → stop when no pass helps, iteration limit, or repeated TAC text
```

Each optimizer runs alone and returns a **new** program. The original TAC is
never modified.

## Review-1 frozen baseline

Historical scope (unchanged artifacts):

- `dataset/dataset.csv`, `dataset_expanded.csv`, `dataset_final.csv` (200-row final set).
- `models/production_decision_tree.pkl`, `models/production_random_forest.pkl`.
- Training/evaluation via `python train.py` and `python evaluate.py`.
- Reference metrics in `REVIEW_1_BASELINE.md` and `REVIEW_1_VS_REVIEW_2.md`.

## Features

**Eight original columns** (`features/extractor.py`, `FEATURE_COLUMNS`) : unchanged.

| Feature | Meaning |
| --- | --- |
| `num_statements` | All instructions, including labels, jumps, and prints. |
| `num_constants` | Binary ops with two integer operands (not `/` or `%` by zero), plus unary minus of an integer. Plain `x = 10` does not count. |
| `num_repeated_expr` | Extra copies of the same binary expression in written order (`a + b` ≠ `b + a`). |
| `num_dead_stmts` | Assignments removable by iterative liveness (same analysis as DCE). |
| `num_algebraic_ops` | Count of `x+0`, `0+x`, `x-0`, `x*1`, `1*x`, `x/1`, `x*0`, `0*x`. |
| `num_loop_ops` | Instructions inside a back-edge loop (counted once). |
| `num_tmp_vars` | Unique names matching `t` + digits as result or operand. |
| `const_ratio` | `num_constants / num_statements`, or 0 if empty. |

**Three extended columns** (`features/extended_extractor.py`, Review 2 only) document
optimizer-aligned opportunities (constant propagation uses, safe CSE sites, `×2` in loops).

## Optimizers

Each module exposes `optimize(program) -> TACProgram`:

| Name | Role |
| --- | --- |
| `constant_folding` | Evaluate constant arithmetic when safe (integer `/` truncates toward zero; no fold on divide/mod by zero). |
| `constant_propagation` | Substitute constants from simple assignments and copies (does not fold new expressions). |
| `algebraic_simplification` | Apply the eight algebraic identities listed above. |
| `common_subexpression_elimination` | Reuse temporaries for repeated expressions; respects reassignment; no commutativity. |
| `dead_code_elimination` | Remove dead assignments; keeps prints, labels, and jumps. |
| `strength_reduction` | Inside loops only: `x * 2` or `2 * x` → `x + x`. |

`apply_strategy(program, name)` runs one pass. Chaining happens only in the
iterative engine.

## Selectors (Review 2)

| Selector | Input schema | Notes |
| --- | --- | --- |
| Rule-Based | 8 features | Hand-written scores; not probabilities. |
| Decision Tree | 8 or 11 | Single interpretable tree. |
| Random Forest | 8 or 11 | 100 trees, `random_state=42`. |
| Logistic Regression | 8 or 11 | `StandardScaler` + multinomial logistic. |
| MLP | 8 or 11 | Small `MLPClassifier` with scaler. |

All ML selectors predict one of **seven** classes: the six optimizers or
`no_improvement`.

## Training

Labels come from `dataset/labeler.py` (independent optimizer runs). Only feature
columns enter `model.fit`; gain columns are diagnostic.

**Review 1**

```text
python train.py
python evaluate.py
```

**Review 2**

```text
python train_review2.py
python evaluate_review2.py
```

`train_review2.py` writes `dataset/dataset_v2.csv` once; evaluation loads the CSV
via `dataset/v2_loader.py` without regenerating rows.

## Prediction and confidence

`predict` / `predict_proba` rank candidates. Class probability is **not** a promise
of optimization gain; the engine still measures every candidate.

## Semantic correctness testing

Conservative optimizer rules (for example, not folding division by a variable
that might be zero, and keeping unused `/` or `%` when the divisor may be zero
during dead-code elimination) are intended to preserve semantics, but they are
not a formal proof.

**Offline differential testing** compares observable behavior before and after
optimization using the TAC interpreter (`tac/interpreter.py`, step limit
10,000): printed integers, runtime errors, and termination status.

| Suite | Role |
| --- | --- |
| `tests/test_semantics.py` | Hand-written regression cases for each of the six passes. |
| `tests/test_semantic_fuzz.py` | Deterministic generated corpus (seed `20261029`): 80 programs × 10 input environments × 6 passes (4,800 comparisons), plus 24 programs × 10 environments through the fixed-order iterative engine (240 comparisons). |

These tests provide evidence for the cases they exercise; they do **not** verify
every possible TAC program. The production engine does not re-run the interpreter
on the original program at optimization time.

## Important distinction

| Phase | What happens |
| --- | --- |
| **Training** | Fit on labeled CSV rows; labels come from running each optimizer once. |
| **Prediction** | Classify features of the **current** program to rank candidates. |
| **Optimization iteration** | Run optimizers, accept only measured improvements, update TAC, repeat. |

## Evaluation

- **Holdout** : strict grouped split by feature signature (`STRICT_SPLIT`).
- **Grouped cross-validation** : `StratifiedGroupKFold` when possible.
- Review-1 outputs under `outputs/`; Review-2 under `outputs/review2/`.

## Iterative engine

`python optimize.py` compares all five selectors on sample TAC (Review-2 production
models when present). Search-efficiency metrics are recorded per run.

## Limitations

- Synthetic dataset; not representative of production compilers.
- Simplified TAC (no functions, pointers, or rich types).
- Feature vectors can be ambiguous under eight columns; extended features reduce but
  do not remove all limits on this corpus.
- Project 1 parser integration is **not** implemented.

## Reproducing the project

From the `project2_optimizer` directory:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
python train_review2.py
python evaluate_review2.py
python optimize.py
python validate.py
```

Review-1 reproduction (frozen baseline only):

```text
python train.py
python evaluate.py
```

### Web demo (Streamlit)

```text
cd project2_optimizer
python app.py
```

Pinned: `scikit-learn==1.9.1`, `streamlit==1.63.0` (see `requirements.txt`).

## TAC syntax (quick reference)

One statement per line. Operators `+ - * / %`. Labels, `goto`, `if rel goto`,
`print`. Negative constants: `t1 = - 10`.

## Datasets

| File | Purpose |
| --- | --- |
| `dataset/dataset.csv` | Review-1 original (57 rows) : **frozen** |
| `dataset/dataset_expanded.csv` | Review-1 expanded : **frozen** |
| `dataset/dataset_final.csv` | Review-1 final (200 rows) : **frozen** |
| `dataset/dataset_v2.csv` | Review-2 (400 rows) |

## Review 2 deliverables

- Models: `models/review2_production_*` (v1 and v2 feature schemas).
- Reports: `REVIEW_2_FINAL_REPORT.md`, `REVIEW_1_VS_REVIEW_2.md`.
- Submission: `python package_submission.py` then `python scripts/verify_submission_zip.py`.
- Git root: this directory. A former nested duplicate was moved to `../_local_backups/` (local only).
