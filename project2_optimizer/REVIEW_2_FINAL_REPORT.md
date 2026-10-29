# Review 2 — Final Report

## Objective and architecture

Review 2 completes the Automated Optimization Strategy Selection System while
keeping Review 1 frozen. The pipeline is unchanged at a high level: parse TAC,
extract features, rank optimizers, run the iterative engine with measured gains,
and report classification separately from optimization outcomes.

New work adds **Logistic Regression**, a small **MLP**, an **11-feature schema**,
`dataset/dataset_v2.csv` (400 programs), five-selector evaluation, Review-2
production models, frontend support, and `outputs/review2/` artifacts.

## Features

**Original eight** (`features/extractor.py`, `FEATURE_COLUMNS`) — unchanged.

**Extended three** (`features/extended_extractor.py`, appended to v1 order):

| Feature | Meaning |
| --- | --- |
| `num_constant_propagation_uses` | Sites where constant propagation can substitute a known integer constant (respects reassignment; same scope as the CP pass). |
| `num_safe_repeated_expr` | Binary expressions CSE can reuse under current reassignment-aware availability. |
| `num_mul_by_two_in_loops` | `x * 2` / `2 * x` inside loop bodies recognized by strength reduction. |

## Dataset

- **File:** `dataset/dataset_v2.csv` (not overwriting Review-1 CSVs).
- **Rows:** 400 (111 hand samples + 89 `fin_*` + 200 `v2_*` extension).
- **Seeds:** final block `GENERATION_SEED=42`, extension `V2_DATASET_SEED=43`.
- **Labels:** independent six-optimizer measurement (`dataset/labeler.py`).
- **Audit:** `outputs/review2/dataset_audit.json`
  - Unique 8-feature vectors: **333** (6 conflicting signatures).
  - Unique 11-feature vectors: **340** (0 conflicting signatures).
  - Class distribution: dead_code 129, CSE 63, no_improvement 63, constant_folding 45,
    algebraic 43, constant_propagation 40, strength_reduction 17.

## Selectors

1. **Rule-Based** — eight features only (no weights for v2 features).
2. **Decision Tree** — v1 or v2 schema per artifact.
3. **Random Forest** — v1 or v2.
4. **Logistic Regression** — `StandardScaler` + `LogisticRegression(solver=lbfgs, C=1.0, max_iter=3000, random_state=42)`.
5. **MLP** — `StandardScaler` + `MLPClassifier(hidden_layer_sizes=(32,16), max_iter=2000, alpha=0.001, random_state=42)`.

Coefficients: `outputs/review2/logistic_coefficients_v2.csv` (standardized; not causal).

## Evaluation methodology

- Strict **feature-grouped holdout** (`STRICT_SPLIT`, `random_state=42`).
- **Five-fold grouped CV** (`StratifiedGroupKFold` when possible).
- Scalers fit **inside** each training fold (pipelines for LR/MLP).
- Review-1 procedure and frozen artifacts untouched.

## Results — five selectors, 8 features, `dataset_v2`

**Strict holdout**

| Selector | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Rule-Based | 0.537 | 0.521 |
| Decision Tree | 0.939 | 0.911 |
| Random Forest | 0.927 | 0.879 |
| Logistic Regression | 0.939 | 0.913 |
| MLP | 0.939 | 0.931 |

**5-fold grouped CV (mean ± std, 8-feature grouping)**

| Selector | Accuracy | Macro F1 |
| --- | --- | --- |
| Rule-Based | 0.555 ± 0.046 | 0.536 ± 0.037 |
| Decision Tree | 0.915 ± 0.039 | 0.896 ± 0.048 |
| Random Forest | 0.905 ± 0.026 | 0.874 ± 0.041 |
| Logistic Regression | 0.910 ± 0.038 | 0.880 ± 0.057 |
| MLP | 0.917 ± 0.021 | 0.888 ± 0.035 |

## Eight vs eleven features (controlled ablation)

Holdout and CV ablations use **one partition grouped by eight-feature signatures**
(`outputs/review2/ablation_holdout_program_ids.json`: 318 train / 82 test program IDs
shared across schemas). Same labels and `random_state=42`.

**Shared holdout macro F1**

| Model | 8 features | 11 features |
| --- | ---: | ---: |
| Decision Tree | 0.911 | 1.000 |
| Random Forest | 0.879 | 0.988 |
| Logistic Regression | 0.913 | 0.991 |
| MLP | 0.931 | 0.979 |

**Shared 5-fold CV macro F1 (mean ± std)**

| Model | 8 features | 11 features |
| --- | --- | --- |
| Decision Tree | 0.896 ± 0.048 | 0.997 ± 0.008 |
| Random Forest | 0.874 ± 0.041 | 0.995 ± 0.007 |
| Logistic Regression | 0.880 ± 0.057 | 0.976 ± 0.024 |
| MLP | 0.888 ± 0.035 | 0.983 ± 0.018 |

The 11-feature schema removes label conflicts present under eight features on this
dataset, so high v2 scores are expected here and are not proof of generalization.

## Review 1 baseline (frozen, 200 rows) vs Review 2 (400 rows)

See `REVIEW_1_VS_REVIEW_2.md`. Review 1 reference holdout: RF accuracy 0.850 /
macro F1 0.810. Review 2 on the larger set with the same eight features shows
similar ranking (ML ≫ rules) with higher headline numbers on this synthetic corpus.

## Practical optimization and search efficiency

`outputs/review2/optimization_results.csv` — per holdout program, per selector:
instruction gain, weighted-cost gain, candidate attempts, failed attempts, strategy
sequences. Classification confidence remains separate from measured gains.

## Semantic regression

Training and evaluation read **`dataset/dataset_v2.csv`** via `dataset/v2_loader.py`
(rows are validated against reconstructed programs). Generation is explicit
(`python train_review2.py` only).

**Hand-written semantics:** `tests/test_semantics.py` checks observable output
for representative programs on each optimizer pass.

**Deterministic differential fuzz:** `tests/test_semantic_fuzz.py` uses fixed RNG
seeds (`20261029`, etc.) to generate valid TAC with arithmetic, branches, loops,
repeated expressions, and optional trapping dead divisions. Each case is run under
five boundary and five seeded-random input environments. The suite performs 4,800
single-pass comparisons (80 programs × 10 environments × 6 strategies) and 240
iterative-engine comparisons (24 programs × 10 environments via `FixedOrderSelector`).
Failures name the program index, strategy, and inputs via `subTest`. This is
stronger regression evidence, not formal verification.

Run `python -m unittest discover -s tests -p "test_*.py"` in the pinned environment
for the current full-suite count (187 tests as of this semantic-fuzz integration;
the historical Review-2 freeze reported 181 tests before this fuzz suite and
other post-freeze tests were added).

## Frontend

Streamlit (`python app.py`): five-selector comparison, schema caption per ML model
(8 vs 11 features), probabilities/scores for all seven classes including
`no_improvement`.

## Project 1 parser

**Not integrated.** No Project 1 parser exists in this repository. Stub:
`integration/project1_adapter.py` documents the required `TACProgram` interface.

## Reproduction

```text
cd project2_optimizer
python -m pip install -r requirements.txt
python train_review2.py
python evaluate_review2.py
python optimize.py
python -m unittest discover -s tests -p "test_*.py"
streamlit run app.py
```

Review-1 commands (`train.py`, `evaluate.py`) remain valid and unchanged.

## Dependencies

Pinned: `scikit-learn==1.9.1` (recorded in each saved model).

## Limitations

- Synthetic TAC only; extended features are conservative approximations of pass scope.
- Near-perfect v2 holdout metrics reflect resolved feature ambiguity on this dataset, not production readiness.
- MLP may emit convergence warnings on some folds; warnings are not suppressed.
- Rule-based selector does not use the three extended features.
