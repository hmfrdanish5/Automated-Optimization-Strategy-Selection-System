# Review-1 frozen baseline

This document records the Project 2 state frozen for Review 1. Later phases
should add new datasets (for example `dataset/dataset_v2.csv`) instead of
silently overwriting `dataset/dataset_final.csv`.

## System state

Completed modules:

- TAC representation and parser (`tac/`)
- Eight-feature extraction (`features/extractor.py`)
- Six independent optimizers (`optimizers/`)
- Automatic ground-truth labeling (`dataset/labeler.py`)
- Synthetic dataset generation (`dataset/generator.py`, three CSV generations)
- Rule-based selector (`selection/rule_based.py`)
- Decision Tree and Random Forest (`selection/classifiers.py`)
- Model probabilities and confidence (`predict_proba`, engine history)
- Strict grouped evaluation and grouped cross-validation (`selection/splitting.py`, `evaluation/folds.py`)
- Iterative optimization engine with candidate fallback (`engine/loop.py`)
- Semantic-equivalence interpreter and tests (`tac/interpreter.py`, `tests/test_semantics.py`)
- Unseen hand-written benchmark (`examples/unseen_benchmark.txt`)
- Required visualizations and audit outputs (`evaluation/plots.py`, `evaluation/validation.py`)
- Model persistence with scikit-learn version metadata (`selection/classifiers.py`)
- Search-efficiency instrumentation on optimization runs (`engine/loop.py`)

Deferred to later reviews: Logistic Regression, MLP, online learning, Project 1
parser integration, new ML features, new optimizer types, hyperparameter search.

## Dataset

| Artifact | Rows | Unique feature vectors | Notes |
| --- | ---: | ---: | --- |
| `dataset/dataset.csv` | 57 | 34 | Review-1 row-level demo split |
| `dataset/dataset_expanded.csv` | 111 | 108 | Expanded training experiment |
| `dataset/dataset_final.csv` | 200 | 199 | Primary Review-1 evaluation set |

`dataset_final.csv` has **0** conflicting feature signatures in the CSV. One
duplicate pair (`loop_heavy`, `fin_067`) shares the strength-reduction label.

Class distribution on the final set: dead_code_elimination 63, no_improvement 33,
common_subexpression_elimination 31, algebraic_simplification 22,
constant_folding 21, constant_propagation 20, strength_reduction 10.

## Model results

### Strict grouped holdout (40 test programs, overlap 0)

| Selector | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Rule-based | 0.600 | 0.554 |
| Decision Tree | 0.775 | 0.736 |
| Random Forest | 0.850 | 0.810 |

### 5-fold grouped cross-validation (mean +/- sample std)

| Selector | Accuracy | Macro F1 |
| --- | --- | --- |
| Rule-based | 0.570 +/- 0.074 | 0.546 +/- 0.064 |
| Decision Tree | 0.880 +/- 0.037 | 0.858 +/- 0.054 |
| Random Forest | 0.890 +/- 0.042 | 0.851 +/- 0.086 |

Model configuration: `DecisionTreeClassifier(random_state=42)`;
`RandomForestClassifier(n_estimators=100, random_state=42)`. No hyperparameter
search in Review 1.

Holdout models: `models/holdout_decision_tree.pkl`,
`models/holdout_random_forest.pkl`. Production models (full 200 rows, saved
after holdout metrics): `models/production_decision_tree.pkl`,
`models/production_random_forest.pkl`.

## Important findings

1. Random Forest outperforms the rule-based selector in classification on the
   current dataset.
2. Decision Tree is also substantially better than the rule baseline.
3. Different selectors can produce different strategy sequences while reaching
   the same final optimization gain on many programs (71 of 200 differ in
   sequence; gains match for rule, holdout tree, and holdout forest).
4. ML improves strategy-selection accuracy more clearly than it changes final
   code reduction under the fallback engine, because rejected candidates are
   retried until nothing helps.
5. Confidence has a moderate positive association with gain but is not a
   guarantee (Pearson on engine attempts roughly 0.34–0.54 depending on
   selector and gain type).
6. The eight features have known blind spots, especially around data-flow and
   operation-specific details (constant propagation without `num_constants`,
   killed repeated expressions, loops that are not `x * 2`).

Constructed limitation: two programs can share signature
`(5, 0, 1, 0, 0, 0, 2, 0.0)` and still measure different labels (`no_improvement`
versus CSE). Feature representation is not sufficient to uniquely determine
the label for this signature.

## Review-1 limitations

- Synthetic TAC programs, not production compiler workloads.
- Direct TAC input only; Project 1 source integration is future work.
- Eight coarse features; identical vectors can map to different labels.
- Instruction-count gain is primary for labeling; weighted operation costs are
  educational weights, not CPU cycles.
- Constant propagation has no dedicated feature; the rule baseline under-predicts CP.
- Models are not tuned for deployment as real compiler passes.
- Full-dataset runtime comparisons with holdout models include training rows;
  the unseen benchmark is the cleaner generalization check.

## Search efficiency (200 programs, holdout ML models)

Mean candidate attempts (optimizer executions per program): rule-based 8.500,
decision tree 8.460, random forest 8.015. Final instruction and weighted-cost
gains match across the three selectors; sequences differ on 71 programs. Fewer
attempts does not automatically mean better final quality. See
`outputs/search_efficiency.csv`.

## Reproduce

From `project2_optimizer`:

```text
python -m pip install -r requirements.txt
python train.py
python evaluate.py
python optimize.py
python validate.py
```

See `outputs/reproducibility_report.txt` for versions and seeds.
