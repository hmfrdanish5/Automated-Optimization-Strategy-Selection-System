# Review 1 vs Review 2

| Aspect | Review 1 (frozen) | Review 2 |
| --- | --- | --- |
| Dataset | `dataset_final.csv`, 200 rows | `dataset_v2.csv`, 400 rows |
| Features | 8 | 8 + 3 extended (optional per model) |
| Selectors | Rule, DT, RF | Rule, DT, RF, Logistic, MLP |
| Production models | `models/production_*.pkl` | `models/review2_production_*_v2.pkl` (preferred at runtime when present) |
| Evaluation outputs | `outputs/` | `outputs/review2/` |

## Reference metrics (do not regenerate Review-1 artifacts)

**Review 1 strict holdout**

| Selector | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Rule-Based | 0.600 | 0.554 |
| Decision Tree | 0.775 | 0.736 |
| Random Forest | 0.850 | 0.810 |

**Review 1 grouped CV**

| Selector | Accuracy | Macro F1 |
| --- | --- | --- |
| Rule-Based | 0.570 ± 0.074 | 0.546 ± 0.064 |
| Decision Tree | 0.880 ± 0.037 | 0.858 ± 0.054 |
| Random Forest | 0.890 ± 0.042 | 0.851 ± 0.086 |

## Review 2 comparable slice (8 features, `dataset_v2`, strict holdout)

| Selector | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Rule-Based | 0.537 | 0.521 |
| Decision Tree | 0.939 | 0.911 |
| Random Forest | 0.927 | 0.879 |
| Logistic Regression | 0.939 | 0.913 |
| MLP | 0.939 | 0.931 |

Datasets differ in size and composition; direct numeric comparison is indicative only.
