"""Build optimization_dataset and train production models."""

from __future__ import annotations

from dataset.extended_builder import audit_dataset, write_dataset
from dataset.loader import load_dataset_rows, pair_dataset_examples
from evaluation.experiment import fit_production_models
from features.extended_extractor import FEATURE_COLUMNS_EXTENDED


def main() -> None:
    write_dataset()
    rows = load_dataset_rows()
    audit = audit_dataset(rows)
    print(f"Wrote optimization_dataset.csv with {audit['total_rows']} rows")
    fit_production_models(pair_dataset_examples(rows, FEATURE_COLUMNS_EXTENDED))
    print("Saved production models under models/production_*_11_features.pkl")


if __name__ == "__main__":
    main()
