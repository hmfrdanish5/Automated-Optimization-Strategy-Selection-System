"""Build dataset_v2 and train Review-2 production models."""

from __future__ import annotations

from dataset.v2_builder import audit_dataset_v2, write_dataset_v2
from dataset.v2_loader import load_dataset_v2_rows, pair_v2_examples
from evaluation.review2_experiment import fit_review2_production
from features.extended_extractor import FEATURE_COLUMNS_V2
from features.extractor import FEATURE_COLUMNS


def main() -> None:
    write_dataset_v2()
    rows = load_dataset_v2_rows()
    audit = audit_dataset_v2(rows)
    print(f"Wrote dataset_v2.csv with {audit['total_rows']} rows")
    fit_review2_production(
        pair_v2_examples(rows, FEATURE_COLUMNS),
        pair_v2_examples(rows, FEATURE_COLUMNS_V2),
    )
    print("Saved Review-2 production models under models/review2_production_*")


if __name__ == "__main__":
    main()
