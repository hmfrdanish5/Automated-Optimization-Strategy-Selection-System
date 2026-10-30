"""Hand-written TAC samples and deterministic dataset utilities."""

from dataset.dataset_builder import (
    build_dataset,
    distribution_report,
    format_gain_table,
    write_dataset,
)
from dataset.examples import load_sample_programs, load_sample_sources
from dataset.labeler import label_program, label_row, training_features

__all__ = [
    "build_dataset",
    "distribution_report",
    "format_gain_table",
    "label_program",
    "label_row",
    "load_sample_programs",
    "load_sample_sources",
    "training_features",
    "write_dataset",
]
