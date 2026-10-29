"""How much variety the eight feature columns actually contain.

Identical feature tuples are the same model input. This module counts them.
It does not drop rows and it does not add features.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from features.extractor import FEATURE_COLUMNS


def feature_signature(
    features: dict[str, float],
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple:
    """A hashable form of the model inputs for grouping and diversity reports.

    Count features are integers. `const_ratio` is rounded to 10 decimal
    places so 1/2 and 2/4 stay one signature, while distinct ratios stay apart.
    """
    columns = list(feature_columns or FEATURE_COLUMNS)
    signature = []
    for column in columns:
        value = features[column]
        if column == "const_ratio":
            signature.append(round(float(value), 10))
        else:
            signature.append(int(value))
    return tuple(signature)


@dataclass(frozen=True)
class SignatureRecord:
    """One feature tuple and the rows that share it."""

    signature: tuple
    count: int
    labels: tuple[str, ...]
    program_ids: tuple[str, ...]

    @property
    def labels_agree(self) -> bool:
        return len(set(self.labels)) <= 1


@dataclass(frozen=True)
class DiversityReport:
    """Counts and ranges for a labeled table of feature vectors."""

    row_count: int
    unique_feature_vectors: int
    duplicate_rows: int
    unique_values: dict[str, int]
    ranges: dict[str, tuple[float, float]]
    class_counts: dict[str, int]
    signatures: tuple[SignatureRecord, ...]

    @property
    def duplicate_signatures(self) -> tuple[SignatureRecord, ...]:
        return tuple(record for record in self.signatures if record.count > 1)

    @property
    def inconsistent_signatures(self) -> tuple[SignatureRecord, ...]:
        return tuple(record for record in self.signatures if not record.labels_agree)


def diversity_report(
    rows: list[dict[str, object]],
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> DiversityReport:
    """Summarize rows that already contain the feature columns and best_strategy."""
    columns = list(feature_columns or FEATURE_COLUMNS)
    grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[feature_signature({column: row[column] for column in columns}, columns)].append(row)
    records = []
    for signature, members in grouped.items():
        records.append(
            SignatureRecord(
                signature=signature,
                count=len(members),
                labels=tuple(str(member["best_strategy"]) for member in members),
                program_ids=tuple(str(member["program_id"]) for member in members),
            )
        )
    records.sort(key=lambda record: (-record.count, record.signature))
    unique_values = {
        column: len({_number(row[column], column) for row in rows})
        for column in columns
    }
    ranges = {
        column: (
            min(_number(row[column], column) for row in rows),
            max(_number(row[column], column) for row in rows),
        )
        for column in columns
    }
    return DiversityReport(
        row_count=len(rows),
        unique_feature_vectors=len(grouped),
        duplicate_rows=len(rows) - len(grouped),
        unique_values=unique_values,
        ranges=ranges,
        class_counts=dict(Counter(str(row["best_strategy"]) for row in rows)),
        signatures=tuple(records),
    )


def format_diversity_report(report: DiversityReport) -> str:
    """Text form of a diversity report, including conflicting labels."""
    lines = [
        f"Total samples: {report.row_count}",
        f"Unique feature vectors: {report.unique_feature_vectors}",
        f"Duplicate feature rows: {report.duplicate_rows}",
        "",
        "Unique values per feature:",
    ]
    feature_names = tuple(report.ranges.keys())
    for column in feature_names:
        low, high = report.ranges[column]
        lines.append(
            f"  {column}: {report.unique_values[column]} distinct, range {low} to {high}"
        )
    lines.extend(["", "Class counts:"])
    for name, count in sorted(report.class_counts.items()):
        lines.append(f"  {name}: {count}")
    lines.extend(["", f"Duplicate feature signatures: {len(report.duplicate_signatures)}"])
    for record in report.duplicate_signatures:
        label_text = ", ".join(sorted(set(record.labels)))
        lines.append(f"  count {record.count}, labels {label_text}, ids {', '.join(record.program_ids)}")
    if report.inconsistent_signatures:
        lines.extend(
            [
                "",
                "WARNING: the same feature vector maps to more than one best_strategy.",
                "Feature representation is not sufficient to uniquely determine the label for this signature.",
                "The eight features do not uniquely determine the label.",
            ]
        )
        for record in report.inconsistent_signatures:
            lines.append(
                f"  ids {', '.join(record.program_ids)}: labels {', '.join(record.labels)}"
            )
    else:
        lines.extend(
            [
                "",
                "Identical feature vectors have consistent labels.",
            ]
        )
    return "\n".join(lines)


def overlap_count(train_signatures: set[tuple], test_rows: list[dict[str, float]]) -> int:
    """How many test rows use a feature vector that also occurs in training."""
    return sum(1 for row in test_rows if feature_signature(row) in train_signatures)


def _number(value: object, column: str) -> float:
    if column == "const_ratio":
        return round(float(value), 10)
    return int(value)
