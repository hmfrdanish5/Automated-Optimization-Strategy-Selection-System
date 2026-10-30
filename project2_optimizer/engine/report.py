"""Text report for one optimization run."""

from __future__ import annotations

from engine.loop import OptimizationRun
from features.extractor import FEATURE_COLUMNS


def format_run(run: OptimizationRun) -> str:
    """The console trace for one selector and one program."""
    lines = [
        f"Selector: {run.selector_type}",
        f"Program: {run.program_id or '(unnamed)'}",
        "",
    ]
    for record in run.history:
        lines.extend(_format_iteration(record))
        lines.append("")
    sequence = " -> ".join(run.strategy_sequence) if run.strategy_sequence else "(none)"
    lines.extend(
        [
            "==================================================",
            "OPTIMIZATION COMPLETE",
            "==================================================",
            f"Initial instruction count: {run.initial_instruction_count}",
            f"Final instruction count: {run.final_instruction_count}",
            f"Total instruction reduction: {run.total_instruction_gain}",
            f"Total weighted-cost reduction: {run.total_weighted_cost_gain}",
            f"Successful optimization steps: {run.successful_iterations}",
            "",
            "Strategy sequence:",
            f"    {sequence}",
            "",
            "Stopped because:",
            f"    {run.stopped_because}",
            "",
            "Confidence is the selector's score for its top class.",
            "It is not a guarantee that the optimizer will improve the program.",
        ]
    )
    return "\n".join(lines)


def format_comparison(runs: list[OptimizationRun]) -> str:
    """One line per run: sequence, instruction counts, and both gains."""
    lines = [
        f"{'program':<22} {'selector':<16} {'before':>6} {'after':>6} "
        f"{'instr':>6} {'weighted':>8}  sequence",
    ]
    for run in runs:
        sequence = " -> ".join(run.strategy_sequence) if run.strategy_sequence else "(none)"
        lines.append(
            f"{run.program_id:<22} {run.selector_type:<16} "
            f"{run.initial_instruction_count:>6} {run.final_instruction_count:>6} "
            f"{run.total_instruction_gain:>6} {run.total_weighted_cost_gain:>8}  {sequence}"
        )
    return "\n".join(lines)


def _format_iteration(record) -> list[str]:
    lines = [
        "==================================================",
        f"Iteration {record.iteration}",
        "==================================================",
        "",
        "Features:",
    ]
    for column in FEATURE_COLUMNS:
        lines.append(f"    {column} = {record.features[column]}")
    lines.extend(["", "Selector scores:"])
    ordered = sorted(
        record.candidate_probabilities,
        key=lambda name: (-record.candidate_probabilities[name], name),
    )
    for name in ordered:
        lines.append(f"    {name} = {record.candidate_probabilities[name]:.2f}")
    lines.extend(
        [
            "",
            f"Top class: {record.selected_strategy}",
            f"Confidence: {record.confidence:.2f}",
            "",
            "Candidate ranking:",
        ]
    )
    for index, name in enumerate(record.strategy_ranking, start=1):
        lines.append(f"    {index}. {name}")
    lines.append("")
    for attempt in record.attempts:
        lines.extend(
            [
                "Trying:",
                f"    {attempt.strategy}",
                "",
                "Result:",
                f"    instructions: {attempt.instruction_count_before} -> {attempt.instruction_count_after}",
                f"    instruction gain: {attempt.instruction_gain}",
                f"    weighted-cost gain: {attempt.weighted_cost_gain}",
                f"    operation gain: {attempt.operation_gain}",
                f"    temporary gain: {attempt.temporary_gain}",
                f"    code-size gain: {attempt.code_size_gain}",
                "",
                "Accepted:",
                f"    {'yes' if attempt.accepted else 'no'} ({attempt.outcome})",
                "",
            ]
        )
    if record.accepted:
        lines.extend(["New TAC:", record.program_after])
    return lines
