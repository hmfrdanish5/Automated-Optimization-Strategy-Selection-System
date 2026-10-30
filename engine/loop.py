"""Iterative optimization: rank, try, measure, accept.

The selector orders candidate optimizers. A candidate is accepted only when
`improves_on_original` says it is better than leaving the TAC unchanged.
That is the same comparison the labels use.

Two optional search controls exist for experiments and the demo. Both are off
by default, so the default behaviour is unchanged:

- ``stop_on_no_improvement``: stop before trying any optimizer when the
  selector's top class is ``no_improvement`` with at least this confidence.
  Without it, predicting ``no_improvement`` does not stop the loop; every run
  ends with a sweep over all six optimizers that confirms nothing helps.
- ``max_attempts_per_iteration``: try only the top-k ranked optimizers in each
  round. With a small budget a wrong ranking costs real improvement, so the
  quality of the prediction becomes visible in the final code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from dataset.labeler import score_strategy
from engine.selectors import SelectorAdvice, require_feature_keys
from evaluation.label_policy import NO_IMPROVEMENT
from evaluation.comparison import improves_on_original
from evaluation.metrics import instruction_count, instruction_gain, weighted_cost_gain
from features.extractor import FEATURE_COLUMNS, extract_features
from optimizers.registry import apply_strategy
from tac.program import TACProgram


MAX_ITERATIONS = 20

STOP_NO_BENEFIT = "no beneficial strategy remains"
STOP_ITERATION_LIMIT = "maximum iterations reached"
STOP_REPEATED_STATE = "repeated TAC state"
STOP_PREDICTED_NO_IMPROVEMENT = "selector predicted no improvement"
STOP_BUDGET_EXHAUSTED = "no improvement within the candidate budget"


@dataclass(frozen=True)
class CandidateAttempt:
    """One optimizer applied to a copy of the current program."""

    strategy: str
    instruction_count_before: int
    instruction_count_after: int
    instruction_gain: int
    weighted_cost_gain: int
    operation_gain: int
    temporary_gain: int
    code_size_gain: int
    accepted: bool
    outcome: str


@dataclass(frozen=True)
class IterationRecord:
    """Everything one pass of the loop needs for a later report or plot."""

    iteration: int
    features: dict[str, float]
    strategy_ranking: tuple[str, ...]
    candidate_probabilities: dict[str, float]
    selected_strategy: str
    confidence: float
    instruction_count_before: int
    instruction_count_after: int
    instruction_gain: int
    weighted_cost_gain: int
    operation_gain: int
    temporary_gain: int
    code_size_gain: int
    accepted: bool
    applied_strategy: str | None
    attempts: tuple[CandidateAttempt, ...]
    program_after: str


@dataclass(frozen=True)
class OptimizationRun:
    """One selector's complete pass over a copy of a program."""

    selector_type: str
    program_id: str
    initial_instruction_count: int
    final_instruction_count: int
    total_instruction_gain: int
    total_weighted_cost_gain: int
    successful_iterations: int
    strategy_sequence: tuple[str, ...]
    stopped_because: str
    history: tuple[IterationRecord, ...]
    candidate_attempts: int
    successful_candidate_attempts: int
    failed_candidate_attempts: int
    optimizer_evaluations_total: int
    final_program: TACProgram = field(repr=False)

    def summary(self) -> dict[str, object]:
        """The fields kept for a later comparison table."""
        return {
            "selector_type": self.selector_type,
            "program_id": self.program_id,
            "initial_instruction_count": self.initial_instruction_count,
            "final_instruction_count": self.final_instruction_count,
            "total_instruction_gain": self.total_instruction_gain,
            "total_weighted_cost_gain": self.total_weighted_cost_gain,
            "successful_iterations": self.successful_iterations,
            "strategy_sequence": list(self.strategy_sequence),
            "stopped_because": self.stopped_because,
            "candidate_attempts": self.candidate_attempts,
            "successful_candidate_attempts": self.successful_candidate_attempts,
            "failed_candidate_attempts": self.failed_candidate_attempts,
            "optimizer_evaluations_total": self.optimizer_evaluations_total,
        }

    def history_rows(self) -> list[dict[str, object]]:
        """Plain dictionaries so a later plot does not depend on these classes."""
        return [asdict(record) for record in self.history]


def optimize_program(
    program: TACProgram,
    selector,
    program_id: str = "",
    max_iterations: int = MAX_ITERATIONS,
    apply=apply_strategy,
    *,
    stop_on_no_improvement: float | None = None,
    max_attempts_per_iteration: int | None = None,
) -> OptimizationRun:
    """Optimize a copy of `program`. The caller's program is not modified.

    `apply(program, strategy_name)` runs one optimizer. Tests can pass a
    substitute. The default is the real registry.

    `stop_on_no_improvement` and `max_attempts_per_iteration` are optional
    search controls (see the module docstring). Both default to off.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if stop_on_no_improvement is not None and not 0.0 <= stop_on_no_improvement <= 1.0:
        raise ValueError("stop_on_no_improvement must be between 0 and 1")
    if max_attempts_per_iteration is not None and max_attempts_per_iteration < 1:
        raise ValueError("max_attempts_per_iteration must be at least 1")
    original_text = program.to_tac()
    current = program.copy()
    seen = {original_text}
    history: list[IterationRecord] = []
    sequence: list[str] = []
    stopped_because = STOP_ITERATION_LIMIT

    for iteration in range(1, max_iterations + 1):
        if hasattr(selector, "prepare_features"):
            features = selector.prepare_features(current)
        else:
            features = extract_features(current)
            require_feature_keys(features)
        advice = selector.advise(features)
        if (
            stop_on_no_improvement is not None
            and advice.selected_strategy == NO_IMPROVEMENT
            and advice.confidence >= stop_on_no_improvement
        ):
            current_count = instruction_count(current)
            history.append(
                _record(
                    iteration,
                    features,
                    advice,
                    current_count,
                    current_count,
                    None,
                    False,
                    None,
                    (),
                    current.to_tac(),
                )
            )
            stopped_because = STOP_PREDICTED_NO_IMPROVEMENT
            break
        record, accepted_program, stop_now = _run_iteration(
            iteration,
            current,
            features,
            advice,
            seen,
            apply,
            max_attempts_per_iteration,
        )
        history.append(record)
        if accepted_program is None:
            stopped_because = stop_now or STOP_NO_BENEFIT
            break
        current = accepted_program
        seen.add(current.to_tac())
        sequence.append(record.applied_strategy or "")
    else:
        stopped_because = STOP_ITERATION_LIMIT

    if program.to_tac() != original_text:
        raise RuntimeError("the engine changed the program it was given")

    search = search_metrics_from_history(tuple(history))
    return OptimizationRun(
        selector_type=advice.selector_type if history else getattr(selector, "selector_type", ""),
        program_id=program_id,
        initial_instruction_count=instruction_count(program),
        final_instruction_count=instruction_count(current),
        total_instruction_gain=instruction_gain(program, current),
        total_weighted_cost_gain=weighted_cost_gain(program, current),
        successful_iterations=len(sequence),
        strategy_sequence=tuple(sequence),
        stopped_because=stopped_because,
        history=tuple(history),
        candidate_attempts=search["candidate_attempts"],
        successful_candidate_attempts=search["successful_candidate_attempts"],
        failed_candidate_attempts=search["failed_candidate_attempts"],
        optimizer_evaluations_total=search["optimizer_evaluations_total"],
        final_program=current,
    )


def search_metrics_from_history(history: tuple[IterationRecord, ...]) -> dict[str, int]:
    """Count optimizer executions and accept/reject outcomes for one run.

    Each entry in ``record.attempts`` is one optimizer execution on a copy of
    the current program. ``candidate_attempts`` matches
    ``optimizer_evaluations_total`` in this engine because every execution is
    an attempted candidate.
    """
    evaluations = sum(len(record.attempts) for record in history)
    successful = sum(
        1 for record in history for attempt in record.attempts if attempt.accepted
    )
    failed = evaluations - successful
    return {
        "candidate_attempts": evaluations,
        "successful_candidate_attempts": successful,
        "failed_candidate_attempts": failed,
        "optimizer_evaluations_total": evaluations,
    }


def _run_iteration(
    iteration,
    current,
    features,
    advice: SelectorAdvice,
    seen,
    apply,
    max_attempts: int | None = None,
):
    before = instruction_count(current)
    current_text = current.to_tac()
    attempts: list[CandidateAttempt] = []
    accepted_program = None
    blocked_by_repeat = False
    candidates = advice.try_order if max_attempts is None else advice.try_order[:max_attempts]
    budget_limited = max_attempts is not None and len(candidates) < len(advice.try_order)

    for strategy in candidates:
        candidate = apply(current.copy(), strategy)
        if current.to_tac() != current_text:
            raise RuntimeError(f"{strategy} changed the program it was given")
        score = score_strategy(strategy, current, candidate)
        candidate_text = candidate.to_tac()
        if candidate_text == current_text:
            outcome = "unchanged TAC"
            accept = False
        elif candidate_text in seen:
            outcome = STOP_REPEATED_STATE
            accept = False
            if improves_on_original(score):
                blocked_by_repeat = True
        elif improves_on_original(score):
            outcome = "accepted"
            accept = True
        else:
            outcome = "not beneficial"
            accept = False
        attempts.append(
            CandidateAttempt(
                strategy=strategy,
                instruction_count_before=before,
                instruction_count_after=instruction_count(candidate),
                instruction_gain=score.instruction_gain,
                weighted_cost_gain=score.weighted_cost_gain,
                operation_gain=score.operation_gain,
                temporary_gain=score.temporary_gain,
                code_size_gain=score.code_size_gain,
                accepted=accept,
                outcome=outcome,
            )
        )
        if accept:
            accepted_program = candidate
            break

    accepted_attempt = next((attempt for attempt in attempts if attempt.accepted), None)
    if accepted_attempt is None:
        record = _record(
            iteration,
            features,
            advice,
            before,
            before,
            None,
            False,
            None,
            tuple(attempts),
            current_text,
        )
        if blocked_by_repeat:
            stop = STOP_REPEATED_STATE
        elif budget_limited:
            stop = STOP_BUDGET_EXHAUSTED
        else:
            stop = STOP_NO_BENEFIT
        return record, None, stop

    record = _record(
        iteration,
        features,
        advice,
        before,
        accepted_attempt.instruction_count_after,
        accepted_attempt,
        True,
        accepted_attempt.strategy,
        tuple(attempts),
        accepted_program.to_tac(),
    )
    return record, accepted_program, None


def _record(
    iteration,
    features,
    advice: SelectorAdvice,
    before,
    after,
    gain_source,
    accepted,
    applied,
    attempts,
    program_after,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        # The eight original columns first, then any extended columns the
        # selector used (an 11-feature model), so reports show its real inputs.
        features={
            **{column: features[column] for column in FEATURE_COLUMNS},
            **{name: value for name, value in features.items() if name not in FEATURE_COLUMNS},
        },
        strategy_ranking=advice.ranking,
        candidate_probabilities=dict(advice.probabilities),
        selected_strategy=advice.selected_strategy,
        confidence=advice.confidence,
        instruction_count_before=before,
        instruction_count_after=after,
        instruction_gain=0 if gain_source is None else gain_source.instruction_gain,
        weighted_cost_gain=0 if gain_source is None else gain_source.weighted_cost_gain,
        operation_gain=0 if gain_source is None else gain_source.operation_gain,
        temporary_gain=0 if gain_source is None else gain_source.temporary_gain,
        code_size_gain=0 if gain_source is None else gain_source.code_size_gain,
        accepted=accepted,
        applied_strategy=applied,
        attempts=attempts,
        program_after=program_after,
    )
