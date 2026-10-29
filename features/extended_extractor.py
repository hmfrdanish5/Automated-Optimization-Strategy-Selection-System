"""Extended feature schema (Review 2).

FEATURE_COLUMNS_V2 keeps the original eight features in the same order and
appends three analysis features that mirror what the current optimizers can
actually use. They are conservative: if the pass would not rewrite a site,
it is not counted.
"""

from __future__ import annotations

from tac.analysis import forward_dataflow, loop_body_indexes
from tac.instruction import InstructionType, TACInstruction, is_int_literal
from tac.program import TACProgram

from features.extractor import FEATURE_COLUMNS, extract_features
from optimizers.common_subexpression_elimination import _expression_key as cse_expression_key
from optimizers.constant_propagation import _assigned_constant, _substitute_unsigned
from optimizers.strength_reduction import _variable_multiplied_by_two

FEATURE_SCHEMA_V1 = "v1"
FEATURE_SCHEMA_V2 = "v2"

FEATURE_COLUMNS_V2 = [
    *FEATURE_COLUMNS,
    "num_constant_propagation_uses",
    "num_safe_repeated_expr",
    "num_mul_by_two_in_loops",
]


def extract_features_v2(program: TACProgram) -> dict[str, float]:
    """Return eleven features in FEATURE_COLUMNS_V2 order."""
    features = extract_features(program)
    features["num_constant_propagation_uses"] = num_constant_propagation_uses(program)
    features["num_safe_repeated_expr"] = num_safe_repeated_expr(program)
    features["num_mul_by_two_in_loops"] = num_mul_by_two_in_loops(program)
    return {name: features[name] for name in FEATURE_COLUMNS_V2}


def num_constant_propagation_uses(program: TACProgram) -> int:
    """Count operand positions where constant propagation would substitute.

    Supported scope matches ``constant_propagation.optimize``:

    - constants from ``x = N``, ``x = -N``, or ``x = y`` when ``y`` is constant
    - substitution into binary operands, unary minus, print, and if-comparisons
    - only non-negative constants may replace a variable in operand positions
    - reassignment kills constant knowledge on that name
    - join points require the same constant on every path (forward dataflow meet)
    """
    instructions = program.instructions
    states = forward_dataflow(instructions, _cp_transfer, _cp_meet)
    count = 0
    for instruction, state in zip(instructions, states):
        count += _cp_rewrite_count(instruction, state)
    return count


def num_safe_repeated_expr(program: TACProgram) -> int:
    """Count binary sites where CSE could reuse an available expression.

    Uses the same expression key and availability rules as
    ``common_subexpression_elimination`` (including operand reassignment
    invalidation and join agreement). Does not assume commutativity.
    """
    instructions = program.instructions
    available_states = forward_dataflow(instructions, _cse_transfer, _cse_meet)
    count = 0
    for instruction, state in zip(instructions, available_states):
        key = cse_expression_key(instruction)
        if key is None:
            continue
        previous = state.get(key)
        if previous is None or previous == instruction.result:
            continue
        count += 1
    return count


def num_mul_by_two_in_loops(program: TACProgram) -> int:
    """Count ``x * 2`` / ``2 * x`` inside strength-reduction loop bodies."""
    body = loop_body_indexes(program.instructions)
    count = 0
    for index, instruction in enumerate(program.instructions):
        if index not in body:
            continue
        if instruction.instruction_type is not InstructionType.BINARY:
            continue
        if instruction.operator != "*":
            continue
        if _variable_multiplied_by_two(instruction.operand1, instruction.operand2) is not None:
            count += 1
    return count


def _cp_meet(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {name: value for name, value in left.items() if right.get(name) == value}


def _cp_transfer(in_state: dict[str, int], instruction: TACInstruction) -> dict[str, int]:
    out_state = dict(in_state)
    defined = instruction.defined_name()
    if defined is None:
        return out_state
    value = _assigned_constant(instruction, in_state)
    if value is None:
        out_state.pop(defined, None)
    else:
        out_state[defined] = value
    return out_state


def _cp_rewrite_count(instruction: TACInstruction, in_state: dict[str, int]) -> int:
    if instruction.instruction_type is InstructionType.ASSIGN:
        if is_int_literal(instruction.operand1) or instruction.result is None:
            return 0
        if in_state.get(instruction.operand1 or "") is not None:
            return 1
        return 0
    if instruction.instruction_type is InstructionType.BINARY:
        return _token_change_count(instruction.operand1, in_state) + _token_change_count(
            instruction.operand2, in_state
        )
    if instruction.instruction_type is InstructionType.UNARY:
        return _token_change_count(instruction.operand1, in_state)
    if instruction.instruction_type is InstructionType.PRINT:
        return _token_change_count(instruction.operand1, in_state)
    if instruction.instruction_type is InstructionType.IF_GOTO:
        return _token_change_count(instruction.operand1, in_state) + _token_change_count(
            instruction.operand2, in_state
        )
    return 0


def _token_change_count(token: str | None, in_state: dict[str, int]) -> int:
    if token is None or is_int_literal(token):
        return 0
    substituted = _substitute_unsigned(token, in_state)
    return 1 if substituted != token else 0


ExpressionKey = tuple[str, str, str]


def _cse_meet(
    left: dict[ExpressionKey, str], right: dict[ExpressionKey, str]
) -> dict[ExpressionKey, str]:
    return {key: name for key, name in left.items() if right.get(key) == name}


def _cse_transfer(
    in_state: dict[ExpressionKey, str], instruction: TACInstruction
) -> dict[ExpressionKey, str]:
    out_state = dict(in_state)
    defined = instruction.defined_name()
    if defined is not None:
        out_state = {
            key: name
            for key, name in out_state.items()
            if name != defined and defined not in (key[1], key[2])
        }
    key = cse_expression_key(instruction)
    if key is not None and defined is not None and defined not in (key[1], key[2]):
        out_state[key] = defined
    return out_state
