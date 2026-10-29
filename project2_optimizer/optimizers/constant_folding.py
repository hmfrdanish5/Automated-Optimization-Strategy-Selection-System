"""Constant folding: evaluate arithmetic whose operands are already literals.

This pass does not look at earlier assignments. `x = 10` followed by
`y = x + 5` is left unchanged; that is constant propagation's job.

`/` and `%` use truncating integer division (toward zero). A zero divisor
is left unchanged because the operation has no result in this language.
"""

from __future__ import annotations

from optimizers.base import constant_instruction
from tac.instruction import InstructionType, TACInstruction, is_int_literal
from tac.program import TACProgram


def optimize(program: TACProgram) -> TACProgram:
    """Return a new program with safe constant arithmetic evaluated."""
    return TACProgram([_fold(instruction) for instruction in program.instructions])


def _fold(instruction: TACInstruction) -> TACInstruction:
    if instruction.instruction_type is InstructionType.UNARY:
        return _fold_unary(instruction)
    if instruction.instruction_type is InstructionType.BINARY:
        return _fold_binary(instruction)
    return instruction


def _fold_unary(instruction: TACInstruction) -> TACInstruction:
    if instruction.operator != "-" or not is_int_literal(instruction.operand1):
        return instruction
    if instruction.result is None:
        return instruction
    return constant_instruction(instruction.result, -int(instruction.operand1))


def _fold_binary(instruction: TACInstruction) -> TACInstruction:
    if not (
        is_int_literal(instruction.operand1) and is_int_literal(instruction.operand2)
    ):
        return instruction
    if instruction.result is None or instruction.operator is None:
        return instruction
    value = _evaluate(
        instruction.operator,
        int(instruction.operand1),
        int(instruction.operand2),
    )
    if value is None:
        return instruction
    return constant_instruction(instruction.result, value)


def _evaluate(operator: str, left: int, right: int) -> int | None:
    """Evaluate one integer operation, or return None when it is undefined."""
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if right == 0:
            return None
        return _div_toward_zero(left, right)
    if operator == "%":
        if right == 0:
            return None
        quotient = _div_toward_zero(left, right)
        return left - quotient * right
    return None


def _div_toward_zero(left: int, right: int) -> int:
    """Integer division that truncates toward zero, including for negatives."""
    sign = -1 if (left < 0) ^ (right < 0) else 1
    return sign * (abs(left) // abs(right))
