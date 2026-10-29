"""Algebraic simplification for the identities counted by num_algebraic_ops.

Each matching instruction becomes a copy or a constant. The instruction
count does not change. This pass does not evaluate general arithmetic:
`t1 = 10 + 20` is not an identity in the list below, so it stays for
constant folding.

Patterns, where `x` is a variable or an integer literal:

    x + 0,  0 + x  → x
    x - 0          → x
    x * 1,  1 * x  → x
    x / 1          → x
    x * 0,  0 * x  → 0

`0 - x` is not rewritten. `x / 0` is not rewritten. `0 * x` is safe here
because a TAC operand is already a value: reading `x` has no separate side
effect. The assignment that produced `x` is left for dead-code elimination
if this rewrite makes it unused.
"""

from __future__ import annotations

from optimizers.base import constant_instruction
from tac.instruction import InstructionType, TACInstruction, is_int_literal
from tac.program import TACProgram


def optimize(program: TACProgram) -> TACProgram:
    """Return a new program with the identities above applied."""
    return TACProgram([_simplify(instruction) for instruction in program.instructions])


def _simplify(instruction: TACInstruction) -> TACInstruction:
    if instruction.instruction_type is not InstructionType.BINARY:
        return instruction
    if instruction.result is None:
        return instruction
    operand = _identity_operand(instruction.operator, instruction.operand1, instruction.operand2)
    if operand is None:
        return instruction
    if operand == 0:
        return constant_instruction(instruction.result, 0)
    return TACInstruction(
        instruction_type=InstructionType.ASSIGN,
        result=instruction.result,
        operand1=operand,
    )


def _identity_operand(
    operator: str | None, left: str | None, right: str | None
) -> str | int | None:
    """Return the replacement operand, `0` for multiply-by-zero, or None."""
    if operator == "+" and _is_zero(right):
        return _as_operand(left)
    if operator == "+" and _is_zero(left):
        return _as_operand(right)
    if operator == "-" and _is_zero(right):
        return _as_operand(left)
    if operator == "*" and (_is_zero(left) or _is_zero(right)):
        return 0
    if operator == "*" and _is_one(right):
        return _as_operand(left)
    if operator == "*" and _is_one(left):
        return _as_operand(right)
    if operator == "/" and _is_one(right):
        return _as_operand(left)
    return None


def _as_operand(value: str | None) -> str:
    if value is not None and is_int_literal(value):
        return str(int(value))
    return value or ""


def _is_zero(value: str | None) -> bool:
    return is_int_literal(value) and int(value or "0") == 0


def _is_one(value: str | None) -> bool:
    return is_int_literal(value) and int(value or "0") == 1
