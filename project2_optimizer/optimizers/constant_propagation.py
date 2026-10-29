"""Constant propagation: substitute known constants into later uses.

A name is a known constant only when a statement assigns one directly:

- `x = 10`
- `x = - 10`
- `x = y`, when `y` is already a known constant

A binary instruction such as `y = x + 5` may become `y = 10 + 5`. The sum is
not computed here, so `y` does not become a new constant. Folding is a
separate pass.

Reassignment replaces the old constant. At a join, the name stays a constant
only when every analyzed path agrees on the same value. A loop is handled by
repeating that check until the sets stop changing.

Limitation: a negative constant is not written into a binary operand, a
condition, or a print. Those positions only accept an unsigned integer in
this TAC syntax, so the original variable is left in place. A copy can still
become unary minus, as in `y = - 10`.
"""

from __future__ import annotations

from tac.analysis import forward_dataflow
from tac.instruction import InstructionType, TACInstruction, is_int_literal
from tac.program import TACProgram
from optimizers.base import constant_instruction


def optimize(program: TACProgram) -> TACProgram:
    """Return a new program with safe constant uses substituted."""
    instructions = program.instructions
    states = forward_dataflow(instructions, _transfer, _meet)
    rewritten = [
        _rewrite(instruction, state)
        for instruction, state in zip(instructions, states)
    ]
    return TACProgram(rewritten)


def _meet(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Keep a constant only when both paths record the same value."""
    return {name: value for name, value in left.items() if right.get(name) == value}


def _transfer(in_state: dict[str, int], instruction: TACInstruction) -> dict[str, int]:
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


def _assigned_constant(
    instruction: TACInstruction, in_state: dict[str, int]
) -> int | None:
    """The constant this statement stores, if it stores one directly."""
    if instruction.instruction_type is InstructionType.ASSIGN:
        if is_int_literal(instruction.operand1):
            return int(instruction.operand1 or "0")
        if instruction.operand1 is None:
            return None
        return in_state.get(instruction.operand1)
    if (
        instruction.instruction_type is InstructionType.UNARY
        and instruction.operator == "-"
        and is_int_literal(instruction.operand1)
    ):
        return -int(instruction.operand1 or "0")
    return None


def _rewrite(instruction: TACInstruction, in_state: dict[str, int]) -> TACInstruction:
    if instruction.instruction_type is InstructionType.ASSIGN:
        return _rewrite_assign(instruction, in_state)
    if instruction.instruction_type is InstructionType.BINARY:
        return TACInstruction(
            instruction_type=InstructionType.BINARY,
            result=instruction.result,
            operand1=_substitute_unsigned(instruction.operand1, in_state),
            operator=instruction.operator,
            operand2=_substitute_unsigned(instruction.operand2, in_state),
        )
    if instruction.instruction_type is InstructionType.UNARY:
        return TACInstruction(
            instruction_type=InstructionType.UNARY,
            result=instruction.result,
            operator=instruction.operator,
            operand1=_substitute_unsigned(instruction.operand1, in_state),
        )
    if instruction.instruction_type is InstructionType.PRINT:
        return TACInstruction(
            instruction_type=InstructionType.PRINT,
            operand1=_substitute_unsigned(instruction.operand1, in_state),
        )
    if instruction.instruction_type is InstructionType.IF_GOTO:
        return TACInstruction(
            instruction_type=InstructionType.IF_GOTO,
            operand1=_substitute_unsigned(instruction.operand1, in_state),
            operator=instruction.operator,
            operand2=_substitute_unsigned(instruction.operand2, in_state),
            target=instruction.target,
        )
    return instruction


def _rewrite_assign(instruction: TACInstruction, in_state: dict[str, int]) -> TACInstruction:
    if is_int_literal(instruction.operand1) or instruction.result is None:
        return instruction
    value = in_state.get(instruction.operand1 or "")
    if value is None:
        return instruction
    return constant_instruction(instruction.result, value)


def _substitute_unsigned(token: str | None, in_state: dict[str, int]) -> str | None:
    """Replace a variable with an unsigned literal, when that is valid TAC."""
    if token is None or is_int_literal(token):
        return token
    value = in_state.get(token)
    if value is None or value < 0:
        return token
    return str(value)
