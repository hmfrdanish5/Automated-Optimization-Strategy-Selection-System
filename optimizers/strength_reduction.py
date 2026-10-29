"""Strength reduction: replace a loop multiply-by-two with addition.

Supported pattern, and only inside a loop body as defined by
`num_loop_ops`:

    x * 2  →  x + x
    2 * x  →  x + x

`x` must be a variable. `2 * 2` is left alone so this pass does not do
constant folding. `x * 1` and `x * 0` are left alone so this pass does not
do algebraic simplification.

The rewrite is safe for this project's integers: there is no fixed-width
overflow, so `x + x` is the same value as `x * 2`. It does not reduce the
instruction count. It only changes the operator.

Not supported, because the current TAC does not prove them safely, or
because the rewrite would add instructions:

- multiplication by a constant other than 2
- division or remainder
- induction-variable rewrite such as replacing `i * n` with a running sum
- the same multiply-by-two outside a loop

A straight-line `a * 2` is therefore unchanged on purpose.
"""

from __future__ import annotations

from tac.analysis import loop_body_indexes
from tac.instruction import InstructionType, TACInstruction, is_int_literal
from tac.program import TACProgram


def optimize(program: TACProgram) -> TACProgram:
    """Return a new program with loop multiplications by two rewritten."""
    body = loop_body_indexes(program.instructions)
    rewritten = [
        _reduce(instruction) if index in body else instruction
        for index, instruction in enumerate(program.instructions)
    ]
    return TACProgram(rewritten)


def _reduce(instruction: TACInstruction) -> TACInstruction:
    if instruction.instruction_type is not InstructionType.BINARY:
        return instruction
    if instruction.operator != "*" or instruction.result is None:
        return instruction
    variable = _variable_multiplied_by_two(instruction.operand1, instruction.operand2)
    if variable is None:
        return instruction
    return TACInstruction(
        instruction_type=InstructionType.BINARY,
        result=instruction.result,
        operand1=variable,
        operator="+",
        operand2=variable,
    )


def _variable_multiplied_by_two(left: str | None, right: str | None) -> str | None:
    if _is_two(right) and not is_int_literal(left):
        return left
    if _is_two(left) and not is_int_literal(right):
        return right
    return None


def _is_two(value: str | None) -> bool:
    return is_int_literal(value) and int(value or "0") == 2
