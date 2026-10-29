"""Small helpers shared by the optimization passes.

Each pass exposes `optimize(program) -> TACProgram`. It builds new
instructions and a new program. It does not edit the program it was given.
"""

from __future__ import annotations

from tac.instruction import InstructionType, TACInstruction


def constant_instruction(result: str, value: int) -> TACInstruction:
    """Store `value` in `result`.

    A negative value is unary minus, because TAC integers in operand position
    are unsigned. Zero and positive values are plain copies of a literal.
    """
    if value < 0:
        return TACInstruction(
            instruction_type=InstructionType.UNARY,
            result=result,
            operator="-",
            operand1=str(-value),
        )
    return TACInstruction(
        instruction_type=InstructionType.ASSIGN,
        result=result,
        operand1=str(value),
    )
