"""Common subexpression elimination for binary TAC expressions.

`t3 = a + b` can become `t3 = t1` when an earlier instruction stored `a + b`
in `t1` and neither `a` nor `b` nor `t1` has been redefined on any path
that reaches `t3`.

`num_repeated_expr` is not used. That feature also counts a repeated
expression whose operand was assigned in between. This pass refuses that case.

`a + b` and `b + a` are different expressions, matching the feature key.
At a join, the expression is reused only when every path provides it in the
same variable. The check is repeated until loop back-edges settle.

Replacing a binary instruction with a copy does not change the instruction
count. It also does not delete the earlier computation: the copy still reads it.
"""

from __future__ import annotations

from tac.analysis import forward_dataflow
from tac.instruction import InstructionType, TACInstruction, is_int_literal
from tac.program import TACProgram


ExpressionKey = tuple[str, str, str]


def optimize(program: TACProgram) -> TACProgram:
    """Return a new program that reuses available binary results."""
    instructions = program.instructions
    available = forward_dataflow(instructions, _transfer, _meet)
    rewritten = [
        _rewrite(instruction, state)
        for instruction, state in zip(instructions, available)
    ]
    return TACProgram(rewritten)


def _meet(
    left: dict[ExpressionKey, str], right: dict[ExpressionKey, str]
) -> dict[ExpressionKey, str]:
    """Keep an expression only when both paths store it in the same variable."""
    return {key: name for key, name in left.items() if right.get(key) == name}


def _transfer(
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
    key = _expression_key(instruction)
    if key is not None and defined is not None and defined not in (key[1], key[2]):
        out_state[key] = defined
    return out_state


def _rewrite(
    instruction: TACInstruction, available: dict[ExpressionKey, str]
) -> TACInstruction:
    key = _expression_key(instruction)
    if key is None:
        return instruction
    previous = available.get(key)
    if previous is None or previous == instruction.result:
        return instruction
    return TACInstruction(
        instruction_type=InstructionType.ASSIGN,
        result=instruction.result,
        operand1=previous,
    )


def _expression_key(instruction: TACInstruction) -> ExpressionKey | None:
    if instruction.instruction_type is not InstructionType.BINARY:
        return None
    if instruction.operator is None:
        return None
    return (
        instruction.operator,
        _operand_key(instruction.operand1),
        _operand_key(instruction.operand2),
    )


def _operand_key(value: str | None) -> str:
    if value is not None and is_int_literal(value):
        return str(int(value))
    return value or ""
