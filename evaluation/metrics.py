"""Measurements used to compare one optimizer with the original program.

These helpers do not optimize. A positive gain means the optimized program
is smaller on that measure. Zero means no change. A negative gain means the
optimizer made that measure larger.

Instruction count is the primary project metric. The other counts exist so
that a rewrite which keeps the same number of instructions can still be
compared when instruction-count gains tie.
"""

from __future__ import annotations

import re

from evaluation.operation_cost import OPERATION_COST, UNARY_MINUS_COST
from tac.instruction import InstructionType
from tac.program import TACProgram

# Same temporary-name rule as num_tmp_vars: t followed by digits.
_TEMP_NAME = re.compile(r"t\d+\Z")


def instruction_count(program: TACProgram) -> int:
    """Return the number of TAC instructions, including labels, jumps, and prints."""
    return len(program.instructions)


def instruction_gain(original: TACProgram, optimized: TACProgram) -> int:
    """Return how many instructions were removed. Negative means the program grew."""
    return instruction_count(original) - instruction_count(optimized)


def reduction_percentage(original: TACProgram, optimized: TACProgram) -> float:
    """Return `gain / original_count * 100`, or 0 when the original is empty."""
    count = instruction_count(original)
    if count == 0:
        return 0.0
    return instruction_gain(original, optimized) / count * 100.0


def arithmetic_operation_count(program: TACProgram) -> int:
    """Count unary and binary arithmetic instructions.

    A copy, print, label, or branch is not an arithmetic operation.
    The comparison inside `if` is not counted either.
    """
    arithmetic = {InstructionType.BINARY, InstructionType.UNARY}
    return sum(1 for instruction in program.instructions if instruction.instruction_type in arithmetic)


def temporary_variable_count(program: TACProgram) -> int:
    """Count unique temporaries t1, t2, t3, and so on, used as a result or operand."""
    names: set[str] = set()
    for instruction in program.instructions:
        for value in (instruction.result, instruction.operand1, instruction.operand2):
            if value is not None and _TEMP_NAME.fullmatch(value):
                names.add(value)
    return len(names)


def code_size(program: TACProgram) -> int:
    """Return the number of characters in the rendered TAC, including newlines."""
    return len(program.to_tac())


def weighted_operation_cost(program: TACProgram) -> int:
    """Sum the relative costs of unary and binary arithmetic instructions.

    Copies, prints, labels, and branches contribute nothing. A comparison
    inside `if` is not an arithmetic operator. The weights live in
    `evaluation.operation_cost` and are not hardware cycle counts.
    """
    total = 0
    for instruction in program.instructions:
        if instruction.instruction_type is InstructionType.BINARY:
            total += OPERATION_COST.get(instruction.operator or "", 0)
        elif instruction.instruction_type is InstructionType.UNARY and instruction.operator == "-":
            total += UNARY_MINUS_COST
    return total


def weighted_cost_gain(original: TACProgram, optimized: TACProgram) -> int:
    """Return how much relative arithmetic cost was removed."""
    return weighted_operation_cost(original) - weighted_operation_cost(optimized)


def operation_gain(original: TACProgram, optimized: TACProgram) -> int:
    """Return how many arithmetic operations were removed."""
    return arithmetic_operation_count(original) - arithmetic_operation_count(optimized)


def temporary_gain(original: TACProgram, optimized: TACProgram) -> int:
    """Return how many unique temporary names disappeared."""
    return temporary_variable_count(original) - temporary_variable_count(optimized)


def code_size_gain(original: TACProgram, optimized: TACProgram) -> int:
    """Return how many rendered characters were removed."""
    return code_size(original) - code_size(optimized)
