"""Structured three-address code (TAC) instructions.

TAC is the input to this project. A later phase can obtain these same
instructions from the source-language frontend or TAC parser. Optimizers should build new
instructions rather than editing old ones: each instruction is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class InstructionType(Enum):
    """Kinds of TAC statement this project understands."""

    ASSIGN = "assign"  # x = y, or x = 10
    BINARY = "binary"  # x = y + z
    UNARY = "unary"  # x = - y
    LABEL = "label"  # L1:
    GOTO = "goto"  # goto L1
    IF_GOTO = "if_goto"  # if a < b goto L1
    PRINT = "print"  # print x


# Unsigned integer written in the TAC text. Negative constants are unary minus.
_INT_LITERAL = re.compile(r"\d+\Z")

ASSIGNMENT_TYPES = (
    InstructionType.ASSIGN,
    InstructionType.BINARY,
    InstructionType.UNARY,
)


def is_int_literal(value: str | None) -> bool:
    """Return True when `value` is an unsigned integer token such as '0' or '20'."""
    return value is not None and _INT_LITERAL.fullmatch(value) is not None


def _variable_name(value: str | None) -> str | None:
    """Return `value` when it is a variable, and None for missing or integer tokens."""
    if value is None or is_int_literal(value):
        return None
    return value


@dataclass(frozen=True)
class TACInstruction:
    """One TAC statement.

    Fields that do not apply to a statement stay None. For example, a label
    uses only `instruction_type` and `target`, while a binary instruction uses
    `result`, `operand1`, `operator`, and `operand2`.
    """

    instruction_type: InstructionType
    result: str | None = None
    operand1: str | None = None
    operator: str | None = None
    operand2: str | None = None
    target: str | None = None

    def defined_name(self) -> str | None:
        """Variable this statement assigns, if it is an assignment."""
        if self.instruction_type in ASSIGNMENT_TYPES:
            return self.result
        return None

    def used_names(self) -> set[str]:
        """Variable names read by this statement. Integer literals are not names."""
        names: set[str] = set()
        for value in (self.operand1, self.operand2):
            name = _variable_name(value)
            if name is not None:
                names.add(name)
        return names

    def to_tac(self) -> str:
        """Render this instruction in the textual TAC syntax."""
        kind = self.instruction_type
        if kind is InstructionType.LABEL:
            return f"{self.target}:"
        if kind is InstructionType.GOTO:
            return f"goto {self.target}"
        if kind is InstructionType.IF_GOTO:
            return f"if {self.operand1} {self.operator} {self.operand2} goto {self.target}"
        if kind is InstructionType.PRINT:
            return f"print {self.operand1}"
        if kind is InstructionType.ASSIGN:
            return f"{self.result} = {self.operand1}"
        if kind is InstructionType.UNARY:
            return f"{self.result} = {self.operator} {self.operand1}"
        if kind is InstructionType.BINARY:
            return f"{self.result} = {self.operand1} {self.operator} {self.operand2}"
        raise AssertionError(f"unknown instruction type: {kind}")
