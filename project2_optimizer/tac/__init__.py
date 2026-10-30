"""TAC input: textual parser and the structured program it produces."""

from tac.instruction import InstructionType, TACInstruction, is_int_literal
from tac.parser import TACParseError, parse_tac
from tac.program import TACProgram

__all__ = [
    "InstructionType",
    "TACInstruction",
    "TACParseError",
    "TACProgram",
    "is_int_literal",
    "parse_tac",
]
