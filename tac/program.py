"""A TAC program is an ordered list of instructions."""

from __future__ import annotations

from dataclasses import dataclass

from tac.instruction import TACInstruction


@dataclass
class TACProgram:
    """Ordered TAC instructions.

    `instructions` is a new list when the program is copied, so a later
    optimizer can rebuild one program without changing another. The
    instructions themselves are frozen.
    """

    instructions: list[TACInstruction]

    def __len__(self) -> int:
        return len(self.instructions)

    def copy(self) -> TACProgram:
        """Return a program that does not share its instruction list."""
        return TACProgram(list(self.instructions))

    def to_tac(self) -> str:
        """Render the program as TAC text, one instruction per line."""
        return "\n".join(instruction.to_tac() for instruction in self.instructions)
