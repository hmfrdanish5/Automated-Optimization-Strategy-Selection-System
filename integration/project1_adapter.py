"""Adapter hook for an external Project 1 parser (not bundled in this repository).

If a Project 1 front-end becomes available, implement ``source_to_tac_program``
to return a ``TACProgram`` compatible with ``tac.parser.parse_tac``.
"""

from __future__ import annotations

from tac.program import TACProgram


class Project1ParserUnavailable(RuntimeError):
    """Raised when source integration is requested but no parser is installed."""


def source_to_tac_program(source: str) -> TACProgram:
    """Convert Project 1 source text into the Review-2 ``TACProgram`` representation."""
    raise Project1ParserUnavailable(
        "No Project 1 parser is present in this repository. "
        "Supply TAC directly or plug in a parser that emits the existing TAC grammar."
    )
