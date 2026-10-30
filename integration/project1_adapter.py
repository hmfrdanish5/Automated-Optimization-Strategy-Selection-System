"""Adapter hook for converting source text into ``TACProgram``.

This project ships a restricted C-like parser in ``source_frontend``. It is
**not** the missing external Project 1 parser.
"""

from __future__ import annotations

from source_frontend.compiler import compile_source
from source_frontend.errors import format_errors
from tac.program import TACProgram


class SourceCompileFailure(RuntimeError):
    """Raised when source text does not compile."""


def source_to_tac_program(source: str) -> TACProgram:
    """Compile restricted source text into the Review-2 ``TACProgram`` representation."""
    result = compile_source(source)
    if not result.ok or result.program is None:
        raise SourceCompileFailure(format_errors(result.errors) or "compilation failed")
    return result.program
