"""Source-language compile errors with locations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceCompileError(Exception):
    """A lexer, parser, or codegen failure tied to the original source."""

    message: str
    line: int = 1
    column: int = 1

    def format(self) -> str:
        return f"line {self.line}, column {self.column}: {self.message}"


def format_errors(errors: list[SourceCompileError]) -> str:
    return "; ".join(error.format() for error in errors)
