"""Compile restricted source text into a validated TACProgram."""

from __future__ import annotations

from dataclasses import dataclass, field

from source_frontend import ast_nodes as ast
from source_frontend.errors import SourceCompileError, format_errors
from source_frontend.lexer import lex
from source_frontend.parser import parse
from source_frontend.semantic_analysis import validate_definite_assignment
from source_frontend.tokens import Token
from source_frontend.tac_generator import generate_tac
from tac.parser import TACParseError, parse_tac
from tac.program import TACProgram


@dataclass
class CompilationResult:
    """Outcome of compiling source text."""

    source: str
    tokens: list[Token] = field(default_factory=list)
    ast: ast.Program | None = None
    tac_text: str = ""
    program: TACProgram | None = None
    errors: list[SourceCompileError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.program is not None and not self.errors


def compile_source(source: str) -> CompilationResult:
    """Lex, parse, generate TAC, and validate through the existing TAC parser."""
    result = CompilationResult(source=source)
    if not source.strip():
        result.errors.append(SourceCompileError("source input is empty"))
        return result
    try:
        result.tokens = lex(source)
    except SourceCompileError as error:
        result.errors.append(error)
        return result
    try:
        result.ast = parse(result.tokens)
    except SourceCompileError as error:
        result.errors.append(error)
        return result
    semantic_errors = validate_definite_assignment(result.ast)
    if semantic_errors:
        result.errors.extend(semantic_errors)
        return result
    try:
        result.tac_text = generate_tac(result.ast)
    except SourceCompileError as error:
        result.errors.append(error)
        return result
    try:
        result.program = parse_tac(result.tac_text)
    except TACParseError as error:
        line = error.line_number or 1
        result.errors.append(
            SourceCompileError(f"generated TAC failed validation: {error}", line, 1)
        )
    return result


def compile_source_or_raise(source: str) -> CompilationResult:
    result = compile_source(source)
    if not result.ok:
        raise SourceCompileError(format_errors(result.errors))
    return result
