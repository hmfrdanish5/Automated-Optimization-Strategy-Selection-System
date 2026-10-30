"""Parser for textual TAC.

This is not a source-language parser. It only accepts the TAC syntax below.
Source and TAC inputs converge on the same `TACProgram` representation.

Grammar
-------

    program      ::= { line }
    line         ::= comment | label | statement | empty
    comment      ::= '#' { any character except newline }
    label        ::= identifier ':'
    statement    ::= assignment | goto | conditional | print
    assignment   ::= identifier '=' expression
    expression   ::= operand binary_op operand
                   | '-' operand
                   | operand
    operand      ::= identifier | integer
    integer      ::= digit { digit }
    binary_op    ::= '+' | '-' | '*' | '/' | '%'
    goto         ::= 'goto' identifier
    conditional  ::= 'if' operand relop operand 'goto' identifier
    relop        ::= '==' | '!=' | '<' | '<=' | '>' | '>='
    print        ::= 'print' operand
    identifier   ::= letter_or_underscore { letter_or_digit_or_underscore }

One statement per line. Blank lines and comments are ignored.

Assumptions
------------

- `/` and `%` are integer division and remainder. The parser accepts them
  and does not evaluate them. Constant folding does, and it refuses a zero divisor.
- A negative constant is written as unary minus: `t1 = - 10`.
  Binary operands are identifiers or unsigned integers, so `t1 = 1 + - 2`
  is rejected.
- `if`, `goto`, and `print` are reserved words and cannot be variable names.
- Labels must be unique, and every jump target must be a defined label.
"""

from __future__ import annotations

import re

from tac.instruction import InstructionType, TACInstruction
from tac.program import TACProgram


class TACParseError(Exception):
    """The TAC text does not match the grammar."""

    def __init__(self, message: str, line_number: int | None = None):
        self.line_number = line_number
        if line_number is None:
            super().__init__(message)
        else:
            super().__init__(f"line {line_number}: {message}")


_TOKEN = re.compile(
    r"""
        (?P<space>\s+)
      | (?P<comment>\#.*)
      | (?P<relop>==|!=|<=|>=|<|>)
      | (?P<symbol>[:+\-*/%=])
      | (?P<word>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<number>\d+)
      | (?P<bad>\S)
    """,
    re.VERBOSE,
)

_BINARY_OPS = {"+", "-", "*", "/", "%"}
_KEYWORDS = {"if", "goto", "print"}


def _tokenize(line: str) -> list[tuple[str, str]]:
    """Split one physical line into (kind, text) tokens."""
    tokens: list[tuple[str, str]] = []
    for match in _TOKEN.finditer(line):
        kind = match.lastgroup
        text = match.group()
        if kind in {"space", "comment"}:
            continue
        if kind == "bad":
            raise TACParseError(f"unexpected character {text!r}")
        if kind == "word" and text in _KEYWORDS:
            tokens.append(("keyword", text))
        elif kind == "word":
            tokens.append(("identifier", text))
        elif kind == "number":
            tokens.append(("number", text))
        elif kind == "relop":
            tokens.append(("relop", text))
        else:
            tokens.append(("symbol", text))
    return tokens


def _operand(token: tuple[str, str]) -> str:
    kind, text = token
    if kind not in {"identifier", "number"}:
        raise TACParseError(f"expected a variable or integer, found {text!r}")
    return text


def _identifier(token: tuple[str, str], role: str) -> str:
    kind, text = token
    if kind != "identifier":
        raise TACParseError(f"expected {role}, found {text!r}")
    return text


def _parse_statement(tokens: list[tuple[str, str]]) -> TACInstruction:
    first_kind, first_text = tokens[0]

    if first_kind == "keyword" and first_text == "goto":
        if len(tokens) != 2:
            raise TACParseError("goto must look like: goto L1")
        return TACInstruction(
            instruction_type=InstructionType.GOTO,
            target=_identifier(tokens[1], "a label name"),
        )

    if first_kind == "keyword" and first_text == "print":
        if len(tokens) != 2:
            raise TACParseError("print must look like: print x")
        return TACInstruction(
            instruction_type=InstructionType.PRINT,
            operand1=_operand(tokens[1]),
        )

    if first_kind == "keyword" and first_text == "if":
        # if <operand> <relop> <operand> goto <label>
        if (
            len(tokens) != 6
            or tokens[2][0] != "relop"
            or tokens[4] != ("keyword", "goto")
        ):
            raise TACParseError("if must look like: if a < b goto L1")
        return TACInstruction(
            instruction_type=InstructionType.IF_GOTO,
            operand1=_operand(tokens[1]),
            operator=tokens[2][1],
            operand2=_operand(tokens[3]),
            target=_identifier(tokens[5], "a label name"),
        )

    if first_kind == "identifier" and len(tokens) >= 2 and tokens[1] == ("symbol", ":"):
        if len(tokens) != 2:
            raise TACParseError("a label must occupy its own line, like: L1:")
        return TACInstruction(instruction_type=InstructionType.LABEL, target=first_text)

    if first_kind == "identifier" and len(tokens) >= 2 and tokens[1] == ("symbol", "="):
        return _parse_assignment(first_text, tokens[2:])

    raise TACParseError(f"could not parse statement starting with {first_text!r}")


def _parse_assignment(result: str, rhs: list[tuple[str, str]]) -> TACInstruction:
    if len(rhs) == 1:
        return TACInstruction(
            instruction_type=InstructionType.ASSIGN,
            result=result,
            operand1=_operand(rhs[0]),
        )

    if len(rhs) == 2 and rhs[0] == ("symbol", "-"):
        return TACInstruction(
            instruction_type=InstructionType.UNARY,
            result=result,
            operator="-",
            operand1=_operand(rhs[1]),
        )

    if len(rhs) == 3 and rhs[1][0] == "symbol" and rhs[1][1] in _BINARY_OPS:
        return TACInstruction(
            instruction_type=InstructionType.BINARY,
            result=result,
            operand1=_operand(rhs[0]),
            operator=rhs[1][1],
            operand2=_operand(rhs[2]),
        )

    raise TACParseError(
        "assignment must look like 'x = y', 'x = y + z', or 'x = - y'"
    )


def _check_labels(instructions: list[tuple[int, TACInstruction]]) -> None:
    """Reject duplicate labels and jumps to a label that was never defined.

    `instructions` pairs each statement with its physical source line number.
    """
    defined: dict[str, int] = {}
    for line_number, instruction in instructions:
        if instruction.instruction_type is not InstructionType.LABEL:
            continue
        name = instruction.target or ""
        if name in defined:
            raise TACParseError(
                f"label {name!r} is defined more than once",
                line_number=line_number,
            )
        defined[name] = line_number

    for line_number, instruction in instructions:
        if instruction.instruction_type not in {InstructionType.GOTO, InstructionType.IF_GOTO}:
            continue
        target = instruction.target or ""
        if target not in defined:
            raise TACParseError(
                f"jump target {target!r} is not defined",
                line_number=line_number,
            )


def parse_tac(source: str) -> TACProgram:
    """Parse TAC text into a program.

    Line numbers in error messages count physical lines, including blanks.
    """
    located: list[tuple[int, TACInstruction]] = []
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        try:
            tokens = _tokenize(raw_line)
            if not tokens:
                continue
            located.append((line_number, _parse_statement(tokens)))
        except TACParseError as error:
            if error.line_number is None:
                raise TACParseError(str(error), line_number) from None
            raise
    _check_labels(located)
    return TACProgram([instruction for _, instruction in located])
