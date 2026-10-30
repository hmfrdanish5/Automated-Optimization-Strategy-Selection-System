"""Token kinds for the restricted C-like source language."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenKind(Enum):
    INT = auto()
    PRINT = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    IDENT = auto()
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMI = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    EQEQ = auto()
    NE = auto()
    ASSIGN = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    line: int
    column: int
