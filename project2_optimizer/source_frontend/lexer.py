"""Deterministic lexer for the restricted C-like source language."""

from __future__ import annotations

from source_frontend.errors import SourceCompileError
from source_frontend.tokens import Token, TokenKind

_KEYWORDS = {
    "int": TokenKind.INT,
    "print": TokenKind.PRINT,
    "if": TokenKind.IF,
    "else": TokenKind.ELSE,
    "while": TokenKind.WHILE,
}

_TWO_CHAR = {
    "<=": TokenKind.LE,
    ">=": TokenKind.GE,
    "==": TokenKind.EQEQ,
    "!=": TokenKind.NE,
}

_ONE_CHAR = {
    "+": TokenKind.PLUS,
    "-": TokenKind.MINUS,
    "*": TokenKind.STAR,
    "/": TokenKind.SLASH,
    "%": TokenKind.PERCENT,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    ";": TokenKind.SEMI,
    "<": TokenKind.LT,
    ">": TokenKind.GT,
    "=": TokenKind.ASSIGN,
}


def lex(source: str) -> list[Token]:
    """Tokenize `source`, skipping whitespace and // comments."""
    tokens: list[Token] = []
    line = 1
    column = 1
    index = 0
    length = len(source)

    def emit(kind: TokenKind, text: str, start_line: int, start_column: int) -> None:
        tokens.append(Token(kind=kind, text=text, line=start_line, column=start_column))

    while index < length:
        char = source[index]
        if char in " \t\r":
            index += 1
            column += 1
            continue
        if char == "\n":
            index += 1
            line += 1
            column = 1
            continue
        if char == "/" and index + 1 < length and source[index + 1] == "/":
            index += 2
            column += 2
            while index < length and source[index] != "\n":
                index += 1
            continue

        start_line, start_column = line, column

        if char.isdigit():
            start = index
            while index < length and source[index].isdigit():
                index += 1
                column += 1
            text = source[start:index]
            emit(TokenKind.NUMBER, text, start_line, start_column)
            continue

        if char.isalpha() or char == "_":
            start = index
            index += 1
            column += 1
            while index < length and (source[index].isalnum() or source[index] == "_"):
                index += 1
                column += 1
            text = source[start:index]
            kind = _KEYWORDS.get(text, TokenKind.IDENT)
            emit(kind, text, start_line, start_column)
            continue

        if index + 1 < length:
            pair = source[index : index + 2]
            if pair in _TWO_CHAR:
                emit(_TWO_CHAR[pair], pair, start_line, start_column)
                index += 2
                column += 2
                continue

        if char in _ONE_CHAR:
            emit(_ONE_CHAR[char], char, start_line, start_column)
            index += 1
            column += 1
            continue

        raise SourceCompileError(f"invalid character {char!r}", line, column)

    emit(TokenKind.EOF, "", line, column)
    return tokens
