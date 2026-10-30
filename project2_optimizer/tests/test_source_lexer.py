"""Lexer tests for the restricted source language."""

from __future__ import annotations

import unittest

from source_frontend.errors import SourceCompileError
from source_frontend.lexer import lex
from source_frontend.tokens import TokenKind


class LexerTests(unittest.TestCase):
    def test_keywords_and_identifiers(self) -> None:
        tokens = lex("int while if else print foo")
        kinds = [token.kind for token in tokens if token.kind is not TokenKind.EOF]
        self.assertEqual(
            kinds,
            [
                TokenKind.INT,
                TokenKind.WHILE,
                TokenKind.IF,
                TokenKind.ELSE,
                TokenKind.PRINT,
                TokenKind.IDENT,
            ],
        )
        self.assertEqual(tokens[0].text, "int")
        self.assertEqual(tokens[-2].text, "foo")

    def test_integer_and_operators(self) -> None:
        tokens = lex("a = 12 + 3 * 4 / 2 % 1;")
        texts = [token.text for token in tokens if token.kind is not TokenKind.EOF]
        self.assertIn("12", texts)
        self.assertIn("+", texts)
        self.assertIn("%", texts)

    def test_comments_and_whitespace_are_skipped(self) -> None:
        source = "// header\nint a = 1; // tail\n"
        tokens = [token for token in lex(source) if token.kind is not TokenKind.EOF]
        self.assertEqual(tokens[0].kind, TokenKind.INT)
        self.assertEqual(tokens[-1].text, ";")

    def test_line_and_column_tracking(self) -> None:
        tokens = lex("int a;\nint b;")
        b = next(token for token in tokens if token.text == "b")
        self.assertEqual(b.line, 2)
        self.assertEqual(b.column, 5)

    def test_invalid_character_raises(self) -> None:
        with self.assertRaises(SourceCompileError) as ctx:
            lex("int a = $;")
        self.assertIn("invalid character", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
