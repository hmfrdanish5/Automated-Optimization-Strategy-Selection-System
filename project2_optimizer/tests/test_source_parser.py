"""Parser tests for the restricted source language."""

from __future__ import annotations

import unittest

from source_frontend.errors import SourceCompileError
from source_frontend.lexer import lex
from source_frontend.parser import parse


class ParserTests(unittest.TestCase):
    def test_declaration_assignment_and_print(self) -> None:
        source = "int a = 1;\nprint(a);\n"
        program = parse(lex(source))
        self.assertEqual(len(program.body), 2)

    def test_operator_precedence_and_parentheses(self) -> None:
        source = "int a = 1;\nint b = 2;\nint c = a + b * 3;\nint d = (a + b) * 3;\n"
        program = parse(lex(source))
        self.assertEqual(len(program.body), 4)

    def test_unary_minus(self) -> None:
        source = "int a = -5;\nprint(-a);\n"
        program = parse(lex(source))
        self.assertEqual(len(program.body), 2)

    def test_if_else_and_while(self) -> None:
        source = """
        int a = 0;
        int b = 5;
        if (a < b) { print(a); } else { print(b); }
        while (a < 2) { a = a + 1; }
        """
        program = parse(lex(source))
        self.assertEqual(len(program.body), 4)

    def test_nested_blocks(self) -> None:
        source = """
        int a = 0;
        if (a == 0) {
            if (a < 1) { a = 1; }
        }
        """
        program = parse(lex(source))
        self.assertEqual(len(program.body), 2)

    def test_duplicate_declaration_is_rejected(self) -> None:
        with self.assertRaises(SourceCompileError):
            parse(lex("int a = 1;\nint a = 2;\n"))

    def test_undeclared_variable_is_rejected(self) -> None:
        with self.assertRaises(SourceCompileError):
            parse(lex("a = 1;\n"))

    def test_unsupported_for_loop_is_rejected(self) -> None:
        with self.assertRaises(SourceCompileError):
            parse(lex("for (int i = 0; i < 3; i = i + 1) { print(i); }\n"))


if __name__ == "__main__":
    unittest.main()
