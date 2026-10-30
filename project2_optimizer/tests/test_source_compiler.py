"""TAC generation and compiler end-to-end tests."""

from __future__ import annotations

import unittest

from source_frontend.compiler import compile_source
from source_frontend.errors import SourceCompileError
from tac.interpreter import execute
from integration.project1_adapter import source_to_tac_program


class TacGenerationTests(unittest.TestCase):
    def test_example_program_generates_valid_tac(self) -> None:
        source = """
        int a = 10;
        int b = 20;
        int c = a + b * 2;
        print(c);
        """
        result = compile_source(source)
        self.assertTrue(result.ok, result.errors)
        self.assertIn("print", result.tac_text)
        self.assertIsNotNone(result.program)
        self.assertEqual(execute(result.program).prints, (50,))

    def test_if_else_branching(self) -> None:
        source = """
        int a = 1;
        int b = 2;
        if (a < b) { print(a); } else { print(b); }
        """
        result = compile_source(source)
        self.assertTrue(result.ok)
        self.assertEqual(execute(result.program).prints, (1,))

    def test_while_loop_has_back_edge(self) -> None:
        source = """
        int i = 0;
        int sum = 0;
        while (i < 3) {
            sum = sum + i;
            i = i + 1;
        }
        print(sum);
        """
        result = compile_source(source)
        self.assertTrue(result.ok)
        self.assertIn("goto", result.tac_text)
        self.assertEqual(execute(result.program).prints, (3,))

    def test_division_by_zero_is_observable(self) -> None:
        source = """
        int a = 1;
        int b = 0;
        print(a);
        int c = a / b;
        print(c);
        """
        result = compile_source(source)
        self.assertTrue(result.ok)
        run = execute(result.program)
        self.assertEqual(run.prints, (1,))
        self.assertEqual(run.error, "division by zero")

    def test_adapter_entry_point(self) -> None:
        program = source_to_tac_program("int x = 3;\nprint(x);\n")
        self.assertEqual(execute(program).prints, (3,))

    def test_empty_source_reports_error(self) -> None:
        result = compile_source("   ")
        self.assertFalse(result.ok)
        self.assertTrue(result.errors)


if __name__ == "__main__":
    unittest.main()
