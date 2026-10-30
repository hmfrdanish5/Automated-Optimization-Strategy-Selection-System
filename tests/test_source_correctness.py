"""Regression tests for source compiler correctness fixes."""

from __future__ import annotations

import unittest

from source_frontend.compiler import compile_source
from source_frontend.errors import SourceCompileError
from source_frontend.lexer import lex
from source_frontend.parser import parse
from tac.interpreter import execute


class IfControlFlowTests(unittest.TestCase):
    def test_if_without_else_runs_then_when_true(self) -> None:
        source = "int a = 10;\nif (a > 5) { print(a); }\n"
        result = compile_source(source)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(execute(result.program).prints, (10,))

    def test_if_without_else_skips_then_when_false(self) -> None:
        source = "int a = 3;\nif (a > 5) { print(a); }\nprint(a);\n"
        result = compile_source(source)
        self.assertTrue(result.ok)
        self.assertEqual(execute(result.program).prints, (3,))

    def test_nested_if_without_else(self) -> None:
        source = """
        int a = 2;
        int b = 1;
        if (a > b) {
            if (b < 0) { print(b); }
            print(a);
        }
        """
        result = compile_source(source)
        self.assertTrue(result.ok)
        self.assertEqual(execute(result.program).prints, (2,))

    def test_if_else_both_paths(self) -> None:
        source = """
        int a = 1;
        int b = 2;
        if (a < b) { print(a); } else { print(b); }
        """
        result = compile_source(source)
        self.assertTrue(result.ok)
        self.assertEqual(execute(result.program).prints, (1,))
        source_false = """
        int a = 5;
        int b = 2;
        if (a < b) { print(a); } else { print(b); }
        """
        result_false = compile_source(source_false)
        self.assertTrue(result_false.ok)
        self.assertEqual(execute(result_false.program).prints, (2,))


class TempNameCollisionTests(unittest.TestCase):
    def test_user_t1_is_not_overwritten(self) -> None:
        source = """
        int t1 = 5;
        int a = 2;
        int b = a + 1;
        print(t1);
        print(b);
        """
        result = compile_source(source)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(execute(result.program).prints, (5, 3))
        self.assertIn("t1 = 5", result.tac_text)
        self.assertNotIn("t1 = a + 1", result.tac_text)

    def test_user_t2_and_generated_prefix_do_not_collide(self) -> None:
        source = """
        int t2 = 7;
        int x = 1;
        int y = x + 2;
        print(t2);
        print(y);
        """
        result = compile_source(source)
        self.assertTrue(result.ok)
        self.assertEqual(execute(result.program).prints, (7, 3))
        self.assertIn("__src_tmp_", result.tac_text)


class DefiniteAssignmentTests(unittest.TestCase):
    def test_initialized_in_both_branches_is_allowed(self) -> None:
        source = """
        int flag = 0;
        int x = 0;
        if (flag == 0) { x = 1; } else { x = 2; }
        print(x);
        """
        result = compile_source(source)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(execute(result.program).prints, (1,))

    def test_initialized_in_one_branch_is_rejected(self) -> None:
        source = """
        int flag = 0;
        if (flag == 0) { int x = 1; } else { }
        print(x);
        """
        result = compile_source(source)
        self.assertFalse(result.ok)

    def test_initialized_only_in_if_without_else_is_rejected(self) -> None:
        source = """
        int flag = 1;
        if (flag == 0) { int x = 10; }
        print(x);
        """
        result = compile_source(source)
        self.assertFalse(result.ok)

    def test_initialized_only_in_zero_iteration_while_is_rejected(self) -> None:
        source = """
        int i = 0;
        while (i < 0) { int x = 10; }
        print(x);
        """
        result = compile_source(source)
        self.assertFalse(result.ok)

    def test_use_after_declaration_in_same_block_is_allowed(self) -> None:
        source = "int x = 4;\nprint(x);\n"
        result = compile_source(source)
        self.assertTrue(result.ok)
        self.assertEqual(execute(result.program).prints, (4,))


class ReservedIdentifierTests(unittest.TestCase):
    def test_goto_identifier_is_rejected(self) -> None:
        with self.assertRaises(SourceCompileError) as ctx:
            parse(lex("int goto = 1;\n"))
        self.assertIn("reserved", str(ctx.exception))

    def test_function_declaration_error_is_clear(self) -> None:
        with self.assertRaises(SourceCompileError) as ctx:
            parse(lex("int main() { return 0; }\n"))
        self.assertIn("functions are not supported", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
