"""Tests for the TAC representation and the textual TAC parser."""

from __future__ import annotations

import unittest

from tac.instruction import InstructionType
from tac.parser import TACParseError, parse_tac


class ParserTests(unittest.TestCase):
    def test_parses_assignment_binary_and_copy(self) -> None:
        program = parse_tac(
            """
            t1 = 10 + 20
            t2 = a + b
            t3 = a + b
            x = t3
            """
        )
        self.assertEqual(len(program), 4)
        first, second, third, fourth = program.instructions

        self.assertEqual(first.instruction_type, InstructionType.BINARY)
        self.assertEqual(first.result, "t1")
        self.assertEqual(first.operand1, "10")
        self.assertEqual(first.operator, "+")
        self.assertEqual(first.operand2, "20")

        self.assertEqual(second.operand1, "a")
        self.assertEqual(second.operand2, "b")
        self.assertEqual(third.operator, "+")
        self.assertEqual(fourth.instruction_type, InstructionType.ASSIGN)
        self.assertEqual(fourth.result, "x")
        self.assertEqual(fourth.operand1, "t3")

    def test_parses_control_flow_print_and_unary_minus(self) -> None:
        program = parse_tac(
            """
            # start
            n = 10
            L1:
            t1 = - n
            if t1 < n goto L1
            goto L2
            L2:
            print t1
            """
        )
        kinds = [instruction.instruction_type for instruction in program.instructions]
        self.assertEqual(
            kinds,
            [
                InstructionType.ASSIGN,
                InstructionType.LABEL,
                InstructionType.UNARY,
                InstructionType.IF_GOTO,
                InstructionType.GOTO,
                InstructionType.LABEL,
                InstructionType.PRINT,
            ],
        )
        conditional = program.instructions[3]
        self.assertEqual(conditional.operator, "<")
        self.assertEqual(conditional.target, "L1")
        self.assertEqual(program.instructions[2].operand1, "n")

    def test_round_trip_preserves_statements(self) -> None:
        source = "\n".join(
            [
                "t1 = 10 + 20",
                "t2 = - a",
                "L1:",
                "if t1 >= t2 goto L1",
                "print t1",
            ]
        )
        self.assertEqual(parse_tac(source).to_tac(), source)

    def test_copy_does_not_share_the_instruction_list(self) -> None:
        program = parse_tac("x = 1\nprint x\n")
        copied = program.copy()
        copied.instructions.append(parse_tac("y = 2").instructions[0])
        self.assertEqual(len(program), 2)
        self.assertEqual(len(copied), 3)

    def test_rejects_a_missing_jump_target(self) -> None:
        with self.assertRaises(TACParseError):
            parse_tac("goto L1\n")

    def test_rejects_a_duplicate_label(self) -> None:
        with self.assertRaises(TACParseError):
            parse_tac("L1:\nL1:\n")

    def test_rejects_a_malformed_assignment(self) -> None:
        with self.assertRaises(TACParseError):
            parse_tac("t1 = 1 + - 2\n")

    def test_empty_text_is_an_empty_program(self) -> None:
        program = parse_tac("\n# nothing\n\n")
        self.assertEqual(len(program), 0)
        self.assertEqual(program.to_tac(), "")


if __name__ == "__main__":
    unittest.main()
