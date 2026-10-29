"""Numerical features of a TAC program.

These eight numbers are the only model inputs. The order in FEATURE_COLUMNS
is the order used everywhere else. Feature extraction only reads the program;
it does not optimize it.

The counts describe the program as it is now. They are hints for choosing a
strategy. They are not themselves a proof that an optimization is safe.
"""

from __future__ import annotations

import re

from tac.analysis import loop_body_indexes, remove_dead_assignments
from tac.instruction import InstructionType, TACInstruction, is_int_literal
from tac.program import TACProgram


FEATURE_COLUMNS = [
    "num_statements",
    "num_constants",
    "num_repeated_expr",
    "num_dead_stmts",
    "num_algebraic_ops",
    "num_loop_ops",
    "num_tmp_vars",
    "const_ratio",
]

# Temporary names are exactly t followed by digits: t1, t2, t10. Names such as
# temp, total, or t do not count.
_TEMP_NAME = re.compile(r"t\d+\Z")

# Binary operators for which two integer operands can be evaluated.
# Division and remainder by zero are not counted: there is no safe result.
_FOLDABLE_OPERATORS = {"+", "-", "*", "/", "%"}


def extract_features(program: TACProgram) -> dict[str, float]:
    """Return the eight features, with keys in FEATURE_COLUMNS order.

    Integer-valued features are returned as ints. `const_ratio` is a float.
    """
    statements = num_statements(program)
    constants = num_constants(program)
    features: dict[str, float] = {
        "num_statements": statements,
        "num_constants": constants,
        "num_repeated_expr": num_repeated_expr(program),
        "num_dead_stmts": num_dead_stmts(program),
        "num_algebraic_ops": num_algebraic_ops(program),
        "num_loop_ops": num_loop_ops(program),
        "num_tmp_vars": num_tmp_vars(program),
        "const_ratio": const_ratio(constants, statements),
    }
    return {name: features[name] for name in FEATURE_COLUMNS}


def feature_vector(program: TACProgram) -> list[float]:
    """Return the feature values in FEATURE_COLUMNS order."""
    features = extract_features(program)
    return [float(features[name]) for name in FEATURE_COLUMNS]


def num_statements(program: TACProgram) -> int:
    """Count every parsed instruction, including labels, jumps, and prints."""
    return len(program.instructions)


def num_constants(program: TACProgram) -> int:
    """Count statements whose arithmetic can be evaluated now.

    A statement counts when either:

    - it is binary, both operands are unsigned integers, the operator is
      `+ - * / %`, and the operator is not `/` or `%` with a zero divisor
    - it is unary minus of an unsigned integer, such as `t1 = - 10`

    A plain copy of a constant, such as `x = 10`, does not count. Only one
    operand being constant, such as `y = x + 5`, does not count either.
    Conditions such as `if 1 < 2 goto L1` do not count.
    """
    count = 0
    for instruction in program.instructions:
        if _is_foldable_binary(instruction) or _is_foldable_unary(instruction):
            count += 1
    return count


def num_repeated_expr(program: TACProgram) -> int:
    """Count extra copies of the same binary expression.

    `t1 = a + b`, `t2 = a + b`, and `t3 = a + b` contribute 2: the first
    occurrence is the original, and each later copy adds one.

    The key is `(operator, operand1, operand2)` in the written order.
    `a + b` and `b + a` are different. Integer literals are compared by
    numeric value, so `a + 0` and `a + 00` match.

    This count is textual. It does not notice that a later assignment may
    have changed an operand. The CSE optimizer must check that itself.
    Copies such as `t3 = t1` are not expressions and do not count.
    """
    seen: dict[tuple[str, str, str], int] = {}
    for instruction in program.instructions:
        if instruction.instruction_type is not InstructionType.BINARY:
            continue
        key = (
            instruction.operator or "",
            _expression_operand(instruction.operand1),
            _expression_operand(instruction.operand2),
        )
        seen[key] = seen.get(key, 0) + 1
    return sum(count - 1 for count in seen.values() if count > 1)


def num_dead_stmts(program: TACProgram) -> int:
    """Count assignments that iterative dead-code elimination would remove.

    An assignment is dead when its result is not live after the statement.
    A name is live when some print, condition, or surviving assignment still
    needs it. Prints, labels, and jumps are never counted, because they are
    control flow or observable output.

    The count repeats until nothing more is dead. That matters for a chain:

        x = 10
        z = x + 1
        print 0

    `z` is unused, so it is dead. After it is gone, `x` is unused too, so
    both statements count. A use that exists only inside an already-dead
    statement does not keep the earlier assignment alive.

    Liveness follows the control-flow graph, including jumps back to a label.
    A value that reaches a later use around a loop is live.

    The count is exactly the number of instructions `dead_code_elimination`
    deletes. Both call `remove_dead_assignments`.
    """
    kept = remove_dead_assignments(program.instructions)
    return len(program.instructions) - len(kept)


def num_algebraic_ops(program: TACProgram) -> int:
    """Count binary expressions that match a safe algebraic identity.

    Implemented patterns, for any operand `x` (variable or integer):

        x + 0,  0 + x
        x - 0
        x * 1,  1 * x
        x / 1
        x * 0,  0 * x

    `0 - x` is not included: it is not the same value as `x`.
    `x / 0` is not included. Adding the commutative forms of `+` and `*`
    is safe for the integer arithmetic this project uses. Subtraction and
    division are not commutative, so only the forms above are counted.
    """
    return sum(1 for instruction in program.instructions if _is_algebraic(instruction))


def num_loop_ops(program: TACProgram) -> int:
    """Count instructions that sit inside at least one loop body.

    A loop is a jump, conditional or not, whose target label appears at or
    before the jump. The body is that range, including the label and the
    jump. An instruction in two overlapping loops is counted once.
    A forward jump is not a loop. Strength reduction uses this same set.
    """
    return len(loop_body_indexes(program.instructions))


def num_tmp_vars(program: TACProgram) -> int:
    """Count unique temporary names matching t1, t2, t3, and so on.

    A name counts when it appears as a result or as an operand. Label names
    are not scanned, so a label called `L1` is not a temporary.
    """
    names: set[str] = set()
    for instruction in program.instructions:
        for value in (instruction.result, instruction.operand1, instruction.operand2):
            if value is not None and _TEMP_NAME.fullmatch(value):
                names.add(value)
    return len(names)


def const_ratio(num_constant_statements: int, statement_count: int) -> float:
    """Return num_constants / num_statements, or 0 when there are no statements."""
    if statement_count == 0:
        return 0.0
    return num_constant_statements / statement_count


def _expression_operand(value: str | None) -> str:
    if value is not None and is_int_literal(value):
        return str(int(value))
    return value or ""


def _is_foldable_binary(instruction: TACInstruction) -> bool:
    if instruction.instruction_type is not InstructionType.BINARY:
        return False
    if instruction.operator not in _FOLDABLE_OPERATORS:
        return False
    if not (is_int_literal(instruction.operand1) and is_int_literal(instruction.operand2)):
        return False
    if instruction.operator in {"/", "%"} and int(instruction.operand2 or "0") == 0:
        return False
    return True


def _is_foldable_unary(instruction: TACInstruction) -> bool:
    return (
        instruction.instruction_type is InstructionType.UNARY
        and instruction.operator == "-"
        and is_int_literal(instruction.operand1)
    )


def _is_zero(value: str | None) -> bool:
    return is_int_literal(value) and int(value or "0") == 0


def _is_one(value: str | None) -> bool:
    return is_int_literal(value) and int(value or "0") == 1


def _is_algebraic(instruction: TACInstruction) -> bool:
    if instruction.instruction_type is not InstructionType.BINARY:
        return False
    operator = instruction.operator
    left = instruction.operand1
    right = instruction.operand2
    if operator == "+" and (_is_zero(left) or _is_zero(right)):
        return True
    if operator == "-" and _is_zero(right):
        return True
    if operator == "*" and (_is_zero(left) or _is_zero(right) or _is_one(left) or _is_one(right)):
        return True
    if operator == "/" and _is_one(right):
        return True
    return False


