"""A small interpreter for the TAC subset this project optimizes.

It exists so a test can check that an optimizer did not change what a
program prints. It is not a production virtual machine.

Supported statements: assignment, binary arithmetic, unary minus, labels,
goto, conditional goto, and print. `/` and `%` truncate toward zero, matching
constant folding. A zero divisor, an undefined variable, or a step limit
stops execution with an error string. Two runs match when their prints and
their error strings match.

Variables begin at the supplied inputs. Integer literals in the program are
unsigned; a variable may hold a negative value.
"""

from __future__ import annotations

from dataclasses import dataclass

from optimizers.constant_folding import _div_toward_zero
from tac.instruction import InstructionType
from tac.program import TACProgram


@dataclass(frozen=True)
class ExecutionResult:
    """Prints, in order, and an error string when execution stopped early."""

    prints: tuple[int, ...]
    error: str | None = None


def execute(
    program: TACProgram,
    inputs: dict[str, int] | None = None,
    max_steps: int = 10000,
) -> ExecutionResult:
    """Run `program` and collect every printed integer."""
    environment = dict(inputs or {})
    labels = {
        instruction.target: index
        for index, instruction in enumerate(program.instructions)
        if instruction.instruction_type is InstructionType.LABEL and instruction.target
    }
    prints: list[int] = []
    index = 0
    steps = 0
    while index < len(program.instructions):
        if steps >= max_steps:
            return ExecutionResult(tuple(prints), "step limit")
        steps += 1
        instruction = program.instructions[index]
        kind = instruction.instruction_type
        try:
            if kind is InstructionType.LABEL:
                index += 1
            elif kind is InstructionType.GOTO:
                index = labels[instruction.target or ""]
            elif kind is InstructionType.IF_GOTO:
                if _compare(instruction.operator or "", _value(instruction.operand1, environment), _value(instruction.operand2, environment)):
                    index = labels[instruction.target or ""]
                else:
                    index += 1
            elif kind is InstructionType.PRINT:
                prints.append(_value(instruction.operand1, environment))
                index += 1
            elif kind is InstructionType.ASSIGN:
                environment[instruction.result or ""] = _value(instruction.operand1, environment)
                index += 1
            elif kind is InstructionType.UNARY:
                environment[instruction.result or ""] = _unary(instruction.operator or "", _value(instruction.operand1, environment))
                index += 1
            elif kind is InstructionType.BINARY:
                environment[instruction.result or ""] = _binary(
                    instruction.operator or "",
                    _value(instruction.operand1, environment),
                    _value(instruction.operand2, environment),
                )
                index += 1
            else:
                return ExecutionResult(tuple(prints), f"unsupported instruction {kind}")
        except KeyError as error:
            return ExecutionResult(tuple(prints), f"undefined {error}")
        except ZeroDivisionError:
            return ExecutionResult(tuple(prints), "division by zero")
        except ValueError as error:
            return ExecutionResult(tuple(prints), str(error))
    return ExecutionResult(tuple(prints))


def _value(token: str | None, environment: dict[str, int]) -> int:
    if token is None:
        raise ValueError("missing operand")
    if token.isdigit():
        return int(token)
    if token not in environment:
        raise KeyError(token)
    return environment[token]


def _unary(operator: str, value: int) -> int:
    if operator == "-":
        return -value
    raise ValueError(f"unsupported unary {operator}")


def _binary(operator: str, left: int, right: int) -> int:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if right == 0:
            raise ZeroDivisionError
        return _div_toward_zero(left, right)
    if operator == "%":
        if right == 0:
            raise ZeroDivisionError
        return left - _div_toward_zero(left, right) * right
    raise ValueError(f"unsupported operator {operator}")


def _compare(operator: str, left: int, right: int) -> bool:
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    raise ValueError(f"unsupported relation {operator}")
