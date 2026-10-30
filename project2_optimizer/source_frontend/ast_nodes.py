"""AST nodes for the restricted source language."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Program:
    body: list[object] = field(default_factory=list)


@dataclass
class VariableDeclaration:
    name: str
    initializer: object
    line: int
    column: int


@dataclass
class Assignment:
    name: str
    value: object
    line: int
    column: int


@dataclass
class PrintStatement:
    value: object
    line: int
    column: int


@dataclass
class IfStatement:
    condition: object
    then_block: list[object]
    else_block: list[object]
    line: int
    column: int


@dataclass
class WhileStatement:
    condition: object
    body: list[object]
    line: int
    column: int


@dataclass
class BlockStatement:
    body: list[object]
    line: int
    column: int


@dataclass
class BinaryExpression:
    left: object
    operator: str
    right: object
    line: int
    column: int


@dataclass
class UnaryExpression:
    operator: str
    operand: object
    line: int
    column: int


@dataclass
class IntegerLiteral:
    value: int
    line: int
    column: int


@dataclass
class VariableReference:
    name: str
    line: int
    column: int


@dataclass
class ComparisonExpression:
    left: object
    operator: str
    right: object
    line: int
    column: int
