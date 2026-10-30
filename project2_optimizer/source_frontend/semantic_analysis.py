"""Definite-assignment checks for the restricted source language."""

from __future__ import annotations

from source_frontend import ast_nodes as ast
from source_frontend.errors import SourceCompileError

TAC_RESERVED_IDENTIFIERS = frozenset({"if", "goto", "print"})


def collect_declared_names(program: ast.Program) -> set[str]:
    """Return every variable name introduced by an ``int`` declaration."""
    names: set[str] = set()

    def walk(statements: list[object]) -> None:
        for item in statements:
            if isinstance(item, ast.VariableDeclaration):
                names.add(item.name)
            elif isinstance(item, ast.IfStatement):
                walk(item.then_block)
                walk(item.else_block)
            elif isinstance(item, ast.WhileStatement):
                walk(item.body)
            elif isinstance(item, ast.BlockStatement):
                walk(item.body)

    walk(program.body)
    return names


def validate_definite_assignment(program: ast.Program) -> list[SourceCompileError]:
    """Reject reads of variables that may be uninitialized on some paths."""
    errors: list[SourceCompileError] = []
    _analyze_block(program.body, set(), errors)
    return errors


def _analyze_block(statements: list[object], initialized: set[str], errors: list[SourceCompileError]) -> set[str]:
    state = set(initialized)
    for item in statements:
        state = _analyze_statement(item, state, errors)
    return state


def _analyze_statement(item: object, initialized: set[str], errors: list[SourceCompileError]) -> set[str]:
    if isinstance(item, ast.VariableDeclaration):
        _check_expression(item.initializer, initialized, errors, item)
        return initialized | {item.name}
    if isinstance(item, ast.Assignment):
        _check_expression(item.value, initialized, errors, item)
        return initialized | {item.name}
    if isinstance(item, ast.PrintStatement):
        _check_expression(item.value, initialized, errors, item)
        return initialized
    if isinstance(item, ast.IfStatement):
        _check_comparison(item.condition, initialized, errors)
        then_state = _analyze_block(item.then_block, initialized, errors)
        if item.else_block:
            else_state = _analyze_block(item.else_block, initialized, errors)
            return then_state & else_state
        return initialized & then_state
    if isinstance(item, ast.WhileStatement):
        _check_comparison(item.condition, initialized, errors)
        _analyze_block(item.body, initialized, errors)
        return initialized
    if isinstance(item, ast.BlockStatement):
        return _analyze_block(item.body, initialized, errors)
    raise AssertionError(f"unknown statement: {type(item)}")


def _check_comparison(condition: ast.ComparisonExpression, initialized: set[str], errors: list[SourceCompileError]) -> None:
    _check_expression(condition.left, initialized, errors, condition)
    _check_expression(condition.right, initialized, errors, condition)


def _check_expression(
    expr: object,
    initialized: set[str],
    errors: list[SourceCompileError],
    site: object,
) -> None:
    if isinstance(expr, ast.IntegerLiteral):
        return
    if isinstance(expr, ast.VariableReference):
        if expr.name not in initialized:
            errors.append(
                SourceCompileError(
                    f"variable {expr.name!r} may be used before it is initialized",
                    expr.line,
                    expr.column,
                )
            )
        return
    if isinstance(expr, ast.UnaryExpression):
        _check_expression(expr.operand, initialized, errors, site)
        return
    if isinstance(expr, ast.BinaryExpression):
        _check_expression(expr.left, initialized, errors, site)
        _check_expression(expr.right, initialized, errors, site)
        return
    raise AssertionError(f"unknown expression: {type(expr)}")
