"""Generate textual TAC from the source AST."""

from __future__ import annotations

from source_frontend import ast_nodes as ast
from source_frontend.semantic_analysis import collect_declared_names


class TacGenerator:
    def __init__(self, user_identifiers: set[str] | None = None) -> None:
        self._lines: list[str] = []
        self._label = 0
        self._temp_counter = 0
        self._reserved = set(user_identifiers or ())
        self._used_names = set(self._reserved)

    def generate(self, program: ast.Program) -> str:
        for item in program.body:
            self._emit_item(item)
        return "\n".join(self._lines)

    def _emit_item(self, item: object) -> None:
        if isinstance(item, ast.VariableDeclaration):
            dest, lines = self._gen_expr(item.initializer)
            self._extend(lines)
            self._lines.append(f"{item.name} = {dest}")
            return
        if isinstance(item, ast.Assignment):
            dest, lines = self._gen_expr(item.value)
            self._extend(lines)
            self._lines.append(f"{item.name} = {dest}")
            return
        if isinstance(item, ast.PrintStatement):
            dest, lines = self._gen_expr(item.value)
            self._extend(lines)
            self._lines.append(f"print {dest}")
            return
        if isinstance(item, ast.IfStatement):
            self._gen_if(item)
            return
        if isinstance(item, ast.WhileStatement):
            self._gen_while(item)
            return
        if isinstance(item, ast.BlockStatement):
            for child in item.body:
                self._emit_item(child)
            return
        raise AssertionError(f"unknown AST node: {type(item)}")

    def _gen_if(self, stmt: ast.IfStatement) -> None:
        then_label = self._fresh_label("then")
        end_label = self._fresh_label("end")
        if stmt.else_block:
            self._gen_cond_jump(stmt.condition, then_label)
            for item in stmt.else_block:
                self._emit_item(item)
            self._lines.append(f"goto {end_label}")
            self._lines.append(f"{then_label}:")
            for item in stmt.then_block:
                self._emit_item(item)
            self._lines.append(f"{end_label}:")
        else:
            self._gen_cond_jump(stmt.condition, then_label)
            self._lines.append(f"goto {end_label}")
            self._lines.append(f"{then_label}:")
            for item in stmt.then_block:
                self._emit_item(item)
            self._lines.append(f"{end_label}:")

    def _gen_while(self, stmt: ast.WhileStatement) -> None:
        loop_label = self._fresh_label("loop")
        body_label = self._fresh_label("body")
        end_label = self._fresh_label("end")
        self._lines.append(f"{loop_label}:")
        self._gen_cond_jump(stmt.condition, body_label)
        self._lines.append(f"goto {end_label}")
        self._lines.append(f"{body_label}:")
        for item in stmt.body:
            self._emit_item(item)
        self._lines.append(f"goto {loop_label}")
        self._lines.append(f"{end_label}:")

    def _gen_cond_jump(self, condition: ast.ComparisonExpression, target: str) -> None:
        left = self._operand(condition.left)
        right = self._operand(condition.right)
        self._extend(left[1])
        self._extend(right[1])
        self._lines.append(f"if {left[0]} {condition.operator} {right[0]} goto {target}")

    def _gen_expr(self, expr: object) -> tuple[str, list[str]]:
        if isinstance(expr, ast.IntegerLiteral):
            return str(expr.value), []
        if isinstance(expr, ast.VariableReference):
            return expr.name, []
        if isinstance(expr, ast.UnaryExpression):
            operand = self._operand(expr.operand)
            temp = self._fresh_temp()
            lines = operand[1] + [f"{temp} = - {operand[0]}"]
            return temp, lines
        if isinstance(expr, ast.BinaryExpression):
            left = self._operand(expr.left)
            right = self._operand(expr.right)
            temp = self._fresh_temp()
            lines = left[1] + right[1] + [f"{temp} = {left[0]} {expr.operator} {right[0]}"]
            return temp, lines
        raise AssertionError(f"unexpected expression node: {type(expr)}")

    def _operand(self, expr: object) -> tuple[str, list[str]]:
        if isinstance(expr, ast.IntegerLiteral):
            return str(expr.value), []
        if isinstance(expr, ast.VariableReference):
            return expr.name, []
        return self._gen_expr(expr)

    def _fresh_temp(self) -> str:
        while True:
            self._temp_counter += 1
            candidate = f"__src_tmp_{self._temp_counter}"
            if candidate not in self._used_names:
                self._used_names.add(candidate)
                return candidate

    def _fresh_label(self, hint: str) -> str:
        while True:
            self._label += 1
            candidate = f"__src_lbl_{hint}_{self._label}"
            if candidate not in self._used_names:
                self._used_names.add(candidate)
                return candidate

    def _extend(self, lines: list[str]) -> None:
        self._lines.extend(lines)


def generate_tac(program: ast.Program) -> str:
    reserved = collect_declared_names(program)
    return TacGenerator(reserved).generate(program)
