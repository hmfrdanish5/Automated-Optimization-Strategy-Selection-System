"""Recursive-descent parser with Pratt expression parsing."""

from __future__ import annotations

from source_frontend import ast_nodes as ast
from source_frontend.errors import SourceCompileError
from source_frontend.semantic_analysis import TAC_RESERVED_IDENTIFIERS
from source_frontend.tokens import Token, TokenKind

_UNSUPPORTED_KEYWORDS = {
    "for": "for loops are not supported",
    "break": "break is not supported",
    "continue": "continue is not supported",
    "return": "return is not supported",
    "void": "functions are not supported",
    "float": "floating-point types are not supported",
    "double": "floating-point types are not supported",
    "char": "strings and char types are not supported",
    "struct": "structs are not supported",
    "class": "classes are not supported",
}


class Parser:
    def __init__(self, tokens: list[Token]):
        self._tokens = tokens
        self._index = 0
        self._declared: set[str] = set()

    def parse_program(self) -> ast.Program:
        body: list[object] = []
        while not self._check(TokenKind.EOF):
            if self._check(TokenKind.INT):
                body.append(self._parse_declaration())
            else:
                body.append(self._parse_statement())
        return ast.Program(body=body)

    def _parse_declaration(self) -> ast.VariableDeclaration:
        start = self._advance(TokenKind.INT)
        name_token = self._advance(TokenKind.IDENT)
        self._reject_identifier_name(name_token)
        name = name_token.text
        if self._check(TokenKind.LPAREN):
            raise self._error(
                "functions are not supported (declare only simple int variables)",
                name_token,
            )
        if name in self._declared:
            raise self._error(f"duplicate declaration of {name!r}", name_token)
        self._advance(TokenKind.ASSIGN)
        value = self._parse_expression()
        self._advance(TokenKind.SEMI)
        self._declared.add(name)
        return ast.VariableDeclaration(
            name=name,
            initializer=value,
            line=start.line,
            column=start.column,
        )

    def _parse_statement(self) -> object:
        if self._check(TokenKind.PRINT):
            return self._parse_print()
        if self._check(TokenKind.IF):
            return self._parse_if()
        if self._check(TokenKind.WHILE):
            return self._parse_while()
        if self._check(TokenKind.LBRACE):
            return self._parse_block()
        if self._check(TokenKind.IDENT):
            return self._parse_assignment()
        token = self._peek()
        if token.kind is TokenKind.INT:
            raise self._error("variable declarations must appear at block scope with int", token)
        if token.text in _UNSUPPORTED_KEYWORDS:
            raise self._error(_UNSUPPORTED_KEYWORDS[token.text], token)
        raise self._error(f"unexpected token {token.text!r}", token)

    def _parse_block(self) -> ast.BlockStatement:
        start = self._advance(TokenKind.LBRACE)
        body: list[object] = []
        while not self._check(TokenKind.RBRACE):
            if self._check(TokenKind.EOF):
                raise self._error("unclosed block", start)
            if self._check(TokenKind.INT):
                body.append(self._parse_declaration())
            else:
                body.append(self._parse_statement())
        self._advance(TokenKind.RBRACE)
        return ast.BlockStatement(body=body, line=start.line, column=start.column)

    def _parse_print(self) -> ast.PrintStatement:
        start = self._advance(TokenKind.PRINT)
        self._advance(TokenKind.LPAREN)
        value = self._parse_expression()
        self._advance(TokenKind.RPAREN)
        self._advance(TokenKind.SEMI)
        return ast.PrintStatement(value=value, line=start.line, column=start.column)

    def _parse_if(self) -> ast.IfStatement:
        start = self._advance(TokenKind.IF)
        self._advance(TokenKind.LPAREN)
        condition = self._parse_comparison()
        self._advance(TokenKind.RPAREN)
        then_block = self._parse_block_body()
        if self._match(TokenKind.ELSE):
            else_block = self._parse_block_body()
        else:
            else_block = []
        return ast.IfStatement(
            condition=condition,
            then_block=then_block,
            else_block=else_block,
            line=start.line,
            column=start.column,
        )

    def _parse_while(self) -> ast.WhileStatement:
        start = self._advance(TokenKind.WHILE)
        self._advance(TokenKind.LPAREN)
        condition = self._parse_comparison()
        self._advance(TokenKind.RPAREN)
        body = self._parse_block_body()
        return ast.WhileStatement(
            condition=condition,
            body=body,
            line=start.line,
            column=start.column,
        )

    def _parse_block_body(self) -> list[object]:
        if self._check(TokenKind.LBRACE):
            return self._parse_block().body
        return [self._parse_statement()]

    def _parse_assignment(self) -> ast.Assignment:
        name_token = self._advance(TokenKind.IDENT)
        self._reject_identifier_name(name_token)
        if name_token.text not in self._declared:
            raise self._error(f"undeclared variable {name_token.text!r}", name_token)
        if self._check(TokenKind.LPAREN):
            raise self._error("function calls are not supported", self._peek())
        if self._check(TokenKind.LBRACE):
            raise self._error("expected assignment, found block", self._peek())
        self._advance(TokenKind.ASSIGN)
        value = self._parse_expression()
        self._advance(TokenKind.SEMI)
        return ast.Assignment(
            name=name_token.text,
            value=value,
            line=name_token.line,
            column=name_token.column,
        )

    def _parse_comparison(self) -> ast.ComparisonExpression:
        left = self._parse_expression()
        op_token = self._peek()
        if op_token.kind not in {
            TokenKind.LT,
            TokenKind.GT,
            TokenKind.LE,
            TokenKind.GE,
            TokenKind.EQEQ,
            TokenKind.NE,
        }:
            raise self._error("expected comparison operator", op_token)
        self._advance(op_token.kind)
        right = self._parse_expression()
        return ast.ComparisonExpression(
            left=left,
            operator=op_token.text,
            right=right,
            line=op_token.line,
            column=op_token.column,
        )

    def _parse_expression(self) -> object:
        return self._parse_binary(0)

    _PRECEDENCE = {
        TokenKind.PLUS: 1,
        TokenKind.MINUS: 1,
        TokenKind.STAR: 2,
        TokenKind.SLASH: 2,
        TokenKind.PERCENT: 2,
    }

    def _parse_binary(self, min_prec: int) -> object:
        left = self._parse_unary()
        while True:
            token = self._peek()
            prec = self._PRECEDENCE.get(token.kind)
            if prec is None or prec < min_prec:
                break
            self._advance(token.kind)
            right = self._parse_binary(prec + 1)
            left = ast.BinaryExpression(
                left=left,
                operator=token.text,
                right=right,
                line=token.line,
                column=token.column,
            )
        return left

    def _parse_unary(self) -> object:
        if self._match(TokenKind.MINUS):
            token = self._previous()
            operand = self._parse_unary()
            return ast.UnaryExpression(
                operator="-",
                operand=operand,
                line=token.line,
                column=token.column,
            )
        return self._parse_primary()

    def _parse_primary(self) -> object:
        token = self._peek()
        if self._match(TokenKind.NUMBER):
            return ast.IntegerLiteral(
                value=int(token.text),
                line=token.line,
                column=token.column,
            )
        if self._match(TokenKind.IDENT):
            self._reject_identifier_name(token)
            if token.text not in self._declared:
                raise self._error(f"undeclared variable {token.text!r}", token)
            return ast.VariableReference(name=token.text, line=token.line, column=token.column)
        if self._match(TokenKind.LPAREN):
            expr = self._parse_expression()
            self._advance(TokenKind.RPAREN)
            return expr
        raise self._error(f"expected expression, found {token.text!r}", token)

    def _reject_identifier_name(self, token: Token) -> None:
        self._reject_unsupported_identifier(token.text)
        if token.text in TAC_RESERVED_IDENTIFIERS:
            raise self._error(
                f"identifier {token.text!r} is reserved by the TAC backend",
                token,
            )

    def _reject_unsupported_identifier(self, name: str) -> None:
        if name in _UNSUPPORTED_KEYWORDS:
            token = self._previous() if self._previous().text == name else self._peek()
            raise self._error(_UNSUPPORTED_KEYWORDS[name], token)

    def _match(self, kind: TokenKind) -> bool:
        if self._check(kind):
            self._index += 1
            return True
        return False

    def _check(self, kind: TokenKind) -> bool:
        return self._peek().kind is kind

    def _advance(self, kind: TokenKind) -> Token:
        token = self._peek()
        if token.kind is not kind:
            raise self._error(f"expected {kind.name}, found {token.text!r}", token)
        self._index += 1
        return token

    def _peek(self) -> Token:
        return self._tokens[self._index]

    def _previous(self) -> Token:
        return self._tokens[self._index - 1]

    def _error(self, message: str, token: Token) -> SourceCompileError:
        return SourceCompileError(message, token.line, token.column)


def parse(tokens: list[Token]) -> ast.Program:
    return Parser(tokens).parse_program()
