"""Relative costs of TAC arithmetic operators.

These weights are a simplified educational model. They are not measured CPU
cycle counts. They only record that, in this project, a multiply is treated
as more expensive than an add, and division and remainder are treated as more
expensive than a multiply.

Change the table here to change every label that depends on operation cost.
"""

from __future__ import annotations


# Binary operators accepted by the TAC parser.
OPERATION_COST = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 4,
    "%": 4,
}

# Unary minus is a single negation, given the same relative weight as subtraction.
UNARY_MINUS_COST = 1
