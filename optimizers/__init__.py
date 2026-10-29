"""The six TAC optimization passes.

Call any `optimize` function on its own. They do not chain.
"""

from optimizers.registry import OPTIMIZERS, STRATEGY_NAMES, apply_strategy

__all__ = ["OPTIMIZERS", "STRATEGY_NAMES", "apply_strategy"]
