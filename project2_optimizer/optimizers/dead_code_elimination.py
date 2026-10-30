"""Dead code elimination.

An assignment is removed when its result is not needed by a later live
assignment, a print, or a condition. Unused division or remainder whose
divisor may be zero is kept so a runtime trap is not skipped. The decision
is implemented in `tac.analysis.remove_dead_assignments`, which is also what
`num_dead_stmts` counts.

Prints, labels, and jumps are observable or required for control flow, so
they stay. Unused labels are not deleted. Removing one dead assignment can
make an earlier one dead; the helper repeats until none remain.
"""

from __future__ import annotations

from tac.analysis import remove_dead_assignments
from tac.program import TACProgram


def optimize(program: TACProgram) -> TACProgram:
    """Return a new program with dead assignments removed."""
    return TACProgram(remove_dead_assignments(program.instructions))
