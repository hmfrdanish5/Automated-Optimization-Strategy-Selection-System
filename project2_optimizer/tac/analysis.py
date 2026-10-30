"""Control-flow facts shared by feature extraction and the optimizers.

Dead-code elimination and `num_dead_stmts` both call `remove_dead_assignments`,
so the feature count is the number of instructions that pass actually deletes.
Loop detection uses the same back-edge rule as `num_loop_ops`.
"""

from __future__ import annotations

from tac.instruction import InstructionType, TACInstruction, is_int_literal


def label_positions(instructions: list[TACInstruction]) -> dict[str, int]:
    """Map each label name to the index of its label instruction."""
    positions: dict[str, int] = {}
    for index, instruction in enumerate(instructions):
        if instruction.instruction_type is InstructionType.LABEL and instruction.target:
            positions[instruction.target] = index
    return positions


def successors(instructions: list[TACInstruction]) -> list[list[int]]:
    """Control-flow successors of each instruction.

    Ordinary statements fall through. `goto` has only its target. `if` has
    both its target and the fall-through instruction.
    """
    labels = label_positions(instructions)
    edges: list[list[int]] = []
    last = len(instructions) - 1
    for index, instruction in enumerate(instructions):
        if instruction.instruction_type is InstructionType.GOTO:
            edges.append([labels[instruction.target or ""]])
        elif instruction.instruction_type is InstructionType.IF_GOTO:
            following = [labels[instruction.target or ""]]
            if index < last:
                following.append(index + 1)
            edges.append(following)
        elif index < last:
            edges.append([index + 1])
        else:
            edges.append([])
    return edges


def predecessors(edges: list[list[int]]) -> list[list[int]]:
    """Invert a successor graph."""
    incoming: list[list[int]] = [[] for _ in edges]
    for index, followers in enumerate(edges):
        for follower in followers:
            incoming[follower].append(index)
    return incoming


def loop_body_indexes(instructions: list[TACInstruction]) -> set[int]:
    """Indexes of instructions that sit inside at least one loop.

    A loop is a jump whose target label is at or before the jump. The body
    includes the label and the jump. An instruction in two loops is recorded
    once. A forward jump is not a loop.
    """
    labels = label_positions(instructions)
    in_loop: set[int] = set()
    for index, instruction in enumerate(instructions):
        if instruction.instruction_type not in {InstructionType.GOTO, InstructionType.IF_GOTO}:
            continue
        target = labels.get(instruction.target or "")
        if target is None or target > index:
            continue
        in_loop.update(range(target, index + 1))
    return in_loop


def live_out(instructions: list[TACInstruction]) -> list[set[str]]:
    """Names that may be used on some path after each instruction."""
    edges = successors(instructions)
    uses = [instruction.used_names() for instruction in instructions]
    definitions = [instruction.defined_name() for instruction in instructions]
    live_in = [set() for _ in instructions]
    live_after = [set() for _ in instructions]

    changed = True
    while changed:
        changed = False
        for index in range(len(instructions) - 1, -1, -1):
            new_out: set[str] = set()
            for successor in edges[index]:
                new_out |= live_in[successor]
            defined = definitions[index]
            if defined is None:
                new_in = uses[index] | new_out
            else:
                new_in = uses[index] | (new_out - {defined})
            if new_out != live_after[index] or new_in != live_in[index]:
                live_after[index] = new_out
                live_in[index] = new_in
                changed = True
    return live_after


def _unused_divmod_may_trap(instruction: TACInstruction) -> bool:
    """True when a dead `/` or `%` assignment can still stop the program.

    Removing such a statement can skip a division-by-zero error and change
    which prints run before termination. Literal divisors are only treated as
    safe when they are non-zero integers; a variable divisor may be zero at
    runtime.
    """
    if instruction.instruction_type is not InstructionType.BINARY:
        return False
    if instruction.operator not in {"/", "%"}:
        return False
    divisor = instruction.operand2
    if divisor is None:
        return False
    if is_int_literal(divisor):
        return int(divisor) == 0
    return True


def remove_dead_assignments(instructions: list[TACInstruction]) -> list[TACInstruction]:
    """Drop assignments whose results are unused, and repeat until stable.

    Prints, labels, and jumps stay. A use inside an assignment that this
    function removes does not keep an earlier assignment alive. Unused `/` or
    `%` assignments whose divisor may be zero are kept so observable traps are
    not skipped. The input list is not modified.
    """
    remaining = list(instructions)
    changed = True
    while changed:
        changed = False
        live_after = live_out(remaining)
        kept: list[TACInstruction] = []
        for instruction, live in zip(remaining, live_after):
            defined = instruction.defined_name()
            if defined is not None and defined not in live:
                if _unused_divmod_may_trap(instruction):
                    kept.append(instruction)
                else:
                    changed = True
            else:
                kept.append(instruction)
        remaining = kept
    return remaining


def forward_dataflow(instructions: list[TACInstruction], transfer, meet):
    """State before each instruction, for a forward "must" analysis.

    `transfer(in_state, instruction)` returns the state after that instruction.
    `meet(left, right)` combines two predecessor states and returns a new dict.

    The scan repeats so that loops and branches can settle. A predecessor that
    has not been analyzed yet is left out of the meet: a back edge must not
    look like a finished path before the analyzer has walked it. The program
    entry contributes an empty state to the first instruction.
    """
    count = len(instructions)
    in_states: list[dict] = [{} for _ in instructions]
    if count == 0:
        return in_states

    edges = successors(instructions)
    incoming = predecessors(edges)
    out_states: list[dict | None] = [None] * count
    visited = [False] * count

    def state_before(index: int) -> dict:
        states: list[dict] = []
        if index == 0:
            states.append({})
        for predecessor in incoming[index]:
            if visited[predecessor]:
                states.append(out_states[predecessor] or {})
        if not states:
            return {}
        combined = states[0]
        for state in states[1:]:
            combined = meet(combined, state)
        return combined

    changed = True
    sweeps = 0
    while changed:
        sweeps += 1
        if sweeps > count * count + 2:
            raise RuntimeError("dataflow analysis did not converge")
        changed = False
        for index, instruction in enumerate(instructions):
            new_in = state_before(index)
            new_out = transfer(new_in, instruction)
            if new_in != in_states[index] or new_out != out_states[index]:
                in_states[index] = new_in
                out_states[index] = new_out
                visited[index] = True
                changed = True
    return in_states
