"""Controlled random TAC programs.

Each category plants a kind of opportunity, then the labeler measures all
six optimizers. The category name is never copied into `best_strategy`.
A program from the CSE category can still be labeled dead-code elimination
when deleting instructions removes more than reusing an expression.

The seed fixes the sequence. The same seed and count reproduce the same
programs.
"""

from __future__ import annotations

import random

from dataset.diversity import feature_signature
from features.extractor import extract_features
from tac.parser import parse_tac
from tac.program import TACProgram


GENERATION_SEED = 42
GENERATED_PROGRAM_COUNT = 48

CATEGORIES = (
    "constant_heavy",
    "propagation_heavy",
    "algebraic_heavy",
    "cse_heavy",
    "dead_code_heavy",
    "loop_heavy",
    "strength_reduction_heavy",
    "mixed",
)


def generate_programs(
    count: int = GENERATED_PROGRAM_COUNT,
    seed: int = GENERATION_SEED,
) -> list[tuple[str, TACProgram, str]]:
    """Return `(program_id, program, category)` for `count` programs.

    `category` is bookkeeping for inspection. It is not a dataset column.
    """
    rng = random.Random(seed)
    order = [CATEGORIES[index % len(CATEGORIES)] for index in range(count)]
    rng.shuffle(order)
    seen: set[str] = set()
    programs: list[tuple[str, TACProgram, str]] = []
    for index, category in enumerate(order, start=1):
        program = _unique_program(rng, category, seen, index)
        programs.append((f"gen_{index:03d}", program, category))
    return programs


def _unique_program(
    rng: random.Random, category: str, seen: set[str], index: int
) -> TACProgram:
    program = _GENERATORS[category](rng)
    text = program.to_tac()
    attempts = 0
    while text in seen and attempts < 8:
        program = _GENERATORS[category](rng)
        text = program.to_tac()
        attempts += 1
    if text in seen:
        program = parse_tac(text + f"\nu{index} = q{index} + r{index}\nprint u{index}")
        text = program.to_tac()
    seen.add(text)
    return program


def _constant_heavy(rng: random.Random) -> TACProgram:
    """Several literal arithmetic instructions, all printed so they stay live."""
    lines: list[str] = []
    temps: list[str] = []
    for _ in range(rng.randint(2, 5)):
        destination = f"t{len(temps) + 1}"
        operator = rng.choice(["+", "-", "*", "/", "%"])
        if operator in {"/", "%"}:
            right = rng.randint(1, 9)
            left = right * rng.randint(0, 5) + rng.randint(0, right - 1)
        elif operator == "-":
            right = rng.randint(1, 12)
            left = rng.randint(0, 25)
        else:
            # 0 and 1 would also match an algebraic identity.
            left = rng.randint(2, 40)
            right = rng.randint(2, 15)
        lines.append(f"{destination} = {left} {operator} {right}")
        temps.append(destination)
    if len(temps) >= 2 and rng.random() < 0.4:
        destination = f"t{len(temps) + 1}"
        lines.append(f"{destination} = {temps[0]} + {temps[1]}")
        temps.append(destination)
    lines.extend(f"print {temp}" for temp in temps)
    return parse_tac("\n".join(lines))


def _propagation_heavy(rng: random.Random) -> TACProgram:
    """A long name holds a constant and is used again later.

    The later uses are not themselves foldable until the name is replaced,
    and the other operand is neither 0 nor 1, so algebraic simplification
    does not apply to the original program.
    """
    source = rng.choice(["value", "limit", "width", "count", "total", "offset", "length"])
    literal = rng.randint(2, 40)
    lines = [f"{source} = {literal}"]
    for index in range(rng.randint(1, 4)):
        destination = f"t{index + 1}"
        operator = rng.choice(["+", "-", "*"])
        other = rng.randint(2, 9)
        if operator != "-" and rng.random() < 0.5:
            lines.append(f"{destination} = {other} {operator} {source}")
        else:
            lines.append(f"{destination} = {source} {operator} {other}")
        lines.append(f"print {destination}")
    lines.append(f"print {source}")
    return parse_tac("\n".join(lines))


def _algebraic_heavy(rng: random.Random) -> TACProgram:
    """Identities from the algebraic pass, each on its own variable."""
    forms = (
        "{name} + 0",
        "0 + {name}",
        "{name} - 0",
        "{name} * 1",
        "1 * {name}",
        "{name} / 1",
        "{name} * 0",
        "0 * {name}",
    )
    lines: list[str] = []
    for index in range(rng.randint(2, 5)):
        destination = f"t{index + 1}"
        expression = rng.choice(forms).format(name=f"v{index}")
        lines.append(f"{destination} = {expression}")
        lines.append(f"print {destination}")
    return parse_tac("\n".join(lines))


def _cse_heavy(rng: random.Random) -> TACProgram:
    """The same binary expression is computed more than once, then printed."""
    pairs = [("a", "b"), ("left", "right"), ("base", "width"), ("index", "limit")]
    chosen = rng.sample(pairs, rng.randint(1, 2))
    lines: list[str] = []
    next_temp = 1
    produced: list[str] = []
    for left, right in chosen:
        operator = rng.choice(["+", "-", "*"])
        for _ in range(rng.randint(2, 4)):
            destination = f"t{next_temp}"
            next_temp += 1
            lines.append(f"{destination} = {left} {operator} {right}")
            produced.append(destination)
        destination = f"t{next_temp}"
        next_temp += 1
        lines.append(f"{destination} = c + d")
        produced.append(destination)
    lines.extend(f"print {name}" for name in produced)
    return parse_tac("\n".join(lines))


def _dead_code_heavy(rng: random.Random) -> TACProgram:
    """One printed computation plus assignments that nothing reads."""
    lines = ["live = left + right"]
    previous: str | None = None
    for index in range(rng.randint(2, 6)):
        name = f"spare{index}"
        if previous is None or rng.random() < 0.5:
            lines.append(f"{name} = {rng.randint(1, 30)}")
        else:
            lines.append(f"{name} = {previous} + {rng.randint(2, 9)}")
        previous = name
    lines.append("print live")
    return parse_tac("\n".join(lines))


def _loop_heavy(rng: random.Random) -> TACProgram:
    """A counted loop containing one multiply by two.

    Dead code, when present, usually wins the label because deleting an
    instruction outranks a cheaper operator. Otherwise strength reduction
    is the measured winner: the multiply becomes an add, and that lowers
    weighted cost even when the instruction count stays the same.
    """
    bound = rng.choice(["n", "n", "limit", "length", "bound"])
    literal = rng.randint(2, 20)
    lines = [
        "i = 0",
        f"{bound} = {literal}",
        "L1:",
        "t1 = i * 2",
        "i = i + 1",
        f"if i < {bound} goto L1",
        "print t1",
        "print i",
    ]
    if rng.random() < 0.3:
        lines.insert(0, f"spare = {rng.randint(1, 9)}")
    return parse_tac("\n".join(lines))


def _strength_reduction_heavy(rng: random.Random) -> TACProgram:
    """Loop multiplies by two, with no other optimization opportunity.

    Each induction variable is different, so the expressions are not copies
    of each other and CSE does not remove a multiply. The loop bound is a
    literal, so constant propagation has no name to replace. Every computed
    temporary is printed, so dead-code elimination has nothing to delete.
    The label is still whatever the six measured gains select.
    """
    names = ("i", "j", "k")
    chosen = list(names[: rng.randint(1, len(names))])
    lines = [f"{name} = {start}" for start, name in enumerate(chosen)]
    lines.append("L1:")
    for index, name in enumerate(chosen):
        destination = f"t{index + 1}"
        if rng.random() < 0.5:
            lines.append(f"{destination} = {name} * 2")
        else:
            lines.append(f"{destination} = 2 * {name}")
    for name in chosen:
        lines.append(f"{name} = {name} + 1")
    lines.append(f"if i < {rng.randint(3, 12)} goto L1")
    lines.extend(f"print t{index + 1}" for index in range(len(chosen)))
    lines.append("print i")
    return parse_tac("\n".join(lines))


def _mixed(rng: random.Random) -> TACProgram:
    """A few different opportunities in one program.

    Which one wins is measured. A spare assignment usually lets dead-code
    elimination win, because deleting an instruction outranks a cheaper
    operator. A loop multiply by two is included often, but a constant
    fold or a removed repeated add can still score a larger improvement.
    """
    lines = ["anchor = left + right"]
    next_temp = 1
    if rng.random() < 0.75:
        destination = f"t{next_temp}"
        next_temp += 1
        lines.append(f"{destination} = {rng.randint(2, 30)} + {rng.randint(2, 12)}")
        lines.append(f"print {destination}")
    if rng.random() < 0.75:
        destination = f"t{next_temp}"
        next_temp += 1
        lines.append(f"{destination} = base + 0")
        lines.append(f"print {destination}")
    if rng.random() < 0.75:
        first = f"t{next_temp}"
        next_temp += 1
        second = f"t{next_temp}"
        next_temp += 1
        lines.append(f"{first} = width + index")
        lines.append(f"{second} = width + index")
        lines.append(f"print {first}")
        lines.append(f"print {second}")
    if rng.random() < 0.55:
        lines.extend(
            [
                "i = 0",
                "L1:",
                f"t{next_temp} = i * 2",
                "i = i + 1",
                f"if i < {rng.randint(3, 9)} goto L1",
                f"print t{next_temp}",
                "print i",
            ]
        )
        next_temp += 1
    if rng.random() < 0.4:
        lines.append(f"spare = {rng.randint(1, 9)}")
    lines.append("print anchor")
    return parse_tac("\n".join(lines))


_GENERATORS = {
    "constant_heavy": _constant_heavy,
    "propagation_heavy": _propagation_heavy,
    "algebraic_heavy": _algebraic_heavy,
    "cse_heavy": _cse_heavy,
    "dead_code_heavy": _dead_code_heavy,
    "loop_heavy": _loop_heavy,
    "strength_reduction_heavy": _strength_reduction_heavy,
    "mixed": _mixed,
}


# Keep deterministic base and expanded program generators available for tests and reuse.
EXPANDED_GENERATED_COUNT = 102
FINAL_GENERATED_COUNT = 191
FINAL_CATEGORY_MINIMUMS = {
    "constant_heavy": 20,
    "propagation_heavy": 20,
    "algebraic_heavy": 20,
    "cse_heavy": 22,
    "dead_code_heavy": 22,
    "strength_reduction_heavy": 22,
    "loop_heavy": 14,
    "straight_line": 16,
    "mixed": 35,
}


def generate_diverse_programs(
    count: int = EXPANDED_GENERATED_COUNT,
    seed: int = GENERATION_SEED,
) -> list[tuple[str, TACProgram, str]]:
    """Return generated programs whose feature counts cover a wider range.

    Each recipe changes a count the eight features can see: how many
    statements, foldable operations, repeated expressions, dead assignments,
    algebraic identities, loop-body instructions, or temporaries. The label
    is not chosen here. Two recipes that still extract the same eight numbers
    are kept; a few `print 0` lines are tried first so a repeated signature
    can become a different one.
    """
    pool = _diverse_pool()
    rng = random.Random(seed)
    chosen = _cover_categories(pool, count, rng)
    seen: set[tuple] = set()
    programs: list[tuple[str, TACProgram, str]] = []
    for index, (category, program) in enumerate(chosen, start=1):
        program = _separate_signature(program, seen)
        programs.append((f"exp_{index:03d}", program, category))
    return programs


def _cover_categories(
    pool: list[tuple[str, TACProgram]],
    count: int,
    rng: random.Random,
    minimums: dict[str, int] | None = None,
) -> list[tuple[str, TACProgram]]:
    """Keep several programs from each family, then fill from the remainder.

    The family name is the opportunity the recipe was built to contain.
    It is not the measured label.
    """
    quotas = minimums or {
        "constant_heavy": 10,
        "propagation_heavy": 12,
        "algebraic_heavy": 10,
        "cse_heavy": 12,
        "dead_code_heavy": 12,
        "strength_reduction_heavy": 12,
        "loop_heavy": 6,
        "straight_line": 10,
        "mixed": 12,
    }
    by_category: dict[str, list[tuple[str, TACProgram]]] = {}
    for item in pool:
        by_category.setdefault(item[0], []).append(item)
    for items in by_category.values():
        rng.shuffle(items)
    chosen: list[tuple[str, TACProgram]] = []
    for category, items in by_category.items():
        take = min(quotas.get(category, 0), len(items), max(count - len(chosen), 0))
        chosen.extend(items[:take])
        del items[:take]
    remainder = [item for items in by_category.values() for item in items]
    rng.shuffle(remainder)
    chosen.extend(remainder[: max(count - len(chosen), 0)])
    rng.shuffle(chosen)
    return chosen[:count]


def random_demo_tac(seed: int | None = None) -> tuple[str, str]:
    """Return one synthetic TAC string for UI demos.

    The program is drawn from the same recipe pool as ``dataset_final.csv``.
    It is not written to any CSV and is not labeled here.
    """
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    category, program = rng.choice(_final_pool())
    return category, program.to_tac()


def generate_final_programs(
    count: int = FINAL_GENERATED_COUNT,
    seed: int = GENERATION_SEED,
) -> list[tuple[str, TACProgram, str]]:
    """Programs for dataset_final.csv.

    The category is only which opportunity the recipe was built around.
    `best_strategy` is measured later and is not assigned here.
    """
    rng = random.Random(seed)
    chosen = _cover_categories(_final_pool(), count, rng, FINAL_CATEGORY_MINIMUMS)
    seen: set[tuple] = set()
    programs: list[tuple[str, TACProgram, str]] = []
    for index, (category, program) in enumerate(chosen, start=1):
        program = _separate_signature(program, seen)
        programs.append((f"fin_{index:03d}", program, category))
    return programs


def generate_extension_programs(
    count: int = 200,
    seed: int = 43,
) -> list[tuple[str, TACProgram, str]]:
    """Additional generated TAC programs for the frozen optimization dataset."""
    rng = random.Random(seed)
    chosen = _cover_categories(_final_pool(), count, rng, FINAL_CATEGORY_MINIMUMS)
    seen: set[tuple] = set()
    programs: list[tuple[str, TACProgram, str]] = []
    for index, (category, program) in enumerate(chosen, start=1):
        program = _separate_signature(program, seen)
        programs.append((f"ext_{index:03d}", program, category))
    return programs


def _final_pool() -> list[tuple[str, TACProgram]]:
    """A wider set of TAC shapes than the 111-row development set."""
    families: dict[str, list[TACProgram]] = {
        "constant_heavy": [
            _fold_program(constants, padding)
            for constants in range(1, 13)
            for padding in range(0, 4)
        ],
        "algebraic_heavy": [
            _algebraic_program(identities, padding)
            for identities in range(1, 9)
            for padding in range(0, 4)
        ],
        "dead_code_heavy": [
            _dead_program(fold_dead, plain_dead, live_temps)
            for fold_dead in range(0, 6)
            for plain_dead in range(0, 6)
            for live_temps in range(0, 3)
            if fold_dead + plain_dead > 0
        ],
        "cse_heavy": [
            _cse_program(copies, extras, padding)
            for copies in range(2, 10)
            for extras in range(0, 4)
            for padding in range(0, 3)
        ],
        "propagation_heavy": [
            _propagation_program(uses, padding)
            for uses in range(1, 11)
            for padding in range(0, 4)
        ],
        "strength_reduction_heavy": [
            _reduction_program(reductions, fillers, bound, dead)
            for reductions in range(1, 5)
            for fillers in range(0, 7)
            for bound in (4, 9, 15)
            for dead in range(0, 3)
        ],
        "loop_heavy": [
            _plain_loop(body, padding)
            for body in range(1, 11)
            for padding in range(0, 3)
        ],
        "straight_line": [
            _straight_program(temporaries, padding)
            for temporaries in range(1, 9)
            for padding in range(0, 4)
        ],
        "mixed": [
            _mixed_program(constants, copies, identities, reductions, dead)
            for constants in range(0, 4)
            for copies in (0, 2, 3, 5)
            for identities in range(0, 3)
            for reductions in range(0, 3)
            for dead in range(0, 3)
            if constants + copies + identities + reductions + dead >= 2
        ],
    }
    interleaved: list[tuple[str, TACProgram]] = []
    while any(families.values()):
        for category, programs in families.items():
            if programs:
                interleaved.append((category, programs.pop(0)))
    return interleaved


def _diverse_pool() -> list[tuple[str, TACProgram]]:
    """Interleave families so a prefix still contains every opportunity type."""
    families: dict[str, list[TACProgram]] = {
        "constant_heavy": [
            _fold_program(constants, padding)
            for constants in range(1, 9)
            for padding in range(0, 4)
        ],
        "algebraic_heavy": [
            _algebraic_program(identities, padding)
            for identities in range(1, 7)
            for padding in range(0, 3)
        ],
        "dead_code_heavy": [
            _dead_program(fold_dead, plain_dead, live_temps)
            for fold_dead in range(0, 5)
            for plain_dead in range(0, 5)
            for live_temps in (0, 1)
            if fold_dead + plain_dead > 0
        ],
        "cse_heavy": [
            _cse_program(copies, extras, padding)
            for copies in range(2, 8)
            for extras in range(0, 3)
            for padding in range(0, 2)
        ],
        "propagation_heavy": [
            _propagation_program(uses, padding)
            for uses in range(1, 7)
            for padding in range(0, 3)
        ],
        "strength_reduction_heavy": [
            _reduction_program(reductions, fillers)
            for reductions in range(1, 4)
            for fillers in range(0, 6)
        ],
        "loop_heavy": [_plain_loop(body) for body in range(1, 8)],
        "straight_line": [
            _straight_program(temporaries, padding)
            for temporaries in range(1, 6)
            for padding in range(0, 3)
        ],
        "mixed": [
            _mixed_program(constants, copies, identities, reductions, dead)
            for constants, copies, identities, reductions, dead in (
                (1, 2, 0, 0, 0),
                (2, 3, 0, 0, 0),
                (3, 2, 0, 0, 1),
                (0, 2, 1, 0, 0),
                (0, 3, 2, 0, 1),
                (0, 4, 1, 0, 2),
                (1, 0, 0, 1, 0),
                (2, 0, 0, 1, 1),
                (0, 0, 1, 1, 0),
                (0, 0, 2, 2, 0),
                (1, 2, 1, 1, 0),
                (0, 2, 0, 1, 1),
                (4, 0, 1, 0, 2),
                (0, 5, 0, 1, 0),
                (2, 2, 2, 0, 0),
            )
        ],
    }
    interleaved: list[tuple[str, TACProgram]] = []
    while any(families.values()):
        for category, programs in families.items():
            if programs:
                interleaved.append((category, programs.pop(0)))
    return interleaved


def _separate_signature(program: TACProgram, seen: set[tuple]) -> TACProgram:
    """Append observable prints while this feature tuple was already built.

    The program is returned either way. Matching signatures are not dropped.
    """
    for _ in range(8):
        signature = feature_signature(extract_features(program))
        if signature not in seen:
            seen.add(signature)
            return program
        program = parse_tac(program.to_tac() + "\nprint 0")
    seen.add(feature_signature(extract_features(program)))
    return program


def _fold_program(constants: int, padding: int) -> TACProgram:
    lines = [f"t{index + 1} = {index + 2} + {index + 4}" for index in range(constants)]
    lines.extend(f"print t{index + 1}" for index in range(constants))
    lines.extend(["print 0"] * padding)
    return parse_tac("\n".join(lines))


def _algebraic_program(identities: int, padding: int) -> TACProgram:
    forms = (
        "{name} + 0",
        "0 + {name}",
        "{name} - 0",
        "{name} * 1",
        "1 * {name}",
        "{name} / 1",
    )
    lines: list[str] = []
    for index in range(identities):
        destination = f"t{index + 1}"
        lines.append(f"{destination} = {forms[index % len(forms)].format(name=f'v{index}')}")
        lines.append(f"print {destination}")
    lines.extend(["print 0"] * padding)
    return parse_tac("\n".join(lines))


def _dead_program(fold_dead: int, plain_dead: int, live_temps: int) -> TACProgram:
    lines: list[str] = []
    for index in range(live_temps):
        lines.append(f"t{index + 1} = left{index} + right{index}")
    if live_temps == 0:
        lines.append("live = left + right")
    for index in range(fold_dead):
        lines.append(f"sparef{index} = {index + 2} * {index + 3}")
    for index in range(plain_dead):
        lines.append(f"spare{index} = {index + 1}")
    if live_temps:
        lines.extend(f"print t{index + 1}" for index in range(live_temps))
    else:
        lines.append("print live")
    return parse_tac("\n".join(lines))


def _cse_program(copies: int, extras: int, padding: int) -> TACProgram:
    lines = [f"t{index + 1} = left + right" for index in range(copies)]
    next_temp = copies + 1
    for index in range(extras):
        lines.append(f"t{next_temp} = name{index} - other{index}")
        next_temp += 1
    lines.extend(f"print t{index + 1}" for index in range(next_temp - 1))
    lines.extend(["print 0"] * padding)
    return parse_tac("\n".join(lines))


def _propagation_program(uses: int, padding: int) -> TACProgram:
    lines = ["value = 12"]
    for index in range(uses):
        lines.append(f"t{index + 1} = value + {index + 3}")
        lines.append(f"print t{index + 1}")
    lines.append("print value")
    lines.extend(["print 0"] * padding)
    return parse_tac("\n".join(lines))


def _reduction_program(
    reductions: int,
    fillers: int,
    bound: int = 8,
    dead: int = 0,
) -> TACProgram:
    names = ("i", "j", "k", "m")[:reductions]
    lines = [f"unused{index} = {index + 1}" for index in range(dead)]
    lines.extend(f"{name} = {start}" for start, name in enumerate(names))
    lines.append("L1:")
    for index, name in enumerate(names):
        lines.append(f"t{index + 1} = {name} * 2")
    for index in range(fillers):
        lines.append(f"u{index} = i + {index + 3}")
        lines.append(f"print u{index}")
    for name in names:
        lines.append(f"{name} = {name} + 1")
    lines.append(f"if i < {bound} goto L1")
    lines.extend(f"print t{index + 1}" for index in range(reductions))
    lines.append("print i")
    return parse_tac("\n".join(lines))


def _plain_loop(body: int, padding: int = 0) -> TACProgram:
    lines = ["i = 0", "L1:"]
    for index in range(body):
        lines.append(f"t{index + 1} = i + {index + 3}")
        lines.append(f"print t{index + 1}")
    lines.extend(["i = i + 1", "if i < 5 goto L1", "print i"])
    lines.extend(["print 0"] * padding)
    return parse_tac("\n".join(lines))


def _straight_program(temporaries: int, padding: int) -> TACProgram:
    lines: list[str] = []
    for index in range(temporaries):
        lines.append(f"t{index + 1} = v{index} + w{index}")
        lines.append(f"print t{index + 1}")
    lines.extend(["print 0"] * padding)
    return parse_tac("\n".join(lines))


def _mixed_program(
    constants: int,
    copies: int,
    identities: int,
    reductions: int,
    dead: int,
) -> TACProgram:
    lines: list[str] = []
    next_temp = 1
    for index in range(constants):
        lines.append(f"t{next_temp} = {index + 4} + {index + 6}")
        lines.append(f"print t{next_temp}")
        next_temp += 1
    for index in range(copies):
        lines.append(f"t{next_temp} = shared + value")
        lines.append(f"print t{next_temp}")
        next_temp += 1
    for index in range(identities):
        lines.append(f"t{next_temp} = v{index} + 0")
        lines.append(f"print t{next_temp}")
        next_temp += 1
    if reductions:
        lines.extend(["i = 0", "L1:"])
        for index in range(reductions):
            lines.append(f"t{next_temp} = i * 2")
            lines.append(f"print t{next_temp}")
            next_temp += 1
        lines.extend(["i = i + 1", "if i < 4 goto L1", "print i"])
    for index in range(dead):
        lines.append(f"spare{index} = {index + 2}")
    return parse_tac("\n".join(lines))
