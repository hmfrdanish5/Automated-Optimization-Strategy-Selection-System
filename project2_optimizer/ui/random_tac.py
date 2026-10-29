"""Small TAC programs for the live demo.

These strings are not written to any dataset and are not used as labels.
Every program defines the variables it reads and ends with at least one print.
Loops count up to a small bound and then fall through.
"""

from __future__ import annotations

import random


def generate_random_tac(seed: int | None = None) -> str:
    """Return one syntactically valid TAC program.

    A fixed seed repeats the same text. Omitting the seed picks a new program.
    """
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    kind = rng.choice(("fold", "propagate", "algebra", "repeat", "dead", "loop", "mixed"))
    builders = {
        "fold": _fold,
        "propagate": _propagate,
        "algebra": _algebra,
        "repeat": _repeat,
        "dead": _dead,
        "loop": _loop,
        "mixed": _mixed,
    }
    return builders[kind](rng)


def _fold(rng: random.Random) -> str:
    left = rng.randint(2, 20)
    right = rng.randint(1, 9)
    op = rng.choice(("+", "-", "*"))
    lines = [
        f"t1 = {left} {op} {right}",
        f"spare = {rng.randint(1, 30)}",
        "t2 = t1 + 0",
        "print t1",
    ]
    return _text(lines)


def _propagate(rng: random.Random) -> str:
    value = rng.randint(1, 15)
    addend = rng.randint(1, 8)
    lines = [
        f"base = {value}",
        "copy = base",
        f"t1 = copy + {addend}",
        "unused = 7",
        "print t1",
    ]
    return _text(lines)


def _algebra(rng: random.Random) -> str:
    name = rng.choice(("a", "b", "value"))
    start = rng.randint(1, 12)
    lines = [
        f"{name} = {start}",
        f"t1 = {name} + 0",
        f"t2 = 1 * {name}",
        f"t3 = {name} * 0",
        "print t1",
        "print t2",
    ]
    return _text(lines)


def _repeat(rng: random.Random) -> str:
    left = rng.randint(1, 9)
    right = rng.randint(1, 9)
    lines = [
        f"x = {left}",
        f"y = {right}",
        "t1 = x + y",
        "t2 = x + y",
        "t3 = x + y",
        "dead = 4",
        "print t2",
    ]
    return _text(lines)


def _dead(rng: random.Random) -> str:
    live = rng.randint(1, 10)
    lines = [
        f"keep = {live}",
        "gone = 3",
        "also = gone + 1",
        "print keep",
    ]
    return _text(lines)


def _loop(rng: random.Random) -> str:
    bound = rng.randint(3, 6)
    strength = rng.choice((True, False))
    lines = ["i = 0", "L1:"]
    if strength:
        lines.append("t1 = i * 2")
    else:
        lines.append("t1 = i + 3")
    lines.extend(
        [
            "print t1",
            "i = i + 1",
            f"if i < {bound} goto L1",
            "print i",
        ]
    )
    return _text(lines)


def _mixed(rng: random.Random) -> str:
    a = rng.randint(2, 12)
    b = rng.randint(1, 6)
    lines = [
        f"a = {a}",
        f"b = {b}",
        "x = 1",
        "y = 2",
        "z = 5",
        "t1 = a + b",
        "t2 = x + y",
        "t3 = x + y",
        "t4 = z + 0",
        f"unused = {rng.randint(10, 40)}",
        "result = t1 + t4",
        "print result",
    ]
    return _text(lines)


def _text(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"
