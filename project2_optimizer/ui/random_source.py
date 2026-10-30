"""Small C-like programs for the live demo (not added to any dataset)."""

from __future__ import annotations

import random


def generate_random_source(seed: int | None = None) -> str:
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    kind = rng.choice(("arith", "if_else", "while_loop", "dead", "mixed"))
    builders = {
        "arith": _arith,
        "if_else": _if_else,
        "while_loop": _while_loop,
        "dead": _dead,
        "mixed": _mixed,
    }
    return builders[kind](rng)


def _arith(rng: random.Random) -> str:
    a, b, c = rng.randint(2, 12), rng.randint(1, 9), rng.randint(1, 5)
    return f"""int a = {a};
int b = {b};
int c = a + b * {c};
print(c);
"""


def _if_else(rng: random.Random) -> str:
    a, b = rng.randint(1, 8), rng.randint(1, 8)
    return f"""int a = {a};
int b = {b};
if (a < b) {{
    print(a);
}} else {{
    print(b);
}}
"""


def _while_loop(rng: random.Random) -> str:
    bound = rng.randint(3, 6)
    return f"""int i = 0;
int sum = 0;
while (i < {bound}) {{
    sum = sum + i;
    i = i + 1;
}}
print(sum);
"""


def _dead(rng: random.Random) -> str:
    live = rng.randint(1, 10)
    return f"""int keep = {live};
int gone = 3;
int also = gone + 1;
print(keep);
"""


def _mixed(rng: random.Random) -> str:
    a, b = rng.randint(2, 10), rng.randint(1, 6)
    return f"""int a = {a};
int b = {b};
int t1 = a + b;
int t2 = a + b;
int t3 = t1 + 0;
print(t2);
print(t3);
"""


__all__ = ["generate_random_source"]
