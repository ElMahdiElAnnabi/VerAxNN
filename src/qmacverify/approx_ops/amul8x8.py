from __future__ import annotations


def amul8x8_trunc(x: int, y: int, drop: int = 0) -> int:
    prod = int(x) * int(y)
    if drop <= 0:
        return prod
    mask = ~((1 << drop) - 1)
    return int(prod & mask)
