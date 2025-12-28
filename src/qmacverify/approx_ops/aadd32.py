from __future__ import annotations


def aadd32_trunc(x: int, y: int, drop: int = 0) -> int:
    total = int(x) + int(y)
    if drop <= 0:
        return total
    mask = ~((1 << drop) - 1)
    return int(total & mask)
