from __future__ import annotations

from typing import Callable

from qmacverify.approx_ops.aadd32 import aadd32_trunc
from qmacverify.approx_ops.amul8x8 import amul8x8_trunc


def get_operator(op_type: str, name: str, drop: int) -> Callable[[int, int], int]:
    if op_type == "amul" and name == "trunc":
        return lambda x, y: amul8x8_trunc(x, y, drop=drop)
    if op_type == "aadd" and name == "trunc":
        return lambda x, y: aadd32_trunc(x, y, drop=drop)
    raise ValueError(f"Unknown operator {op_type}:{name}")
