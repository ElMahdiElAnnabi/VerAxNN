from __future__ import annotations

import numpy as np


INT8_MIN = -128
INT8_MAX = 127


def clamp(x: np.ndarray | int, min_val: int, max_val: int) -> np.ndarray | int:
    return np.clip(x, min_val, max_val)


def to_int8(x: np.ndarray | int) -> np.ndarray | int:
    return clamp(x, INT8_MIN, INT8_MAX).astype(np.int8)


def to_int32(x: np.ndarray | int) -> np.ndarray | int:
    return np.array(x, dtype=np.int64).astype(np.int32)


def requant_mul_shift(x: np.ndarray, multiplier: int, shift: int, clamp_min: int, clamp_max: int) -> np.ndarray:
    x64 = x.astype(np.int64)
    prod = x64 * np.int64(multiplier)
    if shift > 0:
        rounding = 1 << (shift - 1)
        prod = prod + rounding
        prod = prod >> shift
    result = clamp(prod, clamp_min, clamp_max)
    return result.astype(np.int8)
