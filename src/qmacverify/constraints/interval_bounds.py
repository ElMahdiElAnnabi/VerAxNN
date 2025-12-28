from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from qmacverify.runner.int_math import INT8_MAX, INT8_MIN, requant_mul_shift


@dataclass
class Interval:
    min_val: int
    max_val: int

    def clamp(self, min_val: int, max_val: int) -> "Interval":
        return Interval(max(self.min_val, min_val), min(self.max_val, max_val))


def interval_mul(a: Interval, b: Interval) -> Interval:
    candidates = [a.min_val * b.min_val, a.min_val * b.max_val, a.max_val * b.min_val, a.max_val * b.max_val]
    return Interval(int(min(candidates)), int(max(candidates)))


def interval_add(a: Interval, b: Interval) -> Interval:
    return Interval(a.min_val + b.min_val, a.max_val + b.max_val)


def conv2d_interval(x: Interval, w: Interval, bias: Interval, n: int) -> Interval:
    prod = interval_mul(x, w)
    acc_min = bias.min_val + n * prod.min_val
    acc_max = bias.max_val + n * prod.max_val
    return Interval(int(acc_min), int(acc_max))


def linear_interval(x: Interval, w: Interval, bias: Interval, n: int) -> Interval:
    return conv2d_interval(x, w, bias, n)


def relu_interval(x: Interval) -> Interval:
    return Interval(max(0, x.min_val), max(0, x.max_val))


def requant_interval(x: Interval, multiplier: int, shift: int) -> Interval:
    vals = [x.min_val, x.max_val]
    outs = [
        int(requant_mul_shift(np.array([v], dtype=np.int32), multiplier, shift, INT8_MIN, INT8_MAX)[0])
        for v in vals
    ]
    return Interval(min(outs), max(outs)).clamp(INT8_MIN, INT8_MAX)
