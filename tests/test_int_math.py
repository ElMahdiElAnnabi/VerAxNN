import numpy as np

from qmacverify.runner.int_math import clamp, requant_mul_shift, INT8_MAX, INT8_MIN


def test_clamp_int8_bounds():
    assert clamp(-200, INT8_MIN, INT8_MAX) == INT8_MIN
    assert clamp(200, INT8_MIN, INT8_MAX) == INT8_MAX


def test_requant_mul_shift_identity():
    x = np.array([1000, -1000], dtype=np.int32)
    out = requant_mul_shift(x, multiplier=1 << 30, shift=30, clamp_min=INT8_MIN, clamp_max=INT8_MAX)
    assert out.dtype == np.int8
    assert out[0] == 100
    assert out[1] == -100


def test_requant_rounding():
    x = np.array([3], dtype=np.int32)
    out = requant_mul_shift(x, multiplier=1 << 1, shift=1, clamp_min=INT8_MIN, clamp_max=INT8_MAX)
    assert int(out[0]) == 3
