from qmacverify.approx_ops.aadd32 import aadd32_trunc
from qmacverify.approx_ops.amul8x8 import amul8x8_trunc


def test_amul_trunc_drop_bits():
    assert amul8x8_trunc(3, 5, drop=2) == (15 & ~0b11)
    assert amul8x8_trunc(-3, 5, drop=2) == (-15 & ~0b11)


def test_aadd_trunc_drop_bits():
    assert aadd32_trunc(7, 1, drop=2) == (8 & ~0b11)
    assert aadd32_trunc(-7, -1, drop=2) == (-8 & ~0b11)
