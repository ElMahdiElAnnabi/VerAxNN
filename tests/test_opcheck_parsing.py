from qmacverify.opcheck.compare_ops import parse_verilog_lines


def test_parse_verilog_lines():
    text = "1 2 3 4\n-1 -2 -3 -4\n"
    parsed = parse_verilog_lines(text)
    assert parsed == [(1, 2, 3, 4), (-1, -2, -3, -4)]
