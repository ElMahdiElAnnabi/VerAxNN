from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from qmacverify.approx_ops.aadd32 import aadd32_trunc
from qmacverify.approx_ops.amul8x8 import amul8x8_trunc


def parse_verilog_lines(text: str) -> List[Tuple[int, int, int, int]]:
    lines = []
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        x, y, exact, approx = map(int, parts)
        lines.append((x, y, exact, approx))
    return lines


def compare_outputs(
    vectors: List[Tuple[int, int]],
    verilog_lines: List[Tuple[int, int, int, int]],
    op: Callable[[int, int, int], int],
    drop: int,
) -> Dict[str, object]:
    mismatches = []
    for idx, (x, y) in enumerate(vectors):
        py_val = op(x, y, drop)
        if idx >= len(verilog_lines):
            mismatches.append(
                {
                    "index": idx,
                    "x": x,
                    "y": y,
                    "py": py_val,
                    "verilog": None,
                    "verilog_exact": None,
                }
            )
            break
        vx, vy, v_exact, v_approx = verilog_lines[idx]
        if (vx, vy) != (x, y) or v_approx != py_val:
            mismatches.append(
                {
                    "index": idx,
                    "x": x,
                    "y": y,
                    "py": py_val,
                    "verilog": v_approx,
                    "verilog_exact": v_exact,
                }
            )
            break
    return {
        "vectors_tested": len(vectors),
        "mismatches": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
    }


def compare_amul(vectors_path: Path, verilog_out: str, drop: int, out_path: Path) -> None:
    vectors = [tuple(map(int, line.split())) for line in vectors_path.read_text().strip().splitlines()]
    verilog_lines = parse_verilog_lines(verilog_out)
    report = compare_outputs(vectors, verilog_lines, amul8x8_trunc, drop)
    out_path.write_text(json.dumps(report, indent=2))


def compare_aadd(vectors_path: Path, verilog_out: str, drop: int, out_path: Path) -> None:
    vectors = [tuple(map(int, line.split())) for line in vectors_path.read_text().strip().splitlines()]
    verilog_lines = parse_verilog_lines(verilog_out)
    report = compare_outputs(vectors, verilog_lines, aadd32_trunc, drop)
    out_path.write_text(json.dumps(report, indent=2))
