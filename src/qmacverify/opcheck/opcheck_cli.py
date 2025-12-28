from __future__ import annotations

import argparse
from pathlib import Path

from qmacverify.opcheck.compare_ops import compare_aadd, compare_amul
from qmacverify.opcheck.gen_vectors import gen_vectors
from qmacverify.opcheck.run_iverilog import run_iverilog


def main() -> None:
    parser = argparse.ArgumentParser(description="Run opcheck for approximate ops")
    parser.add_argument("--drop", type=int, default=2)
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rtl_dir = Path(__file__).resolve().parents[1] / "rtl"
    gen_vectors(args.drop, args.num, rtl_dir)

    args.out.mkdir(parents=True, exist_ok=True)

    amul_ok, amul_out = run_iverilog(
        "tb_amul8x8",
        ["tb_amul8x8.v", "amul8x8.v"],
        args.drop,
        rtl_dir,
    )
    if not amul_ok:
        (args.out / "amul_equiv_report.json").write_text(
            "{\n  \"skipped\": true, \"reason\": \"iverilog not found\"\n}\n"
        )
    else:
        compare_amul(rtl_dir / "vectors_amul.txt", amul_out, args.drop, args.out / "amul_equiv_report.json")

    aadd_ok, aadd_out = run_iverilog(
        "tb_aadd32",
        ["tb_aadd32.v", "aadd32.v"],
        args.drop,
        rtl_dir,
    )
    if not aadd_ok:
        (args.out / "aadd_equiv_report.json").write_text(
            "{\n  \"skipped\": true, \"reason\": \"iverilog not found\"\n}\n"
        )
    else:
        compare_aadd(rtl_dir / "vectors_aadd.txt", aadd_out, args.drop, args.out / "aadd_equiv_report.json")


if __name__ == "__main__":
    main()
