from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qmacverify.constraints.extract_D import extract_constraints
from qmacverify.constraints.interval_bounds import Interval
from qmacverify.export.export_quantized import export_quantized
from qmacverify.experiments.compare_outputs import compare_outputs
from qmacverify.opcheck import opcheck_cli
from qmacverify.runner.run_approx_int import run_approx
from qmacverify.runner.run_exact_int import run_exact


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end demo pipeline")
    parser.add_argument("--model", required=True, choices=["cnn_small", "resnet20_cifar"])
    parser.add_argument("--drop", type=int, default=2)
    parser.add_argument("--num-opcheck", type=int, default=2000)
    args = parser.parse_args()

    export_dir = Path("export") / args.model
    export_quantized(args.model, export_dir, Path("src/qmacverify/config/default_quant_spec.json"))

    base_out = Path("results/baseline") / args.model
    approx_out = Path("results/approx") / args.model

    run_exact(export_dir, None, True, base_out)
    run_approx(
        export_dir,
        None,
        True,
        approx_out,
        "trunc",
        args.drop,
        "trunc",
        args.drop,
    )

    compare_outputs(
        base_out / "outputs.npz",
        approx_out / "outputs.npz",
        Path("results/compare") / args.model / "diff_report.json",
    )

    extract_constraints(export_dir, Path("constraints") / args.model / "D_sound.json", Interval(-128, 127))

    argv_backup = sys.argv
    try:
        sys.argv = [
            "opcheck",
            "--drop",
            str(args.drop),
            "--num",
            str(args.num_opcheck),
            "--out",
            "results/opcheck",
        ]
        opcheck_cli.main()
    finally:
        sys.argv = argv_backup


if __name__ == "__main__":
    main()
