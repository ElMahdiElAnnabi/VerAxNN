from __future__ import annotations

import argparse
import sys

from qmacverify.constraints import extract_D
from qmacverify.export import export_quantized
from qmacverify.experiments import demo_pipeline
from qmacverify.opcheck import opcheck_cli
from qmacverify.runner import run_approx_int, run_exact_int


def main() -> None:
    parser = argparse.ArgumentParser(description="qmacverify CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("export")
    sub.add_parser("exact")
    sub.add_parser("approx")
    sub.add_parser("constraints")
    sub.add_parser("opcheck")
    sub.add_parser("demo")

    args, remaining = parser.parse_known_args()
    if args.cmd == "export":
        sys.argv = ["export"] + remaining
        export_quantized.main()
    elif args.cmd == "exact":
        sys.argv = ["exact"] + remaining
        run_exact_int.main()
    elif args.cmd == "approx":
        sys.argv = ["approx"] + remaining
        run_approx_int.main()
    elif args.cmd == "constraints":
        sys.argv = ["constraints"] + remaining
        extract_D.main()
    elif args.cmd == "opcheck":
        sys.argv = ["opcheck"] + remaining
        opcheck_cli.main()
    elif args.cmd == "demo":
        sys.argv = ["demo"] + remaining
        demo_pipeline.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
