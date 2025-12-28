from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from qmacverify.constraints.interval_bounds import (
    Interval,
    conv2d_interval,
    interval_add,
    linear_interval,
    relu_interval,
    requant_interval,
)


def _load_graph(pkg: Path) -> Dict[str, Any]:
    return json.loads((pkg / "graph.json").read_text())


def _load_params(pkg: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(pkg / "params.npz"))


def extract_constraints(pkg: Path, out_path: Path, input_range: Interval) -> None:
    graph = _load_graph(pkg)
    params = _load_params(pkg)

    req_mult = params["requant_multiplier"]
    req_shift = params["requant_shift"]

    tensor_ranges: Dict[str, Interval] = {"input": input_range}
    layers: List[Dict[str, Any]] = []

    for node in graph["nodes"]:
        ntype = node["type"]
        if ntype == "conv2d":
            x_int = tensor_ranges[node["inputs"][0]]
            weight = params[node["params"]["weight_key"]]
            bias = params[node["params"]["bias_key"]]
            w_int = Interval(int(weight.min()), int(weight.max()))
            b_int = Interval(int(bias.min()), int(bias.max()))
            kernel = node["params"]["kernel"]
            n = kernel[0] * kernel[1] * node["params"]["in_ch"]
            acc = conv2d_interval(x_int, w_int, b_int, n)
            layers.append(
                {
                    "name": node["name"],
                    "N": n,
                    "x_min": x_int.min_val,
                    "x_max": x_int.max_val,
                    "w_min": w_int.min_val,
                    "w_max": w_int.max_val,
                    "acc_min": acc.min_val,
                    "acc_max": acc.max_val,
                }
            )
            tensor_ranges[node["outputs"][0]] = acc
        elif ntype == "relu":
            tensor_ranges[node["outputs"][0]] = relu_interval(tensor_ranges[node["inputs"][0]])
        elif ntype == "requant":
            req_id = node["params"]["requant_id"]
            tensor_ranges[node["outputs"][0]] = requant_interval(
                tensor_ranges[node["inputs"][0]],
                int(req_mult[req_id]),
                int(req_shift[req_id]),
            )
        elif ntype in {"flatten", "avg_pool2d", "max_pool2d", "adaptive_avg_pool2d", "adaptive_max_pool2d"}:
            tensor_ranges[node["outputs"][0]] = tensor_ranges[node["inputs"][0]]
        elif ntype == "linear":
            x_int = tensor_ranges[node["inputs"][0]]
            weight = params[node["params"]["weight_key"]]
            bias = params[node["params"]["bias_key"]]
            w_int = Interval(int(weight.min()), int(weight.max()))
            b_int = Interval(int(bias.min()), int(bias.max()))
            n = node["params"]["in_features"]
            tensor_ranges[node["outputs"][0]] = linear_interval(x_int, w_int, b_int, n)
        elif ntype == "add":
            a = tensor_ranges[node["inputs"][0]]
            b = tensor_ranges[node["inputs"][1]]
            tensor_ranges[node["outputs"][0]] = interval_add(a, b)
        else:
            raise ValueError(f"Unsupported node type: {ntype}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"layers": layers}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sound constraints via IBP")
    parser.add_argument("--pkg", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--input-min", type=int, default=-128)
    parser.add_argument("--input-max", type=int, default=127)
    args = parser.parse_args()

    extract_constraints(args.pkg, args.out, Interval(args.input_min, args.input_max))


if __name__ == "__main__":
    main()
