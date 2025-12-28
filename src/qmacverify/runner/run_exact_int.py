from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from qmacverify.runner.ops_exact import (
    add_int32,
    adaptive_avg_pool2d_int32,
    adaptive_max_pool2d_int32,
    avg_pool2d_int32,
    conv2d_int8,
    flatten,
    linear_int8,
    max_pool2d_int32,
    relu_int32,
    requant_int8,
)


def _load_graph(pkg: Path) -> Dict[str, Any]:
    return json.loads((pkg / "graph.json").read_text())


def _load_params(pkg: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(pkg / "params.npz"))


def _random_input(shape: list[int]) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(-128, 127, size=shape, dtype=np.int8)


def run_exact(pkg: Path, input_path: Path | None, random_input: bool, out_dir: Path) -> None:
    graph = _load_graph(pkg)
    params = _load_params(pkg)
    tensors: Dict[str, np.ndarray] = {}

    if random_input or input_path is None:
        tensors["input"] = _random_input(graph["input_shape"])
    else:
        tensors["input"] = np.load(input_path)["input"]

    req_mult = params["requant_multiplier"]
    req_shift = params["requant_shift"]

    for node in graph["nodes"]:
        ntype = node["type"]
        if ntype == "conv2d":
            weight = params[node["params"]["weight_key"]]
            bias = params[node["params"]["bias_key"]]
            x = tensors[node["inputs"][0]]
            out = conv2d_int8(
                x,
                weight,
                bias,
                stride=tuple(node["params"]["stride"]),
                padding=tuple(node["params"]["padding"]),
            )
            tensors[node["outputs"][0]] = out
        elif ntype == "relu":
            tensors[node["outputs"][0]] = relu_int32(tensors[node["inputs"][0]])
        elif ntype == "requant":
            req_id = node["params"]["requant_id"]
            tensors[node["outputs"][0]] = requant_int8(
                tensors[node["inputs"][0]],
                int(req_mult[req_id]),
                int(req_shift[req_id]),
            )
        elif ntype == "flatten":
            params_dict = node.get("params", {})
            tensors[node["outputs"][0]] = flatten(
                tensors[node["inputs"][0]],
                start_dim=params_dict.get("start_dim", 1),
                end_dim=params_dict.get("end_dim", -1),
            )
        elif ntype == "avg_pool2d":
            params_dict = node.get("params", {})
            tensors[node["outputs"][0]] = avg_pool2d_int32(
                tensors[node["inputs"][0]],
                kernel=tuple(params_dict.get("kernel", [2, 2])),
                stride=tuple(params_dict.get("stride", [2, 2])),
                padding=tuple(params_dict.get("padding", [0, 0])),
            )
        elif ntype == "max_pool2d":
            params_dict = node.get("params", {})
            tensors[node["outputs"][0]] = max_pool2d_int32(
                tensors[node["inputs"][0]],
                kernel=tuple(params_dict.get("kernel", [2, 2])),
                stride=tuple(params_dict.get("stride", [2, 2])),
                padding=tuple(params_dict.get("padding", [0, 0])),
            )
        elif ntype == "adaptive_avg_pool2d":
            params_dict = node.get("params", {})
            tensors[node["outputs"][0]] = adaptive_avg_pool2d_int32(
                tensors[node["inputs"][0]],
                output_size=tuple(params_dict.get("output_size", [1, 1])),
            )
        elif ntype == "adaptive_max_pool2d":
            params_dict = node.get("params", {})
            tensors[node["outputs"][0]] = adaptive_max_pool2d_int32(
                tensors[node["inputs"][0]],
                output_size=tuple(params_dict.get("output_size", [1, 1])),
            )
        elif ntype == "linear":
            weight = params[node["params"]["weight_key"]]
            bias = params[node["params"]["bias_key"]]
            x = tensors[node["inputs"][0]]
            tensors[node["outputs"][0]] = linear_int8(x, weight, bias)
        elif ntype == "add":
            a = tensors[node["inputs"][0]]
            b = tensors[node["inputs"][1]]
            tensors[node["outputs"][0]] = add_int32(a, b)
        else:
            raise ValueError(f"Unsupported node type: {ntype}")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_name = graph["nodes"][-1]["outputs"][0]
    np.savez(out_dir / "outputs.npz", logits=tensors[output_name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact integer inference")
    parser.add_argument("--pkg", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--random-input", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    run_exact(args.pkg, args.input, args.random_input, args.out)


if __name__ == "__main__":
    main()
