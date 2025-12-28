from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Any

import numpy as np

from qmacverify.approx_ops.registry import get_operator
from qmacverify.runner.int_math import requant_mul_shift, INT8_MAX, INT8_MIN
from qmacverify.runner.ops_exact import flatten, relu_int32


def _load_graph(pkg: Path) -> Dict[str, Any]:
    return json.loads((pkg / "graph.json").read_text())


def _load_params(pkg: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(pkg / "params.npz"))


def _random_input(shape: list[int]) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(-128, 127, size=shape, dtype=np.int8)


def conv2d_approx(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: tuple[int, int],
    padding: tuple[int, int],
    amul: Callable[[int, int], int],
    aadd: Callable[[int, int], int],
) -> np.ndarray:
    n, c_in, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    pad_h, pad_w = padding
    out_h = (h + 2 * pad_h - kh) // stride[0] + 1
    out_w = (w + 2 * pad_w - kw) // stride[1] + 1

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.int32)

    for batch in range(n):
        for oc in range(c_out):
            for oy in range(out_h):
                for ox in range(out_w):
                    acc = int(bias[oc])
                    for ic in range(c_in):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = oy * stride[0] + ky
                                ix = ox * stride[1] + kx
                                prod = amul(int(x_pad[batch, ic, iy, ix]), int(weight[oc, ic, ky, kx]))
                                acc = aadd(acc, prod)
                    out[batch, oc, oy, ox] = acc
    return out


def linear_approx(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, amul: Callable[[int, int], int], aadd: Callable[[int, int], int]
) -> np.ndarray:
    out = np.zeros((x.shape[0], weight.shape[0]), dtype=np.int32)
    for b in range(x.shape[0]):
        for o in range(weight.shape[0]):
            acc = int(bias[o])
            for i in range(weight.shape[1]):
                prod = amul(int(x[b, i]), int(weight[o, i]))
                acc = aadd(acc, prod)
            out[b, o] = acc
    return out


def run_approx(
    pkg: Path,
    input_path: Path | None,
    random_input: bool,
    out_dir: Path,
    amul_name: str,
    amul_drop: int,
    aadd_name: str,
    aadd_drop: int,
) -> None:
    graph = _load_graph(pkg)
    params = _load_params(pkg)
    tensors: Dict[str, np.ndarray] = {}

    if random_input or input_path is None:
        tensors["input"] = _random_input(graph["input_shape"])
    else:
        tensors["input"] = np.load(input_path)["input"]

    req_mult = params["requant_multiplier"]
    req_shift = params["requant_shift"]

    amul = get_operator("amul", amul_name, drop=amul_drop)
    aadd = get_operator("aadd", aadd_name, drop=aadd_drop)

    for node in graph["nodes"]:
        ntype = node["type"]
        if ntype == "conv2d":
            weight = params[node["params"]["weight_key"]]
            bias = params[node["params"]["bias_key"]]
            x = tensors[node["inputs"][0]]
            out = conv2d_approx(
                x,
                weight,
                bias,
                stride=tuple(node["params"]["stride"]),
                padding=tuple(node["params"]["padding"]),
                amul=amul,
                aadd=aadd,
            )
            tensors[node["outputs"][0]] = out
        elif ntype == "relu":
            tensors[node["outputs"][0]] = relu_int32(tensors[node["inputs"][0]])
        elif ntype == "requant":
            req_id = node["params"]["requant_id"]
            tensors[node["outputs"][0]] = requant_mul_shift(
                tensors[node["inputs"][0]],
                int(req_mult[req_id]),
                int(req_shift[req_id]),
                INT8_MIN,
                INT8_MAX,
            )
        elif ntype == "flatten":
            params_dict = node.get("params", {})
            tensors[node["outputs"][0]] = flatten(
                tensors[node["inputs"][0]],
                start_dim=params_dict.get("start_dim", 1),
                end_dim=params_dict.get("end_dim", -1),
            )
        elif ntype == "linear":
            weight = params[node["params"]["weight_key"]]
            bias = params[node["params"]["bias_key"]]
            x = tensors[node["inputs"][0]]
            tensors[node["outputs"][0]] = linear_approx(x, weight, bias, amul, aadd)
        elif ntype == "add":
            a = tensors[node["inputs"][0]]
            b = tensors[node["inputs"][1]]
            out = np.vectorize(lambda x_val, y_val: aadd(int(x_val), int(y_val)))(a, b).astype(np.int32)
            tensors[node["outputs"][0]] = out
        else:
            raise ValueError(f"Unsupported node type: {ntype}")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_name = graph["nodes"][-1]["outputs"][0]
    np.savez(out_dir / "outputs.npz", logits=tensors[output_name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run approximate integer inference")
    parser.add_argument("--pkg", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--random-input", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--amul", default="trunc")
    parser.add_argument("--amul-drop", type=int, default=2)
    parser.add_argument("--aadd", default="trunc")
    parser.add_argument("--aadd-drop", type=int, default=2)
    args = parser.parse_args()

    run_approx(
        args.pkg,
        args.input,
        args.random_input,
        args.out,
        args.amul,
        args.amul_drop,
        args.aadd,
        args.aadd_drop,
    )


if __name__ == "__main__":
    main()
