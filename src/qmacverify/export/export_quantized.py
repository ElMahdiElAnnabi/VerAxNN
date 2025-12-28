from __future__ import annotations

import argparse
import json
import operator
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import fx, nn

from qmacverify.config.quant_spec import QuantSpec
from qmacverify.export.pack_npz import pack_params

MODEL_REGISTRY = {
    "cnn_small": "qmacverify.models.cnn_small",
    "resnet20_cifar": "qmacverify.models.resnet20_cifar",
}


def _load_model(model_name: str) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name}")
    module_path = MODEL_REGISTRY[model_name]
    module = __import__(module_path, fromlist=["build_model"])
    return module.build_model()


def _quantize_weight(weight: torch.Tensor) -> Tuple[np.ndarray, float]:
    weight_np = weight.detach().cpu().numpy().astype(np.float32)
    max_abs = np.max(np.abs(weight_np))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.round(weight_np / scale).clip(-128, 127).astype(np.int8)
    return q, scale


def _quantize_bias(bias: torch.Tensor | None, scale: float, size: int) -> np.ndarray:
    if bias is None:
        return np.zeros(size, dtype=np.int32)
    bias_np = bias.detach().cpu().numpy().astype(np.float32)
    q = np.round(bias_np / scale).astype(np.int32)
    return q


def _make_requant_params(num_layers: int) -> Tuple[np.ndarray, np.ndarray]:
    multipliers = np.full(num_layers, 1 << 30, dtype=np.int32)
    shifts = np.full(num_layers, 30, dtype=np.int32)
    return multipliers, shifts


def _default_input_shape(model_name: str) -> List[int]:
    if model_name == "cnn_small":
        return [1, 1, 8, 8]
    return [1, 3, 32, 32]


def export_quantized(model_name: str, out_dir: Path, quant_spec_path: Path) -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    model = _load_model(model_name)
    model.eval()

    graph = fx.symbolic_trace(model)

    params: Dict[str, np.ndarray] = {}
    nodes = []
    tensor_map: Dict[fx.Node, str] = {}
    layer_index = 0

    for node in graph.graph.nodes:
        if node.op == "placeholder":
            tensor_map[node] = "input"
            continue
        if node.op == "call_module":
            module = graph.get_submodule(node.target)
            if isinstance(module, nn.Conv2d):
                weight_q, w_scale = _quantize_weight(module.weight)
                bias_q = _quantize_bias(module.bias, w_scale, module.out_channels)
                weight_key = f"weights.{node.target}"
                bias_key = f"bias.{node.target}"
                params[weight_key] = weight_q
                params[bias_key] = bias_q

                input_name = tensor_map[node.args[0]]
                out_name = f"{node.target}_out"
                nodes.append(
                    {
                        "name": node.target,
                        "type": "conv2d",
                        "inputs": [input_name],
                        "outputs": [out_name],
                        "params": {
                            "weight_key": weight_key,
                            "bias_key": bias_key,
                            "stride": list(module.stride),
                            "padding": list(module.padding),
                            "kernel": list(module.kernel_size),
                            "in_ch": module.in_channels,
                            "out_ch": module.out_channels,
                            "groups": module.groups,
                        },
                    }
                )
                tensor_map[node] = out_name

                rq_name = f"requant_{node.target}"
                rq_out = f"{out_name}_rq"
                nodes.append(
                    {
                        "name": rq_name,
                        "type": "requant",
                        "inputs": [out_name],
                        "outputs": [rq_out],
                        "params": {"requant_id": layer_index},
                    }
                )
                tensor_map[node] = rq_out
                layer_index += 1
            elif isinstance(module, nn.ReLU):
                input_name = tensor_map[node.args[0]]
                out_name = f"{node.target}_out"
                nodes.append(
                    {
                        "name": node.target,
                        "type": "relu",
                        "inputs": [input_name],
                        "outputs": [out_name],
                    }
                )
                tensor_map[node] = out_name
            elif isinstance(module, nn.Linear):
                weight_q, w_scale = _quantize_weight(module.weight)
                bias_q = _quantize_bias(module.bias, w_scale, module.out_features)
                weight_key = f"weights.{node.target}"
                bias_key = f"bias.{node.target}"
                params[weight_key] = weight_q
                params[bias_key] = bias_q

                input_name = tensor_map[node.args[0]]
                out_name = f"{node.target}_out"
                nodes.append(
                    {
                        "name": node.target,
                        "type": "linear",
                        "inputs": [input_name],
                        "outputs": [out_name],
                        "params": {
                            "weight_key": weight_key,
                            "bias_key": bias_key,
                            "in_features": module.in_features,
                            "out_features": module.out_features,
                        },
                    }
                )
                tensor_map[node] = out_name

                rq_name = f"requant_{node.target}"
                rq_out = f"{out_name}_rq"
                nodes.append(
                    {
                        "name": rq_name,
                        "type": "requant",
                        "inputs": [out_name],
                        "outputs": [rq_out],
                        "params": {"requant_id": layer_index},
                    }
                )
                tensor_map[node] = rq_out
                layer_index += 1
            elif isinstance(module, nn.Flatten):
                input_name = tensor_map[node.args[0]]
                out_name = f"{node.target}_out"
                nodes.append(
                    {
                        "name": node.target,
                        "type": "flatten",
                        "inputs": [input_name],
                        "outputs": [out_name],
                        "params": {"start_dim": module.start_dim, "end_dim": module.end_dim},
                    }
                )
                tensor_map[node] = out_name
            else:
                raise ValueError(f"Unsupported module: {module}")
        elif node.op == "call_function" and node.target == torch.flatten:
            input_name = tensor_map[node.args[0]]
            out_name = f"flatten_{node.name}"
            nodes.append(
                {
                    "name": out_name,
                    "type": "flatten",
                    "inputs": [input_name],
                    "outputs": [out_name],
                    "params": {"start_dim": node.args[1], "end_dim": node.args[2]},
                }
            )
            tensor_map[node] = out_name
        elif node.op == "call_method" and node.target == "flatten":
            input_name = tensor_map[node.args[0]]
            out_name = f"flatten_{node.name}"
            start_dim = node.args[1] if len(node.args) > 1 else 1
            end_dim = node.args[2] if len(node.args) > 2 else -1
            nodes.append(
                {
                    "name": out_name,
                    "type": "flatten",
                    "inputs": [input_name],
                    "outputs": [out_name],
                    "params": {"start_dim": start_dim, "end_dim": end_dim},
                }
            )
            tensor_map[node] = out_name
        elif node.op == "call_function" and node.target in (torch.add, torch.ops.aten.add.Tensor, operator.add):
            input_a = tensor_map[node.args[0]]
            input_b = tensor_map[node.args[1]]
            out_name = f"add_{node.name}"
            nodes.append(
                {
                    "name": out_name,
                    "type": "add",
                    "inputs": [input_a, input_b],
                    "outputs": [out_name],
                }
            )
            tensor_map[node] = out_name

            rq_name = f"requant_{out_name}"
            rq_out = f"{out_name}_rq"
            nodes.append(
                {
                    "name": rq_name,
                    "type": "requant",
                    "inputs": [out_name],
                    "outputs": [rq_out],
                    "params": {"requant_id": layer_index},
                }
            )
            tensor_map[node] = rq_out
            layer_index += 1
        elif node.op == "call_function" and node.target in (torch.relu, torch.nn.functional.relu):
            input_name = tensor_map[node.args[0]]
            out_name = f"relu_{node.name}"
            nodes.append(
                {
                    "name": out_name,
                    "type": "relu",
                    "inputs": [input_name],
                    "outputs": [out_name],
                }
            )
            tensor_map[node] = out_name
        elif node.op == "output":
            continue
        else:
            raise ValueError(f"Unsupported node: {node.op} {node.target}")

    multipliers, shifts = _make_requant_params(layer_index)
    params["requant_multiplier"] = multipliers
    params["requant_shift"] = shifts

    out_dir.mkdir(parents=True, exist_ok=True)
    graph_payload = {
        "input_shape": _default_input_shape(model_name),
        "nodes": nodes,
    }
    (out_dir / "graph.json").write_text(json.dumps(graph_payload, indent=2))
    pack_params(out_dir / "params.npz", params)

    quant_spec = QuantSpec.load(quant_spec_path)
    quant_spec.dump(out_dir / "quant_spec.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export quantized model package")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--quant-spec",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "default_quant_spec.json",
    )
    args = parser.parse_args()
    export_quantized(args.model, args.out, args.quant_spec)


if __name__ == "__main__":
    main()
