from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class RequantSpec:
    mode: str
    multiplier_bits: int
    rounding: str
    clamp_min: int
    clamp_max: int


@dataclass(frozen=True)
class QuantSpec:
    activation_dtype: str
    weight_dtype: str
    bias_dtype: str
    acc_dtype: str
    symmetric: bool
    zero_point_activation: int
    zero_point_weight: int
    zero_point_output: int
    requant: RequantSpec

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "QuantSpec":
        rq = payload["requant"]
        return QuantSpec(
            activation_dtype=payload["activation_dtype"],
            weight_dtype=payload["weight_dtype"],
            bias_dtype=payload["bias_dtype"],
            acc_dtype=payload["acc_dtype"],
            symmetric=payload["symmetric"],
            zero_point_activation=payload["zero_point_activation"],
            zero_point_weight=payload["zero_point_weight"],
            zero_point_output=payload["zero_point_output"],
            requant=RequantSpec(
                mode=rq["mode"],
                multiplier_bits=rq["multiplier_bits"],
                rounding=rq["rounding"],
                clamp_min=rq["clamp_min"],
                clamp_max=rq["clamp_max"],
            ),
        )

    @staticmethod
    def load(path: Path) -> "QuantSpec":
        return QuantSpec.from_dict(json.loads(path.read_text()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activation_dtype": self.activation_dtype,
            "weight_dtype": self.weight_dtype,
            "bias_dtype": self.bias_dtype,
            "acc_dtype": self.acc_dtype,
            "symmetric": self.symmetric,
            "zero_point_activation": self.zero_point_activation,
            "zero_point_weight": self.zero_point_weight,
            "zero_point_output": self.zero_point_output,
            "requant": {
                "mode": self.requant.mode,
                "multiplier_bits": self.requant.multiplier_bits,
                "rounding": self.requant.rounding,
                "clamp_min": self.requant.clamp_min,
                "clamp_max": self.requant.clamp_max,
            },
        }

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))
