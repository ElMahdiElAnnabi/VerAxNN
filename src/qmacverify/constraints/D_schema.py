from __future__ import annotations

D_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "layers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "N": {"type": "integer"},
                    "x_min": {"type": "integer"},
                    "x_max": {"type": "integer"},
                    "w_min": {"type": "integer"},
                    "w_max": {"type": "integer"},
                    "acc_min": {"type": "integer"},
                    "acc_max": {"type": "integer"},
                },
                "required": ["name", "N", "x_min", "x_max", "w_min", "w_max", "acc_min", "acc_max"],
            },
        }
    },
    "required": ["layers"],
}
