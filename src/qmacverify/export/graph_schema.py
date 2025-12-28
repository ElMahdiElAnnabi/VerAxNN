from __future__ import annotations

GRAPH_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "input_shape": {"type": "array", "items": {"type": "integer"}},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "params": {"type": "object"},
                },
                "required": ["name", "type", "inputs", "outputs"],
            },
        },
    },
    "required": ["input_shape", "nodes"],
}
