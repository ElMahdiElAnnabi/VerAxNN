from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qmacverify.experiments.metrics import diff_stats


def compare_outputs(baseline_npz: Path, approx_npz: Path, out_path: Path) -> None:
    base = np.load(baseline_npz)["logits"]
    approx = np.load(approx_npz)["logits"]
    report = diff_stats(base, approx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
