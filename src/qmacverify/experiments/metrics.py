from __future__ import annotations

import numpy as np


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a.astype(np.int64) - b.astype(np.int64)
    return {
        "max_abs_diff": int(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
    }
