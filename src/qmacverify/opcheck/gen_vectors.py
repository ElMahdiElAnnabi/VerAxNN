from __future__ import annotations

from pathlib import Path
import numpy as np


def gen_vectors(drop: int, num: int, out_dir: Path) -> None:
    rng = np.random.default_rng(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    amul_path = out_dir / "vectors_amul.txt"
    aadd_path = out_dir / "vectors_aadd.txt"

    with amul_path.open("w") as f:
        for _ in range(num):
            x = int(rng.integers(-128, 127))
            y = int(rng.integers(-128, 127))
            f.write(f"{x} {y}\n")

    with aadd_path.open("w") as f:
        for _ in range(num):
            x = int(rng.integers(-2**31, 2**31 - 1))
            y = int(rng.integers(-2**31, 2**31 - 1))
            f.write(f"{x} {y}\n")
