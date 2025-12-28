from pathlib import Path

import numpy as np

from qmacverify.export.export_quantized import export_quantized
from qmacverify.runner.run_approx_int import run_approx
from qmacverify.runner.run_exact_int import run_exact


def test_runner_smoke(tmp_path: Path):
    export_dir = tmp_path / "export" / "cnn_small"
    export_quantized("cnn_small", export_dir, Path("src/qmacverify/config/default_quant_spec.json"))

    base_out = tmp_path / "results" / "baseline"
    approx_out = tmp_path / "results" / "approx"

    run_exact(export_dir, None, True, base_out)
    run_approx(export_dir, None, True, approx_out, "trunc", 2, "trunc", 2)

    base_npz = np.load(base_out / "outputs.npz")["logits"]
    approx_npz = np.load(approx_out / "outputs.npz")["logits"]

    assert base_npz.shape == approx_npz.shape
