from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple


def run_iverilog(tb: str, sources: list[str], drop: int, workdir: Path) -> Tuple[bool, str]:
    if shutil.which("iverilog") is None:
        return False, "iverilog not found"

    out_exe = workdir / "sim.out"
    cmd = ["iverilog", "-g2012", f"-P{tb}.DROP={drop}", "-o", str(out_exe)] + sources
    subprocess.run(cmd, cwd=workdir, check=True, capture_output=True)
    result = subprocess.run([str(out_exe)], cwd=workdir, check=True, capture_output=True, text=True)
    return True, result.stdout
