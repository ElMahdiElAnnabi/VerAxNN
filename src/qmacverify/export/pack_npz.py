from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np


def pack_params(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
