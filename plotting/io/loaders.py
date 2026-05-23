from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from plotting.config import CFG
from plotting.utils import unpack_obj


def load_npz_dict(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: unpack_obj(v) for k, v in data.items()}


def load_train_acc_npz(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        print(f"[train_acc] file not found: {path}")
        return None
    data = np.load(p, allow_pickle=True)
    print(f"[train_acc] loaded: {path}")
    return {k: unpack_obj(v) for k, v in data.items()}


def resolve_eos_path() -> Optional[str]:
    for p in CFG.eos_path_candidates:
        if Path(p).exists():
            return p
    return None
