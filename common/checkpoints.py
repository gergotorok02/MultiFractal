import json
import re
from pathlib import Path
from typing import Optional

def numeric_key(path: Path) -> int:
    match = re.search(r"_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1

def select_by_fraction(paths, fractions):
    if not paths:
        return []
    picked, seen = [], set()
    for frac in fractions:
        i = int(round(min(max(float(frac), 0.0), 1.0) * (len(paths) - 1)))
        if i not in seen:
            picked.append(paths[i]); seen.add(i)
    return picked

def discover(root: str, kind: str = "step", every: int = 1, max_count: Optional[int] = None,
             include_best: bool = False, include_last: bool = False):
    ckpt_dir = Path(root) / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    paths = []
    if kind in ("step", "all"):
        paths += sorted(ckpt_dir.glob("step_*.pt"), key=numeric_key)
    if kind in ("epoch", "all"):
        paths += sorted(ckpt_dir.glob("epoch_*.pt"), key=numeric_key)
    paths = sorted(set(paths), key=lambda p: (numeric_key(p), p.name))
    paths = paths[::max(1, every)]
    paths = paths[:max_count] if max_count is not None else paths
    for name, flag in [("best.pt", include_best), ("last.pt", include_last)]:
        p = ckpt_dir / name
        if flag and p.exists(): paths.append(p)
    return list(dict.fromkeys(paths))

def load_training_log(root: str, path: str | None = None):
    candidate = Path(path) if path else Path(root) / "training_log.json"
    return json.loads(candidate.read_text()) if candidate.exists() else None

def step_log_index(log):
    return {} if log is None else {int(e["global_step"]): e for e in log.get("step_log", [])}
