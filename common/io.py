import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

def to_dict(obj):
    return asdict(obj) if is_dataclass(obj) else dict(obj)

def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
