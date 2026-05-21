import argparse
import json
from dataclasses import fields
from pathlib import Path

def load_dataclass(cls, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    for f in fields(cls):
        parser.add_argument(f"--{f.name}", type=str, default=None)
    args = parser.parse_args(argv)
    values = json.loads(Path(args.config).read_text()) if args.config else {}
    for f in fields(cls):
        raw = getattr(args, f.name)
        if raw is not None:
            values[f.name] = cast_value(raw, f.default)
    return cls(**values)

def cast_value(raw: str, default):
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "y"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if default is None:
        return raw
    return raw
