from __future__ import annotations

import argparse
from pathlib import Path

from plotting.config import CFG
from plotting.runner import run_all_figures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis multifractal figures.")
    p.add_argument("--mf", default=CFG.mf_npz_path, help="Multifractal .npz file")
    p.add_argument("--train-acc", default=CFG.train_acc_npz_path, help="Training accuracy .npz file")
    p.add_argument("--eos", nargs="*", default=list(CFG.eos_path_candidates), help="Candidate EoS .npz files")
    p.add_argument("--out-dir", default=str(CFG.out_dir), help="Figure output directory")
    p.add_argument("--protocol", default=CFG.primary_protocol)
    p.add_argument("--family", default=CFG.primary_family)
    p.add_argument("--variant", default=CFG.primary_variant)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    CFG.mf_npz_path = args.mf
    CFG.train_acc_npz_path = args.train_acc
    CFG.eos_path_candidates = tuple(args.eos)
    CFG.out_dir = Path(args.out_dir)
    CFG.primary_protocol = args.protocol
    CFG.primary_family = args.family
    CFG.primary_variant = args.variant
    run_all_figures(CFG)


if __name__ == "__main__":
    main()
