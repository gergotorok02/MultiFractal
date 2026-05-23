from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from plotting.utils import as_list_of_dicts, maybe_scalar, safe_arr


def get_protocol_entries(payload: Dict[str, Any], protocol_name: str) -> List[Dict[str, Any]]:
    protocol_logs = payload.get("protocol_logs", {})
    if isinstance(protocol_logs, dict):
        return protocol_logs.get(protocol_name, [])
    if isinstance(protocol_logs, np.ndarray) and protocol_logs.dtype == object and protocol_logs.shape == ():
        return protocol_logs.item().get(protocol_name, [])
    return []


def get_family(entry: Dict[str, Any], family_name: str) -> Optional[Dict[str, Any]]:
    return entry.get("families", {}).get(family_name, None)


def get_per_batch(entry: Dict[str, Any], family_name: str) -> List[Dict[str, Any]]:
    fam = get_family(entry, family_name)
    if fam is None or not fam.get("valid", False):
        return []
    return as_list_of_dicts(fam.get("per_batch", []))


def get_variant_names(entries: List[Dict[str, Any]], family_name: str) -> List[str]:
    names = set()
    for e in entries:
        for b in get_per_batch(e, family_name):
            agg = b.get("aggregated", {})
            for k in agg.get("variant_aggregates", {}).keys():
                names.add(k)
    return sorted(names)


def maybe_epoch_x(entries: List[Dict[str, Any]]) -> np.ndarray:
    x = np.asarray([e.get("epoch", np.nan) for e in entries], dtype=np.float64)
    if np.all(~np.isfinite(x)):
        x = np.arange(len(entries), dtype=np.float64)
    return x


def maybe_step_x(entries: List[Dict[str, Any]]) -> np.ndarray:
    x = np.asarray([e.get("step", np.nan) for e in entries], dtype=np.float64)
    if np.all(~np.isfinite(x)):
        x = np.arange(len(entries), dtype=np.float64)
    return x


def test_acc_series(entries: List[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([e.get("test_acc", np.nan) for e in entries], dtype=np.float64)


def align_train_acc_to_entries(entries: List[Dict[str, Any]], train_payload: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    n = len(entries)
    out = {
        "train_acc": np.full(n, np.nan, dtype=np.float64),
        "train_loss": np.full(n, np.nan, dtype=np.float64),
        "train_eval_test_acc": np.full(n, np.nan, dtype=np.float64),
        "train_eval_test_loss": np.full(n, np.nan, dtype=np.float64),
        "train_eval_lr": np.full(n, np.nan, dtype=np.float64),
    }
    if train_payload is None:
        return out

    ckpt_names = np.asarray(train_payload.get("checkpoint_name", []), dtype=object)
    steps = safe_arr(train_payload.get("step", []))
    epochs = safe_arr(train_payload.get("epoch", []))
    train_acc = safe_arr(train_payload.get("train_acc", []))
    train_loss = safe_arr(train_payload.get("train_loss", []))
    train_eval_test_acc = safe_arr(train_payload.get("test_acc", []))
    train_eval_test_loss = safe_arr(train_payload.get("test_loss", []))
    train_eval_lr = safe_arr(train_payload.get("lr", []))
    m = train_acc.size
    if m == 0:
        print("[train_acc] train_acc field missing or empty")
        return out

    def copy_from(src_idx: int, dst_idx: int) -> None:
        if 0 <= src_idx < m and 0 <= dst_idx < n:
            out["train_acc"][dst_idx] = train_acc[src_idx] if src_idx < train_acc.size else np.nan
            out["train_loss"][dst_idx] = train_loss[src_idx] if src_idx < train_loss.size else np.nan
            out["train_eval_test_acc"][dst_idx] = train_eval_test_acc[src_idx] if src_idx < train_eval_test_acc.size else np.nan
            out["train_eval_test_loss"][dst_idx] = train_eval_test_loss[src_idx] if src_idx < train_eval_test_loss.size else np.nan
            out["train_eval_lr"][dst_idx] = train_eval_lr[src_idx] if src_idx < train_eval_lr.size else np.nan

    # 1. Exact checkpoint_name matching.
    name_to_idx: Dict[str, int] = {}
    for i, name in enumerate(ckpt_names):
        if name is not None:
            name_to_idx[str(name)] = i

    matched_by_name = 0
    for j, e in enumerate(entries):
        name = str(e.get("checkpoint_name", ""))
        if name in name_to_idx:
            copy_from(name_to_idx[name], j)
            matched_by_name += 1

    if matched_by_name >= max(1, n // 2):
        print(f"[train_acc] aligned by checkpoint_name: {matched_by_name}/{n}")
        return out

    # 2. Nearest step matching.
    entry_steps = maybe_step_x(entries)
    if steps.size == m and np.isfinite(steps).any() and np.isfinite(entry_steps).any():
        valid = np.isfinite(steps)
        xs = steps[valid]
        idxs = np.where(valid)[0]
        for j, s in enumerate(entry_steps):
            if np.isfinite(s):
                src = idxs[int(np.argmin(np.abs(xs - s)))]
                copy_from(src, j)
        print("[train_acc] aligned by nearest step")
        return out

    # 3. Nearest epoch matching.
    entry_epochs = maybe_epoch_x(entries)
    if epochs.size == m and np.isfinite(epochs).any() and np.isfinite(entry_epochs).any():
        valid = np.isfinite(epochs)
        xs = epochs[valid]
        idxs = np.where(valid)[0]
        for j, ep in enumerate(entry_epochs):
            if np.isfinite(ep):
                src = idxs[int(np.argmin(np.abs(xs - ep)))]
                copy_from(src, j)
        print("[train_acc] aligned by nearest epoch")
        return out

    # 4. Direct-order fallback.
    k = min(n, m)
    for j in range(k):
        copy_from(j, j)
    print(f"[train_acc] aligned by direct order fallback: {k}/{n}")
    return out


def train_acc_series(train_aligned: Dict[str, np.ndarray], n: int) -> np.ndarray:
    if train_aligned is None or "train_acc" not in train_aligned:
        return np.full(n, np.nan, dtype=np.float64)
    arr = safe_arr(train_aligned["train_acc"])
    if arr.size != n:
        out = np.full(n, np.nan, dtype=np.float64)
        out[: min(n, arr.size)] = arr[: min(n, arr.size)]
        return out
    return arr


def extract_eos_series(payload: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if payload is None:
        return {}
    logs = as_list_of_dicts(payload.get("logs", None))
    out: Dict[str, np.ndarray] = {}
    fields = [
        "checkpoint_index", "step", "epoch", "epoch_step", "lr",
        "loss", "grad_norm", "update_norm", "test_acc",
        "lambda_max", "lambda_max_residual", "eta_lambda_max",
        "lambda_min", "lambda_min_residual",
        "trace_H", "trace_H_std", "trace_H_per_dim",
        "gHg_over_gg", "uHu_over_uu", "eta_gHg", "eta_uHu",
    ]
    if len(logs) > 0:
        for f in fields:
            out[f] = np.asarray([entry.get(f, np.nan) for entry in logs], dtype=np.float64)
        return out
    for f in fields:
        if f in payload:
            out[f] = safe_arr(payload[f])
    return out


def align_eos_to_entries(entries: List[Dict[str, Any]], eos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if len(eos) == 0:
        return {}

    n = len(entries)
    epoch_ref = maybe_epoch_x(entries)
    step_ref = maybe_step_x(entries)
    eos_epoch = safe_arr(eos.get("epoch", []))
    eos_step = safe_arr(eos.get("step", []))
    aligned: Dict[str, np.ndarray] = {}

    for key, arr in eos.items():
        arr = safe_arr(arr)
        if arr.size == n:
            aligned[key] = arr
            continue

        dst = np.full(n, np.nan)
        if eos_step.size == arr.size and np.isfinite(step_ref).any() and np.isfinite(eos_step).any():
            valid = np.isfinite(eos_step) & np.isfinite(arr)
            if valid.any():
                xs = eos_step[valid]
                ys = arr[valid]
                for i, x in enumerate(step_ref):
                    if np.isfinite(x):
                        dst[i] = ys[int(np.argmin(np.abs(xs - x)))]
                aligned[key] = dst
                continue

        if eos_epoch.size == arr.size and np.isfinite(epoch_ref).any() and np.isfinite(eos_epoch).any():
            valid = np.isfinite(eos_epoch) & np.isfinite(arr)
            if valid.any():
                xs = eos_epoch[valid]
                ys = arr[valid]
                for i, x in enumerate(epoch_ref):
                    if np.isfinite(x):
                        dst[i] = ys[int(np.argmin(np.abs(xs - x)))]
                aligned[key] = dst
                continue

        dst[: min(n, arr.size)] = arr[: min(n, arr.size)]
        aligned[key] = dst

    # Robust derived EoS proxy. Some EoS NPZ files store eta_lambda_max directly,
    # while others store lr and lambda_max separately.  Build a finite
    # eta_lambda_max whenever possible, so plotting does not silently go blank.
    eta_direct = safe_arr(aligned.get("eta_lambda_max", []))
    lam = safe_arr(aligned.get("lambda_max", []))
    lr = safe_arr(aligned.get("lr", []))

    if eta_direct.size != n:
        eta_direct = np.full(n, np.nan, dtype=np.float64)
    if lam.size != n:
        lam_fixed = np.full(n, np.nan, dtype=np.float64)
        lam_fixed[: min(n, lam.size)] = lam[: min(n, lam.size)]
        lam = lam_fixed
    if lr.size != n:
        lr_fixed = np.full(n, np.nan, dtype=np.float64)
        lr_fixed[: min(n, lr.size)] = lr[: min(n, lr.size)]
        lr = lr_fixed

    eta_from_lr_lam = lr * lam
    eta_combined = eta_direct.copy()
    missing = ~np.isfinite(eta_combined) & np.isfinite(eta_from_lr_lam)
    eta_combined[missing] = eta_from_lr_lam[missing]
    aligned["eta_lambda_max"] = eta_combined

    print(
        "[eos] aligned finite counts: "
        f"eta_lambda_max={np.isfinite(aligned.get('eta_lambda_max', [])).sum()}/{n}, "
        f"lambda_max={np.isfinite(aligned.get('lambda_max', [])).sum()}/{n}, "
        f"lr={np.isfinite(aligned.get('lr', [])).sum()}/{n}"
    )

    return aligned


def get_eta_lambda_max(eos_aligned: Dict[str, np.ndarray], n: int) -> np.ndarray:
    """
    Return eta*lambda_max robustly.

    Priority:
      1. aligned['eta_lambda_max'] if finite;
      2. aligned['lr'] * aligned['lambda_max'] where eta_lambda_max is missing.
    """
    eta = safe_arr(eos_aligned.get("eta_lambda_max", np.full(n, np.nan))) if eos_aligned else np.full(n, np.nan)
    if eta.size != n:
        out = np.full(n, np.nan, dtype=np.float64)
        out[: min(n, eta.size)] = eta[: min(n, eta.size)]
        eta = out

    lam = safe_arr(eos_aligned.get("lambda_max", np.full(n, np.nan))) if eos_aligned else np.full(n, np.nan)
    lr = safe_arr(eos_aligned.get("lr", np.full(n, np.nan))) if eos_aligned else np.full(n, np.nan)
    if lam.size != n:
        tmp = np.full(n, np.nan, dtype=np.float64)
        tmp[: min(n, lam.size)] = lam[: min(n, lam.size)]
        lam = tmp
    if lr.size != n:
        tmp = np.full(n, np.nan, dtype=np.float64)
        tmp[: min(n, lr.size)] = lr[: min(n, lr.size)]
        lr = tmp

    derived = lr * lam
    missing = ~np.isfinite(eta) & np.isfinite(derived)
    eta[missing] = derived[missing]
    return eta
