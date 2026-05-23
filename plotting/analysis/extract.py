from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from plotting.io.protocols import get_per_batch
from plotting.utils import as_list_of_dicts, maybe_scalar, safe_arr


def infer_q_values(entry: Dict[str, Any], family_name: str) -> np.ndarray:
    per_batch = get_per_batch(entry, family_name)
    for b in per_batch:
        agg = b.get("aggregated", {})
        q = safe_arr(agg.get("q_list", []))
        if q.size > 0:
            return q
        for bank in as_list_of_dicts(b.get("banks", [])):
            q = safe_arr(bank.get("q_list", []))
            if q.size > 0:
                return q
    return np.asarray([], dtype=np.float64)


def collect_checkpoint_distributions(entry: Dict[str, Any], family_name: str, variant_name: str) -> Dict[str, Any]:
    per_batch = get_per_batch(entry, family_name)
    q_ref = infer_q_values(entry, family_name)

    tau_vectors: List[np.ndarray] = []
    alpha_vectors: List[np.ndarray] = []
    f_vectors: List[np.ndarray] = []
    window_tau_vectors: List[np.ndarray] = []
    window_alpha_vectors: List[np.ndarray] = []
    window_f_vectors: List[np.ndarray] = []
    bank_alpha_dirs: List[np.ndarray] = []
    bank_tau_nlin: List[float] = []
    tau_nonlinearity_vals: List[float] = []
    alpha_std_vals: List[float] = []
    f_width_vals: List[float] = []
    tau_neg_level_vals: List[float] = []
    tau_pos_level_vals: List[float] = []

    for batch_obj in per_batch:
        agg = batch_obj.get("aggregated", {})
        variant_agg = agg.get("variant_aggregates", {}).get(variant_name, None)
        if variant_agg is not None and variant_agg.get("valid", False):
            pt = variant_agg.get("partition_tau", {})
            cj = variant_agg.get("chhabra_jensen", {})
            ad = variant_agg.get("alpha_distribution_summary", {})
            tau = safe_arr(pt.get("tau_q_mean", []))
            alpha_q = safe_arr(cj.get("alpha_q_mean", []))
            f_q = safe_arr(cj.get("f_q_mean", []))
            if tau.size > 0:
                tau_vectors.append(tau)
            if alpha_q.size > 0:
                alpha_vectors.append(alpha_q)
            if f_q.size > 0:
                f_vectors.append(f_q)
            tau_nonlinearity_vals.append(maybe_scalar(pt.get("tau_nonlinearity_mean", np.nan)))
            alpha_std_vals.append(maybe_scalar(ad.get("alpha_std_mean", np.nan)))
            f_width_vals.append(maybe_scalar(cj.get("f_width_mean", np.nan)))

            if q_ref.size and tau.size == q_ref.size:
                neg = q_ref < 0
                pos = q_ref > 0
                tau_neg_level_vals.append(np.nanmean(tau[neg]) if np.any(neg) else np.nan)
                tau_pos_level_vals.append(np.nanmean(tau[pos]) if np.any(pos) else np.nan)
            else:
                tau_neg_level_vals.append(np.nan)
                tau_pos_level_vals.append(np.nan)

        for bank in as_list_of_dicts(batch_obj.get("banks", [])):
            variant_bank = bank.get("variants", {}).get(variant_name, None)
            if variant_bank is None or not variant_bank.get("valid", False):
                continue
            bagg = variant_bank.get("aggregated", {})
            bpt = bagg.get("partition_tau", {})
            best = bagg.get("best_window", {})
            best_ad = best.get("alpha_distribution", {})
            bank_tau_nlin.append(maybe_scalar(bpt.get("tau_nonlinearity_mean", np.nan)))
            alpha_dir = safe_arr(best_ad.get("alpha_dir", []))
            if alpha_dir.size > 0:
                bank_alpha_dirs.append(alpha_dir)
            for w in as_list_of_dicts(bagg.get("all_windows", [])):
                wpt = w.get("partition_tau", {})
                wcj = w.get("chhabra_jensen", {})
                tau_w = safe_arr(wpt.get("tau_q", []))
                alpha_w = safe_arr(wcj.get("alpha_q", []))
                f_w = safe_arr(wcj.get("f_q", []))
                if tau_w.size > 0:
                    window_tau_vectors.append(tau_w)
                if alpha_w.size > 0:
                    window_alpha_vectors.append(alpha_w)
                if f_w.size > 0:
                    window_f_vectors.append(f_w)

    return {
        "q": q_ref,
        "tau_vectors": tau_vectors,
        "alpha_vectors": alpha_vectors,
        "f_vectors": f_vectors,
        "window_tau_vectors": window_tau_vectors,
        "window_alpha_vectors": window_alpha_vectors,
        "window_f_vectors": window_f_vectors,
        "bank_alpha_dirs": bank_alpha_dirs,
        "bank_tau_nlin": np.asarray(bank_tau_nlin, dtype=np.float64),
        "tau_nonlinearity_vals": np.asarray(tau_nonlinearity_vals, dtype=np.float64),
        "alpha_std_vals": np.asarray(alpha_std_vals, dtype=np.float64),
        "f_width_vals": np.asarray(f_width_vals, dtype=np.float64),
        "tau_neg_level_vals": np.asarray(tau_neg_level_vals, dtype=np.float64),
        "tau_pos_level_vals": np.asarray(tau_pos_level_vals, dtype=np.float64),
    }


def extract_distribution_time_series(entries: List[Dict[str, Any]], family_name: str, variant_name: str) -> Dict[str, np.ndarray]:
    out = {k: [] for k in [
        "tau_nonlinearity_median", "tau_nonlinearity_q25", "tau_nonlinearity_q75",
        "alpha_std_median", "alpha_std_q25", "alpha_std_q75",
        "f_width_median", "f_width_q25", "f_width_q75",
        "tau_neg_level_median", "tau_neg_level_q25", "tau_neg_level_q75",
        "tau_pos_level_median", "tau_pos_level_q25", "tau_pos_level_q75",
    ]}

    for e in entries:
        d = collect_checkpoint_distributions(e, family_name, variant_name)
        mapping = {
            "tau_nonlinearity": d["tau_nonlinearity_vals"],
            "alpha_std": d["alpha_std_vals"],
            "f_width": d["f_width_vals"],
            "tau_neg_level": d["tau_neg_level_vals"],
            "tau_pos_level": d["tau_pos_level_vals"],
        }
        for base, vals in mapping.items():
            vals = np.asarray(vals, dtype=np.float64)
            if vals.size == 0 or not np.isfinite(vals).any():
                med = q25 = q75 = np.nan
            else:
                med = float(np.nanmedian(vals))
                q25 = float(np.nanquantile(vals, 0.25))
                q75 = float(np.nanquantile(vals, 0.75))
            out[f"{base}_median"].append(med)
            out[f"{base}_q25"].append(q25)
            out[f"{base}_q75"].append(q75)

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def extract_heatmap_time_series(entries: List[Dict[str, Any]], family_name: str, variant_name: str, field: str):
    rows = []
    q_ref = None
    for e in entries:
        d = collect_checkpoint_distributions(e, family_name, variant_name)
        if q_ref is None and d["q"].size > 0:
            q_ref = d["q"]
        key = {
            "tau": "window_tau_vectors",
            "alpha": "window_alpha_vectors",
            "f": "window_f_vectors",
        }[field]
        vecs = d[key]
        if len(vecs) == 0:
            rows.append(np.full_like(q_ref if q_ref is not None else np.asarray([np.nan]), np.nan))
            continue
        mat = np.stack(vecs, axis=0)
        rows.append(np.nanmedian(mat, axis=0))
    if q_ref is None or len(rows) == 0:
        return np.asarray([], dtype=np.float64), np.empty((0, 0))
    rows = [r if len(r) == len(q_ref) else np.full_like(q_ref, np.nan) for r in rows]
    return q_ref, np.stack(rows, axis=0)
