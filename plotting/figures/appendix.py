from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plotting.config import CFG
from plotting.analysis.extract import *
from plotting.analysis.spectrum import *
from plotting.io.protocols import *
from plotting.style.base import *
from plotting.utils import *


def figA1_variant_comparison(entries, variants):
    x = maybe_epoch_x(entries)
    chosen = [v for v in CFG.variant_priority if v in variants]
    if len(chosen) == 0:
        chosen = variants[: min(4, len(variants))]
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.0), sharex=True)
    palette = plt.cm.tab10(np.linspace(0, 1, len(chosen)))
    for c, v in zip(palette, chosen):
        s = extract_distribution_time_series(entries, CFG.primary_family, v)
        axs[0].plot(x, s["tau_nonlinearity_median"], lw=2.0, color=c, label=v)
        axs[1].plot(x, s["alpha_std_median"], lw=2.0, color=c, label=v)
    axs[0].set_title(r"$\mathcal{M}_{\tau}$ by variant")
    axs[1].set_title(r"$\mathrm{std}(\widehat{\alpha}_i)$ by variant")
    axs[0].set_ylabel("Value")
    for ax in axs:
        ax.set_xlabel("Epoch")
    axs[0].legend(frameon=False)
    save(fig, CFG.appendix_dir / "figA1_variant_comparison.pdf")


def figA2_batch_robustness(entries, variant_name):
    x = maybe_epoch_x(entries)
    batch_tau: Dict[int, List[float]] = {}
    batch_alpha: Dict[int, List[float]] = {}
    for e in entries:
        for b in get_per_batch(e, CFG.primary_family):
            bi = int(b.get("batch_index", 0))
            agg = b.get("aggregated", {})
            v = agg.get("variant_aggregates", {}).get(variant_name, None)
            if v is None or not v.get("valid", False):
                continue
            batch_tau.setdefault(bi, []).append(maybe_scalar(v.get("partition_tau", {}).get("tau_nonlinearity_mean", np.nan)))
            batch_alpha.setdefault(bi, []).append(maybe_scalar(v.get("alpha_distribution_summary", {}).get("alpha_std_mean", np.nan)))
    if len(batch_tau) == 0:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.9), sharex=True)
    palette = plt.cm.Set2(np.linspace(0, 1, len(batch_tau)))
    for c, bi in zip(palette, sorted(batch_tau.keys())):
        axs[0].plot(x[:len(batch_tau[bi])], batch_tau[bi], lw=1.8, color=c, label=f"batch {bi}")
        axs[1].plot(x[:len(batch_alpha[bi])], batch_alpha[bi], lw=1.8, color=c, label=f"batch {bi}")
    axs[0].set_title(r"$\mathcal{M}_{\tau}$ across probe batches")
    axs[1].set_title(r"$\mathrm{std}(\widehat{\alpha}_i)$ across probe batches")
    axs[0].set_ylabel("Value")
    for ax in axs:
        ax.set_xlabel("Epoch")
    axs[0].legend(frameon=False)
    save(fig, CFG.appendix_dir / "figA2_batch_robustness.pdf")


def figA3_bank_spaghetti(entries, variant_name):
    x = maybe_epoch_x(entries)
    bank_vals_by_checkpoint = []
    for e in entries:
        d = collect_checkpoint_distributions(e, CFG.primary_family, variant_name)
        bank_vals_by_checkpoint.append(d["bank_tau_nlin"])
    max_banks = max((len(v) for v in bank_vals_by_checkpoint), default=0)
    if max_banks == 0:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for bi in range(max_banks):
        ys = [vals[bi] if bi < len(vals) else np.nan for vals in bank_vals_by_checkpoint]
        ax.plot(x, ys, lw=1.4, alpha=0.55)
    s = extract_distribution_time_series(entries, CFG.primary_family, variant_name)
    ax.plot(x, s["tau_nonlinearity_median"], color="black", lw=2.8, label="median")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathcal{M}_{\tau}$")
    ax.set_title("Bank-level trajectories")
    ax.legend(frameon=False)
    save(fig, CFG.appendix_dir / "figA3_bank_spaghetti.pdf")


def figA4_selected_times_tau_f_alpha(entries, variant_name, fracs=CFG.appendix_fracs):
    n = len(entries)
    idxs = []
    for frac in fracs:
        idx = choose_index(n, frac)
        if idx not in idxs:
            idxs.append(idx)
    if len(idxs) == 0:
        return

    fig, axs = plt.subplots(len(idxs), 3, figsize=(12.0, 2.5 * len(idxs)), sharex="col")
    if len(idxs) == 1:
        axs = np.asarray([axs])

    tau_ylims = []
    alpha_ylims = []
    f_ylims = []
    cache = []
    for idx in idxs:
        d = collect_checkpoint_distributions(entries[idx], CFG.primary_family, variant_name)
        q = d["q"]
        tau = np.stack(d["window_tau_vectors"], axis=0) if len(d["window_tau_vectors"]) else np.empty((0, 0))
        alpha = np.stack(d["window_alpha_vectors"], axis=0) if len(d["window_alpha_vectors"]) else np.empty((0, 0))
        f = np.stack(d["window_f_vectors"], axis=0) if len(d["window_f_vectors"]) else np.empty((0, 0))
        cache.append((idx, q, tau, alpha, f))
        if tau.size:
            tau_ylims.extend(list(quantile_summary(tau)))
        if alpha.size:
            alpha_ylims.extend(list(quantile_summary(alpha)))
        if f.size:
            f_ylims.extend(list(quantile_summary(f)))

    tau_lim = robust_limits(tau_ylims) if tau_ylims else (-1, 1)
    alpha_lim = robust_limits(alpha_ylims) if alpha_ylims else (-1, 1)
    f_lim = robust_limits(f_ylims) if f_ylims else (-1, 1)

    for row, (idx, q, tau, alpha, f) in enumerate(cache):
        epoch = int(entries[idx].get("epoch", idx))
        title_prefix = f"epoch {epoch}"
        for col, (mat, color, ylabel, ylim) in enumerate([
            (tau, "0.12", r"$\widehat{\tau}(q)$", tau_lim),
            (alpha, ALPHA_LINE, r"$\widehat{\alpha}(q)$", alpha_lim),
            (f, F_LINE, r"$\widehat{f}(q)$", f_lim),
        ]):
            ax = axs[row, col]
            if mat.size and q.size:
                med, q25, q75 = quantile_summary(mat)
                line_with_band(ax, q, med, q25, q75, color=color, band_color=CI_RED, alpha_fill=0.20)
                draw_zero_q(ax)
                ax.set_ylim(*ylim)
            if row == 0:
                ax.set_title([r"$\widehat{\tau}(q)$", r"$\widehat{\alpha}(q)$", r"$\widehat{f}(q)$"][col])
            if row == len(cache) - 1:
                ax.set_xlabel(r"Moment order $q$")
            if col == 0:
                ax.set_ylabel(f"{title_prefix}\n{ylabel}")
    save(fig, CFG.appendix_dir / "figA4_selected_times_tau_alpha_f.pdf")


def figA5_eos_vs_mf_scatter(entries, variant_name, eos_aligned):
    if len(eos_aligned) == 0:
        return
    x_epoch = maybe_epoch_x(entries)
    color_values = balanced_epoch_unit(x_epoch)
    s = extract_distribution_time_series(entries, CFG.primary_family, variant_name)
    pairs = [
        (s["tau_nonlinearity_median"], get_eta_lambda_max(eos_aligned, len(entries)), r"$\mathcal{M}_{\tau}$", r"$\eta\lambda_{\max}$"),
        (s["alpha_std_median"], safe_arr(eos_aligned.get("eta_uHu", np.full(len(entries), np.nan))), r"$\mathrm{std}(\widehat{\alpha}_i)$", r"$\eta u^\top H u$"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.0))
    for ax, (x, y, xlabel, ylabel) in zip(axs, pairs):
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], c=color_values[mask], cmap=CMAP_TIME, vmin=0.0, vmax=1.0, s=52, edgecolor="white", linewidth=0.6)
        fit = linear_fit_with_band(x, y)
        if fit is not None:
            xg, yg, lo, hi, *_ = fit
            ax.plot(xg, yg, color="black", lw=1.9)
            ax.fill_between(xg, lo, hi, color="black", alpha=0.10)
        r = nan_corr(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(fr"$r={r:.3f}$" if np.isfinite(r) else r"$r=\mathrm{nan}$")
    save(fig, CFG.appendix_dir / "figA5_eos_vs_mf_scatter.pdf")


def figA6_metadata(entries, eos_path, variant_name, train_aligned):
    fig = plt.figure(figsize=(8.0, 3.8))
    acc = test_acc_series(entries)
    train_acc = train_acc_series(train_aligned, len(entries))
    text = [
        "Experiment metadata",
        f"checkpoints = {len(entries)}",
        f"protocol = {CFG.primary_protocol}",
        f"family = {CFG.primary_family}",
        f"primary_variant = {variant_name}",
        f"best_train_acc = {np.nanmax(train_acc):.4f}" if np.isfinite(train_acc).any() else "best_train_acc = nan",
        f"best_test_acc = {np.nanmax(acc):.4f}" if np.isfinite(acc).any() else "best_test_acc = nan",
        f"eos_path = {eos_path if eos_path is not None else 'not available'}",
        f"train_acc_npz = {CFG.train_acc_npz_path}",
        "epoch color scale = balanced [min,5],[5,10],[10,30],[30,max]",
    ]
    fig.text(0.04, 0.92, "\n".join(text), va="top", ha="left", family="monospace", fontsize=12)
    save(fig, CFG.appendix_dir / "figA6_metadata_panel.pdf")
