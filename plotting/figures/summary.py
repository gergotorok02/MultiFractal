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


def fig06_triptych(entries, variant_name):
    x = maybe_epoch_x(entries)
    q_tau, tau_med = extract_heatmap_time_series(entries, CFG.primary_family, variant_name, "tau")
    q_alpha, alpha_med = extract_heatmap_time_series(entries, CFG.primary_family, variant_name, "alpha")
    q_f, f_med = extract_heatmap_time_series(entries, CFG.primary_family, variant_name, "f")
    if tau_med.size == 0:
        return

    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True, constrained_layout=True)
    mats = [tau_med, alpha_med, f_med]
    qvals = [q_tau, q_alpha, q_f]
    titles = [r"$\widehat{\tau}(q,t)$", r"$\widehat{\alpha}(q,t)$", r"$\widehat{f}(q,t)$"]
    cmaps = [CMAP_DIVERGING, CMAP_SEQ, CMAP_SEQ]

    for i, (ax, mat, q, title, cmap) in enumerate(zip(axs, mats, qvals, titles, cmaps)):
        if mat.size == 0 or q.size == 0:
            continue
        vals = mat[np.isfinite(mat)]
        if vals.size == 0:
            continue
        if i == 0:
            vmax = float(np.nanquantile(np.abs(vals), 0.98))
            norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            im = ax.imshow(mat.T, aspect="auto", origin="lower", interpolation="nearest", extent=[x[0], x[-1], q[0], q[-1]], cmap=cmap, norm=norm)
        else:
            vmin = float(np.nanquantile(vals, 0.02))
            vmax = float(np.nanquantile(vals, 0.98))
            im = ax.imshow(mat.T, aspect="auto", origin="lower", interpolation="nearest", extent=[x[0], x[-1], q[0], q[-1]], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, pad=6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"Moment order $q$")
        draw_zero_q(ax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=8)
    panel_label(axs[0], "A")
    panel_label(axs[1], "B")
    panel_label(axs[2], "C")
    save(fig, CFG.main_dir / "fig06_tau_alpha_f_triptych.pdf")


def fig07_phase_diagram(entries, variant_name, eos_aligned):
    if len(eos_aligned) == 0:
        return
    x = maybe_epoch_x(entries)
    s = extract_distribution_time_series(entries, CFG.primary_family, variant_name)
    mtau = s["tau_nonlinearity_median"]
    eta_lambda = get_eta_lambda_max(eos_aligned, len(entries))
    fig, ax = plt.subplots(figsize=(8.0, 3.9), constrained_layout=True)
    sc = ax.scatter(x, mtau, c=eta_lambda, cmap=CMAP_SEQ, s=58, edgecolor="white", linewidth=0.5, zorder=3)
    ax.plot(x, mtau, color="0.25", lw=1.3, alpha=0.7, zorder=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Multifractal complexity $\mathcal{M}_{\tau}$")
    ax.set_title("Phase portrait of training dynamics", pad=6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\eta\lambda_{\max}$")
    panel_label(ax, "A")
    save(fig, CFG.main_dir / "fig07_phase_diagram_mf_eos_epoch.pdf")


def fig08_train_test_accuracy(entries, train_aligned):
    x = maybe_epoch_x(entries)
    train_acc = train_acc_series(train_aligned, len(entries))
    test_acc = test_acc_series(entries)

    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
    if np.isfinite(train_acc).any():
        ax.plot(x, train_acc, color="tab:blue", lw=2.0, label="Train accuracy")
    if np.isfinite(test_acc).any():
        ax.plot(x, test_acc, color="tab:orange", lw=2.0, label="Test accuracy")
    if np.isfinite(train_acc).any() and np.isfinite(test_acc).any():
        gap = train_acc - test_acc
        ax.fill_between(x, test_acc, train_acc, color="0.3", alpha=0.10, linewidth=0)
        valid_gap = gap[np.isfinite(gap)]
        final_gap = valid_gap[-1] if valid_gap.size else np.nan
        ax.text(
            0.98, 0.04,
            fr"final gap = {final_gap:.3f}" if np.isfinite(final_gap) else "final gap = nan",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.85", alpha=0.92),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train-test accuracy dynamics", pad=6)
    ax.legend(frameon=False)
    panel_label(ax, "A")
    save(fig, CFG.main_dir / "fig08_train_test_accuracy.pdf")
