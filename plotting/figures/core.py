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


def fig01a_core_claim_mf_eos_test(entries, variant_name, eos_aligned, train_aligned):
    """
    Fig. 1a:
      A: x = M_tau, y = eta*lambda_max
      B: x = M_tau, y = train accuracy

    Both panels use multifractal complexity as horizontal axis.
    """
    n = len(entries)

    x_epoch = maybe_epoch_x(entries)
    color_values = balanced_epoch_unit(x_epoch)

    s = extract_distribution_time_series(entries, CFG.primary_family, variant_name)
    mtau = safe_arr(s["tau_nonlinearity_median"])

    eta_lambda = get_eta_lambda_max(eos_aligned, n)

    # THIS is the important part:
    # use the same aligned training-accuracy source as fig01b
    train_acc = safe_arr(train_acc_series(train_aligned, n))

    cmap = plt.cm.viridis

    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.1), constrained_layout=True)

    specs = [
        (
            axs[0],
            mtau,
            eta_lambda,
            r"Multifractal complexity $\mathcal{M}_{\tau}$",
            r"EoS proxy $\eta\lambda_{\max}$",
        ),
        (
            axs[1],
            mtau,
            train_acc,
            r"Multifractal complexity $\mathcal{M}_{\tau}$",
            "Train accuracy",
        ),
    ]

    for ax, x, y, xlabel, ylabel in specs:
        x = safe_arr(x)
        y = safe_arr(y)
        mask = np.isfinite(x) & np.isfinite(y)

        ax.scatter(
            x[mask],
            y[mask],
            c=color_values[mask],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=54,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )

        fit = linear_fit_with_band(x, y)
        if fit is not None:
            xg, yg, lo, hi, *_ = fit
            ax.fill_between(xg, lo, hi, color="0.2", alpha=0.10, zorder=1)
            ax.plot(xg, yg, color="0.2", lw=1.65, zorder=2)

        r = nan_corr(x, y)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(
            0.98,
            0.04,
            fr"$r={r:.3f}$" if np.isfinite(r) else r"$r=\mathrm{{nan}}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="0.85",
                alpha=0.92,
            ),
        )

    panel_label(axs[0], "A")
    panel_label(axs[1], "B")

    finite = x_epoch[np.isfinite(x_epoch)]
    if finite.size > 0:
        xmin = float(np.nanmin(finite))
        xmax = float(np.nanmax(finite))

        ticks_epoch = [xmin, 5.0, 10.0, 30.0, xmax]
        ticks_epoch = [t for t in ticks_epoch if xmin <= t <= xmax]
        ticks_unit = balanced_epoch_unit(np.asarray(ticks_epoch, dtype=np.float64))

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0),
        )
        cbar = fig.colorbar(sm, ax=axs, fraction=0.028, pad=0.02)
        cbar.set_ticks(ticks_unit)
        cbar.set_ticklabels([f"{t:g}" for t in ticks_epoch])
        cbar.set_label("Epoch")

    save(fig, CFG.main_dir / "fig01a_core_claim_mf_eos_train_accuracy.pdf")


def fig01b_core_claim_train_test(entries, variant_name, train_aligned):
    """
    Optional Fig. 1b-style figure:
      A: x = M_tau, y = train accuracy
      B: x = M_tau, y = test accuracy
    """
    x_epoch = maybe_epoch_x(entries)
    color_values = balanced_epoch_unit(x_epoch)

    s = extract_distribution_time_series(entries, CFG.primary_family, variant_name)
    mtau = s["tau_nonlinearity_median"]
    train_acc = train_acc_series(train_aligned, len(entries))
    test_acc = test_acc_series(entries)

    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.1), constrained_layout=True)
    specs = [
        (axs[0], mtau, train_acc, r"Multifractal complexity $\mathcal{M}_{\tau}$", "Train accuracy"),
        (axs[1], mtau, test_acc, r"Multifractal complexity $\mathcal{M}_{\tau}$", "Test accuracy"),
    ]

    for ax, x, y, xlabel, ylabel in specs:
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], c=color_values[mask], cmap=CMAP_TIME, vmin=0.0, vmax=1.0,
                   s=54, edgecolor="white", linewidth=0.55, zorder=3)
        fit = linear_fit_with_band(x, y)
        if fit is not None:
            xg, yg, lo, hi, *_ = fit
            ax.fill_between(xg, lo, hi, color="0.2", alpha=0.10, zorder=1)
            ax.plot(xg, yg, color="0.2", lw=1.65, zorder=2)
        r = nan_corr(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(
            0.98, 0.04,
            fr"$r={r:.3f}$" if np.isfinite(r) else r"$r=\mathrm{nan}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.85", alpha=0.92),
        )

    panel_label(axs[0], "A")
    panel_label(axs[1], "B")
    add_balanced_epoch_colorbar(fig, axs, x_epoch)
    save(fig, CFG.main_dir / "fig01b_core_claim_mf_train_vs_test_accuracy.pdf")


def fig02_temporal_dynamics(entries, variant_name, eos_aligned, train_aligned):
    x = maybe_epoch_x(entries)
    s = extract_distribution_time_series(entries, CFG.primary_family, variant_name)

    train_acc = train_acc_series(train_aligned, len(entries))
    eta_lambda = get_eta_lambda_max(eos_aligned, len(entries))

    fig, axs = plt.subplots(
        2, 1,
        figsize=(8.8, 5.8),
        sharex=True,
        constrained_layout=True,
    )

    # ------------------------------------------------------------------
    # Panel A: tau nonlinearity + train accuracy
    # ------------------------------------------------------------------
    line_with_band(
        axs[0],
        x,
        s["tau_nonlinearity_median"],
        s["tau_nonlinearity_q25"],
        s["tau_nonlinearity_q75"],
        color="tab:blue",
        band_color="tab:blue",
        alpha_fill=0.14,
        lw=1.8,
    )

    ax0b = axs[0].twinx()

    if np.isfinite(train_acc).any():
        ax0b.plot(
            x,
            train_acc,
            color="0.15",
            lw=1.7,
            ls=":",
        )

    # Axis labels with compact line-style indicators.
    # Solid curve: ━
    # Dotted curve: ⋯
    axs[0].set_ylabel(
        r"━ $\mathcal{M}_{\tau}$",
        color="tab:blue",
        labelpad=8,
    )

    ax0b.set_ylabel(
        r"⋯ Accuracy",
        color="0.15",
        labelpad=8,
    )

    axs[0].tick_params(axis="y", colors="tab:blue")
    ax0b.tick_params(axis="y", colors="0.15")

    axs[0].set_xlabel("Epoch")
    axs[0].tick_params(axis="x", labelbottom=True)

    # ------------------------------------------------------------------
    # Panel B: alpha std + eta lambda
    # ------------------------------------------------------------------
    line_with_band(
        axs[1],
        x,
        s["alpha_std_median"],
        s["alpha_std_q25"],
        s["alpha_std_q75"],
        color="tab:green",
        band_color="tab:green",
        alpha_fill=0.14,
        lw=1.8,
    )

    ax1b = axs[1].twinx()

    if np.isfinite(eta_lambda).any():
        ax1b.plot(
            x,
            eta_lambda,
            color="tab:red",
            lw=1.5,
            ls="-.",
        )
        ax1b.axhline(
            2.0,
            color="tab:red",
            lw=0.9,
            ls=":",
            alpha=0.7,
        )

    axs[1].set_ylabel(
        r"━ $\mathrm{std}(\widehat{\alpha}_i)$",
        color="tab:green",
        labelpad=8,
    )

    ax1b.set_ylabel(
        r"−·− $\eta\lambda_{\max}$",
        color="tab:red",
        labelpad=8,
    )

    axs[1].tick_params(axis="y", colors="tab:green")
    ax1b.tick_params(axis="y", colors="tab:red")

    axs[1].set_xlabel("Epoch")

    panel_label(axs[0], "A")
    panel_label(axs[1], "B")

    save(fig, CFG.main_dir / "fig02_temporal_dynamics.pdf")
