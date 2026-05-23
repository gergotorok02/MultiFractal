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


def fig03_tau_structure(entries, variant_name):
    """
    Early / middle / late multifractal spectrum figure.

    Styling:
      - fixed tau(q) y-limits: [-2.5, 1.5]
      - grey IQR bands in the first two columns
      - bootstrap 95% confidence band for the median f(alpha) profile
      - grey slightly darker
      - median curves black
      - main curves slightly wider
      - tau(q) insets in all rows of first column
      - insets use readable tick/title font sizes
      - red/blue grouping from tau(q) is reused in alpha(q) and f(alpha)
      - one horizontal legend at the bottom for the full figure
    """
    n = len(entries)
    idxs = [
        choose_index(n, CFG.early_frac),
        choose_index(n, CFG.mid_frac),
        choose_index(n, CFG.late_frac),
    ]

    row_titles = ["Early", "Middle", "Late"]
    col_titles = [
        r"$\widehat{\tau}(q)$",
        r"$\widehat{\alpha}(q)$",
        r"$\widehat{f}(\widehat{\alpha})$",
    ]

    BLACK = "#111111"
    BLUE = "#1f77b4"
    RED = "#d62728"
    GREY_FILL = "#9a9a9a"

    cache = []

    alpha_ylims: List[np.ndarray] = []
    falpha_xlims: List[np.ndarray] = []
    falpha_ylims: List[np.ndarray] = []

    for idx, row_title in zip(idxs, row_titles):
        d = collect_checkpoint_distributions(entries[idx], CFG.primary_family, variant_name)

        q = d["q"]
        tau = (
            np.stack(d["window_tau_vectors"], axis=0)
            if len(d["window_tau_vectors"])
            else np.empty((0, 0))
        )
        alpha = (
            np.stack(d["window_alpha_vectors"], axis=0)
            if len(d["window_alpha_vectors"])
            else np.empty((0, 0))
        )
        f = (
            np.stack(d["window_f_vectors"], axis=0)
            if len(d["window_f_vectors"])
            else np.empty((0, 0))
        )

        cache.append((idx, q, tau, alpha, f))

        epoch = int(entries[idx].get("epoch", idx))
        summarize_tau_curvature_sign_changes(
            q=q,
            tau=tau,
            label=f"{row_title.lower()} epoch {epoch}",
        )

        if alpha.size:
            med, q25, q75 = quartile_summary(alpha)
            alpha_ylims.extend([med, q25, q75])
            alpha_ylims.extend(list(alpha[:: max(1, len(alpha) // 50)]))

        if alpha.size and f.size:
            med_a, _, _ = quartile_summary(alpha)
            med_f, _, _ = quartile_summary(f)
            falpha_xlims.extend([med_a])
            falpha_ylims.extend([med_f])

            falpha_xlims.extend(list(alpha[:: max(1, len(alpha) // 50)]))
            falpha_ylims.extend(list(f[:: max(1, len(f) // 50)]))

    tau_lim = (-2.5, 1.5)
    alpha_lim = robust_limits(alpha_ylims, pad_frac=0.08) if alpha_ylims else (-1.0, 1.0)
    falpha_xlim = robust_limits(falpha_xlims, pad_frac=0.08) if falpha_xlims else (-1.0, 1.0)
    falpha_ylim = robust_limits(falpha_ylims, pad_frac=0.08) if falpha_ylims else (-1.0, 1.0)

    fig, axs = plt.subplots(
        3,
        3,
        figsize=(11.6, 9.6),
        sharex=False,
        constrained_layout=False,
    )

    for row, ((idx, q, tau, alpha, f), row_title) in enumerate(zip(cache, row_titles)):
        epoch = int(entries[idx].get("epoch", idx))

        curve_colors = tau_curve_color_list(q, tau) if (q.size and tau.size) else []

        # ============================================================
        # Column 1: tau-hat(q)
        # ============================================================
        ax = axs[row, 0]

        if row == 0:
            ax.set_title(col_titles[0], pad=8, fontsize=13)

        if q.size and tau.size:
            for curve, c in zip(tau, curve_colors):
                ax.plot(
                    q,
                    curve,
                    color=c,
                    alpha=0.17,
                    lw=1.02,
                    zorder=1,
                )

            med, q25, q75 = quartile_summary(tau)

            ax.fill_between(
                q,
                q25,
                q75,
                color=GREY_FILL,
                alpha=0.38,
                linewidth=0.0,
                edgecolor="none",
                zorder=2,
            )

            ax.plot(q, med, color=BLACK, lw=2.35, zorder=4)

            draw_zero_q(ax)
            ax.axhline(
                0.0,
                color="0.55",
                lw=0.85,
                ls=":",
                alpha=0.78,
                zorder=2,
            )
            ax.set_ylim(*tau_lim)

            add_tau_zoom_inset(ax, q, tau, med, curve_colors)

        ax.set_ylabel(f"{row_title}\n(epoch {epoch})")

        if row == 2:
            ax.set_xlabel(r"Moment order $q$")

        # ============================================================
        # Column 2: alpha-hat(q)
        # ============================================================
        ax = axs[row, 1]

        if row == 0:
            ax.set_title(col_titles[1], pad=8, fontsize=13)

        if q.size and alpha.size:
            if len(curve_colors) != len(alpha):
                curve_colors = [BLUE] * len(alpha)

            for curve, c in zip(alpha, curve_colors):
                ax.plot(
                    q,
                    curve,
                    color=c,
                    alpha=0.10,
                    lw=0.98,
                    zorder=1,
                )

            med, q25, q75 = quartile_summary(alpha)

            ax.fill_between(
                q,
                q25,
                q75,
                color=GREY_FILL,
                alpha=0.38,
                linewidth=0.0,
                edgecolor="none",
                zorder=2,
            )

            ax.plot(q, med, color=BLACK, lw=2.35, zorder=4)

            draw_zero_q(ax)
            ax.set_ylim(*alpha_lim)

        if row == 2:
            ax.set_xlabel(r"Moment order $q$")

        # ============================================================
        # Column 3: f-hat(alpha-hat)
        # ============================================================
        ax = axs[row, 2]

        if row == 0:
            ax.set_title(col_titles[2], pad=8, fontsize=13)

        if alpha.size and f.size:
            if len(curve_colors) != len(alpha):
                curve_colors = [BLUE] * len(alpha)

            for a_curve, f_curve, c in zip(alpha, f, curve_colors):
                mask = np.isfinite(a_curve) & np.isfinite(f_curve)
                if np.any(mask):
                    ax.plot(
                        a_curve[mask],
                        f_curve[mask],
                        color=c,
                        alpha=0.10,
                        lw=0.98,
                        zorder=1,
                    )

            # Median parametric f(alpha) curve only.
            # No confidence interval or filled band is shown in the last column.
            med_a, _, _ = quartile_summary(alpha)
            med_f, _, _ = quartile_summary(f)

            maskmed = np.isfinite(med_a) & np.isfinite(med_f)

            if np.sum(maskmed) >= 2:
                order = np.argsort(med_a[maskmed])
                ax.plot(
                    med_a[maskmed][order],
                    med_f[maskmed][order],
                    color=BLACK,
                    lw=2.35,
                    zorder=4,
                )

            ax.set_xlim(*falpha_xlim)
            ax.set_ylim(*falpha_ylim)

        if row == 2:
            ax.set_xlabel(r"Estimated Hölder exponent $\widehat{\alpha}$")

    # Leave room at the bottom for the figure-wide legend
    fig.subplots_adjust(bottom=0.13, left=0.08, right=0.98, top=0.95, wspace=0.18, hspace=0.22)

    # ================================================================
    # Figure-wide horizontal legend at bottom
    # ================================================================
    legend_handles = [
        Line2D([0], [0], color=RED, lw=1.8, label=r"Lower branch / corresponding curves"),
        Line2D([0], [0], color=BLUE, lw=1.8, label=r"Upper branch / corresponding curves"),
        Patch(facecolor=GREY_FILL, edgecolor="none", alpha=0.38, label=r"IQR band"),
        Line2D([0], [0], color=BLACK, lw=2.35, label=r"Median / average curve"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=2.6,
        handletextpad=0.6,
        columnspacing=1.8,
        borderaxespad=0.0,
    )

    save(fig, CFG.main_dir / "fig03_tau_structure_early_mid_late_spectrum.pdf")


def fig04_alpha_distributions(entries, variant_name):
    n = len(entries)
    idxs = [choose_index(n, CFG.early_frac), choose_index(n, CFG.mid_frac), choose_index(n, CFG.late_frac)]
    titles = ["Early", "Middle", "Late"]
    colors = ["tab:blue", "tab:orange", "tab:red"]

    samples = []
    for idx in idxs:
        d = collect_checkpoint_distributions(entries[idx], CFG.primary_family, variant_name)
        vals = np.concatenate(d["bank_alpha_dirs"]) if len(d["bank_alpha_dirs"]) else np.asarray([], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        samples.append(vals)
    xlo, xhi = robust_limits(samples, pad_frac=0.08)
    xs = np.linspace(xlo, xhi, 500)

    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.6), sharey=True, constrained_layout=True)
    y_max = 0.0
    for ax, vals, title, color in zip(axs, samples, titles, colors):
        if vals.size == 0:
            continue
        bins = np.linspace(xlo, xhi, 60)
        hist, edges = np.histogram(vals, bins=bins, density=True)
        mids = 0.5 * (edges[:-1] + edges[1:])
        smooth = moving_average_nan(hist, 5)
        curve = np.interp(xs, mids, smooth, left=np.nan, right=np.nan)
        curve = np.nan_to_num(curve, nan=0.0)
        ax.fill_between(xs, 0.0, curve, color=color, alpha=0.16)
        ax.plot(xs, curve, color=color, lw=1.8)
        ax.axvline(np.nanmedian(vals), color=color, ls="--", lw=0.9, alpha=0.8)
        y_max = max(y_max, float(np.nanmax(curve)))
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"Directional exponent $\widehat{\alpha}_i$")
    axs[0].set_ylabel("Density")
    for ax in axs:
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0.0, 1.08 * y_max if y_max > 0 else 1.0)
    panel_label(axs[0], "A")
    panel_label(axs[1], "B")
    panel_label(axs[2], "C")
    save(fig, CFG.main_dir / "fig04_alpha_distribution_evolution.pdf")


def fig05_signed_q_asymmetry(entries, variant_name):
    """
    Replacement Fig. 05:
    Signed-q localization of spectral evolution.

    This replaces the older negative-positive q asymmetry gap. Instead of
    comparing raw tau-levels or branch curvature, it asks a sharper question:

        Where in q-space does the tau-spectrum actually change during training?

    Panel A:
      Heatmap of Delta tau(q,t) relative to the first finite checkpoint.

    Panel B:
      Integrated absolute spectral displacement on the negative-q and
      positive-q branches.

    Panel C:
      Fraction of total spectral displacement carried by the negative-q branch.
    """
    x = maybe_epoch_x(entries)

    s = extract_signed_q_spectral_evolution(
        entries,
        CFG.primary_family,
        variant_name,
    )

    q = s["q"]
    delta_tau = s["delta_tau_med"]
    ref_index = int(s["ref_index"])

    if q.size == 0 or delta_tau.size == 0:
        print("[fig05] no finite tau spectra available")
        return

    ref_epoch = float(x[ref_index]) if ref_index < len(x) and np.isfinite(x[ref_index]) else float(ref_index)

    # Light smoothing for branch summaries only, not for the heatmap.
    cneg_med = moving_average_nan(s["cneg_median"], 3)
    cneg_q25 = moving_average_nan(s["cneg_q25"], 3)
    cneg_q75 = moving_average_nan(s["cneg_q75"], 3)

    cpos_med = moving_average_nan(s["cpos_median"], 3)
    cpos_q25 = moving_average_nan(s["cpos_q25"], 3)
    cpos_q75 = moving_average_nan(s["cpos_q75"], 3)

    rho_med = moving_average_nan(s["rho_neg_median"], 3)
    rho_q25 = moving_average_nan(s["rho_neg_q25"], 3)
    rho_q75 = moving_average_nan(s["rho_neg_q75"], 3)

    BLUE = "#1f77b4"
    RED = "#d62728"
    BLACK = "#111111"
    GREY_FILL = "#9a9a9a"

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(13.2, 3.9),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
    )

    # ============================================================
    # Panel A: Delta tau(q,t) heatmap
    # ============================================================
    ax = axs[0]

    vals = delta_tau[np.isfinite(delta_tau)]
    if vals.size:
        vmax = float(np.nanquantile(np.abs(vals), 0.98))
        vmax = max(vmax, 1e-6)
    else:
        vmax = 1.0

    # pcolormesh expects [q, epoch] shape, so transpose.
    im = ax.pcolormesh(
        x,
        q,
        delta_tau.T,
        shading="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.axhline(0.0, color="0.25", lw=0.9, ls="--", alpha=0.85)
    ax.axvline(ref_epoch, color="0.25", lw=0.8, ls=":", alpha=0.85)

    ax.set_title(r"Spectral displacement $\Delta\widehat{\tau}(q,t)$", pad=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Moment order $q$")

    ax.text(
        0.03,
        0.04,
        rf"reference: epoch {ref_epoch:g}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="0.20",
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="0.85",
            alpha=0.85,
        ),
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.018)
    cbar.set_label(r"$\widehat{\tau}(q,t)-\widehat{\tau}(q,t_0)$")

    panel_label(ax, "A")

    # ============================================================
    # Panel B: Branch-wise spectral displacement
    # ============================================================
    ax = axs[1]

    ax.fill_between(
        x,
        cneg_q25,
        cneg_q75,
        where=np.isfinite(cneg_q25) & np.isfinite(cneg_q75),
        color=BLUE,
        alpha=0.14,
        linewidth=0.0,
        zorder=1,
    )
    ax.plot(
        x,
        cneg_med,
        color=BLUE,
        lw=2.15,
        zorder=3,
    )

    ax.fill_between(
        x,
        cpos_q25,
        cpos_q75,
        where=np.isfinite(cpos_q25) & np.isfinite(cpos_q75),
        color=RED,
        alpha=0.14,
        linewidth=0.0,
        zorder=1,
    )
    ax.plot(
        x,
        cpos_med,
        color=RED,
        lw=2.15,
        zorder=3,
    )

    ax.set_title("Branch-wise spectral change", pad=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Mean absolute $\Delta\widehat{\tau}$")

    ax.text(
        0.04,
        0.94,
        r"━ $q<0$ branch",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=BLUE,
        fontsize=9.0,
    )
    ax.text(
        0.04,
        0.85,
        r"━ $q>0$ branch",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=RED,
        fontsize=9.0,
    )

    ax.text(
        0.98,
        0.04,
        r"$C_{\pm}(t)=|Q_{\pm}|^{-1}\int_{Q_{\pm}}|\Delta\widehat{\tau}(q,t)|\,dq$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="0.15",
    )

    panel_label(ax, "B")

    # ============================================================
    # Panel C: Negative-q share of spectral displacement
    # ============================================================
    ax = axs[2]

    ax.axhspan(0.5, 1.0, color=BLUE, alpha=0.035, zorder=0)
    ax.axhspan(0.0, 0.5, color=RED, alpha=0.035, zorder=0)

    ax.fill_between(
        x,
        rho_q25,
        rho_q75,
        where=np.isfinite(rho_q25) & np.isfinite(rho_q75),
        color=GREY_FILL,
        alpha=0.24,
        linewidth=0.0,
        zorder=1,
    )

    ax.plot(
        x,
        rho_med,
        color=BLACK,
        lw=2.25,
        zorder=3,
    )

    ax.axhline(
        0.5,
        color="0.45",
        lw=0.95,
        ls="--",
        zorder=2,
    )

    ax.set_ylim(0.0, 1.0)
    ax.set_title(r"Signed-$q$ localization of change", pad=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Negative-branch share $\rho_{-}$")

    ax.text(
        0.04,
        0.94,
        r"$q<0$ carries more change",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=BLUE,
        fontsize=8.7,
    )
    ax.text(
        0.04,
        0.06,
        r"$q>0$ carries more change",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=RED,
        fontsize=8.7,
    )

    ax.text(
        0.98,
        0.04,
        r"$\rho_{-}(t)=C_{-}(t)/(C_{-}(t)+C_{+}(t)+\varepsilon)$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="0.15",
    )

    panel_label(ax, "C")

    save(fig, CFG.main_dir / "fig05_negative_positive_q_asymmetry.pdf")
