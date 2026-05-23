from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from plotting.config import CFG
from plotting.analysis.spectrum import _tau_zoom_ylim


CMAP_TIME = plt.cm.plasma
CMAP_DIVERGING = plt.cm.RdBu_r
CMAP_SEQ = plt.cm.viridis
CI_RED = "#f2a6a6"
TAU_POS_GREEN = "#1b9e77"
TAU_NEG_BLUE = "#377eb8"
ALPHA_LINE = "#238b45"
F_LINE = "#7b3294"


def setup_style():
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "axes.labelsize": 13.0,
        "axes.titlesize": 13.0,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "legend.fontsize": 11.0,
        "axes.grid": True,
        "grid.alpha": 0.14,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def balanced_epoch_unit(x: np.ndarray) -> np.ndarray:
    """
    Map epoch values to [0,1] with approximately equal color-space allocation
    to [min,5], [5,10], [10,30], [30,max].
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float64)

    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    b0 = xmin
    b1, b2, b3 = CFG.balanced_epoch_breaks
    b4 = xmax
    # make sure boundaries are monotone for unusual runs
    bounds = np.asarray([b0, b1, b2, b3, b4], dtype=np.float64)
    bounds = np.maximum.accumulate(bounds)
    if bounds[-1] <= bounds[0]:
        return np.zeros_like(x, dtype=np.float64)

    levels = np.asarray(CFG.balanced_color_levels, dtype=np.float64)
    for i, val in enumerate(x):
        if not np.isfinite(val):
            continue
        if val <= bounds[0]:
            out[i] = levels[0]
        elif val >= bounds[-1]:
            out[i] = levels[-1]
        else:
            j = int(np.searchsorted(bounds, val, side="right") - 1)
            j = max(0, min(j, len(bounds) - 2))
            lo, hi = bounds[j], bounds[j + 1]
            t = 0.0 if hi <= lo else (val - lo) / (hi - lo)
            out[i] = levels[j] + t * (levels[j + 1] - levels[j])
    return np.nan_to_num(out, nan=0.0)


def balanced_epoch_colors(x: np.ndarray) -> np.ndarray:
    return CMAP_TIME(np.clip(balanced_epoch_unit(x), 0.0, 1.0))


def add_balanced_epoch_colorbar(fig: plt.Figure, axs, x_epoch: np.ndarray):
    finite = x_epoch[np.isfinite(x_epoch)]
    if finite.size == 0:
        return None
    xmin = float(np.nanmin(finite))
    xmax = float(np.nanmax(finite))
    ticks_epoch = [xmin, 5.0, 10.0, 30.0, xmax]
    ticks_epoch = [t for t in ticks_epoch if xmin <= t <= xmax]
    ticks_unit = balanced_epoch_unit(np.asarray(ticks_epoch, dtype=np.float64))
    sm = plt.cm.ScalarMappable(cmap=CMAP_TIME, norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
    cbar = fig.colorbar(sm, ax=axs, fraction=0.028, pad=0.02)
    cbar.set_ticks(ticks_unit)
    cbar.set_ticklabels([f"{t:g}" for t in ticks_epoch])
    cbar.set_label("Epoch")
    return cbar


def save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def line_with_band(ax, x, med, q25, q75, color=None, band_color=None, label=None, lw=2.2, alpha_fill=0.18, zorder=3):
    if band_color is None:
        band_color = color
    ax.plot(x, med, color=color, lw=lw, label=label, zorder=zorder)
    ax.fill_between(x, q25, q75, color=band_color, alpha=alpha_fill, linewidth=0, zorder=zorder - 1)


def panel_label(ax, text: str):
    ax.text(-0.10, 1.04, text, transform=ax.transAxes, va="bottom", ha="left", fontsize=13, fontweight="bold")


def draw_zero_q(ax):
    ax.axvline(0.0, color="0.3", lw=1.0, ls="--", alpha=0.8)


def add_tau_zoom_inset(
    ax,
    q: np.ndarray,
    tau: np.ndarray,
    med: np.ndarray,
    curve_colors: List[str],
):
    """
    Small bottom-right inset for tau(q), zooming into q in [0, 2].

    The inset uses the same tick/title font size as the main plot ticks, so it
    remains readable after export or when embedded in the thesis PDF.
    """
    BLACK = "#111111"

    q = np.asarray(q, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    if q.size == 0 or tau.size == 0:
        return

    zoom_mask = np.isfinite(q) & (q >= 0.0) & (q <= 2.0)
    if zoom_mask.sum() < 2:
        return

    ylims = _tau_zoom_ylim(q, tau, qmin=0.0, qmax=2.0)
    if ylims is None:
        return

    axins = ax.inset_axes([0.59, 0.11, 0.34, 0.29])

    # Original thinner widths in inset
    for curve, c in zip(tau, curve_colors):
        axins.plot(q, curve, color=c, alpha=0.16, lw=0.72, zorder=1)

    axins.plot(q, med, color=BLACK, lw=1.35, zorder=3)

    axins.set_xlim(0.0, 2.0)
    axins.set_ylim(*ylims)
    axins.set_xticks([0.0, 1.0, 2.0])
    main_tick_size = mpl.rcParams.get("xtick.labelsize", 9)
    if isinstance(main_tick_size, str):
        main_tick_size = mpl.rcParams.get("font.size", 9.5)

    axins.tick_params(axis="both", labelsize=main_tick_size, length=2.4, pad=1.4)

    for spine in axins.spines.values():
        spine.set_linewidth(0.65)
        spine.set_alpha(0.75)

    axins.grid(True, alpha=0.12, linewidth=0.4)
    axins.set_title(r"$q\in[0,2]$", fontsize=main_tick_size, pad=1.8)
