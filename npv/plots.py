"""Figures: idea-discovery curves and rank-probability curves.

Fixed categorical palette (assigned by condition order, never cycled past 8:
extra conditions fold into gray and a warning is logged)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("npv")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def _color(i: int) -> str:
    if i >= len(SERIES):
        log.warning("more than %d conditions: extra series drawn in gray", len(SERIES))
        return "#898781"
    return SERIES[i]


def plot_discovery(curves: dict[str, dict], out_path: Path, empirical: dict[str, "np.ndarray"] | None = None):
    """curves[label] = output of estimators.discovery_curves; empirical[label] = accumulation array."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for i, (label, cv) in enumerate(curves.items()):
        col = _color(i)
        m, rare, model, n = cv["m_grid"], cv["rarefaction"], cv["model"], cv["n"]
        inside = m <= n
        ax.plot(m[inside], rare[inside], color=col, lw=2, label=f"{label} (rarefaction, m<=n)")
        ax.plot(m[~inside], model[~inside], color=col, lw=2, ls="--",
                label=f"{label} (model E[K_m], extrapolated)")
        ax.plot(m[inside], model[inside], color=col, lw=1, ls=":", alpha=0.7)
        ax.scatter([n], [cv["S_obs"]], color=col, s=36, zorder=5, edgecolor="white", linewidth=1)
        ax.axhline(cv["S_obs"] + cv["n_pseudo_unseen"], color=col, lw=0.8, ls="-.", alpha=0.5)
        if empirical and label in empirical:
            e = empirical[label]
            ax.plot(range(1, len(e) + 1), e, color=col, lw=0.8, alpha=0.5)
    _style(ax, "samples (m)", "expected distinct ideas E[K_m]", "Idea-discovery curve")
    ax.text(0.01, 0.99, "dot = S_obs at n   dash-dot = Chao1 N_hat   dotted = model E[K_m] for m <= n",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_rank_probability(rank_tables: dict[str, "pd.DataFrame"], out_path: Path, log_y: bool = True):
    """rank_tables[label] = DataFrame with columns rank, pi_hat (and pi0 as attrs['pi0'])."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for i, (label, tbl) in enumerate(rank_tables.items()):
        col = _color(i)
        ax.plot(tbl["rank"], tbl["pi_hat"], color=col, lw=2, marker="o", ms=3, label=label)
        pi0 = tbl.attrs.get("pi0")
        if pi0 is not None and pi0 == pi0:  # not nan
            ax.axhline(pi0, color=col, lw=0.8, ls="--", alpha=0.6)
    if log_y:
        ax.set_yscale("log")
    _style(ax, "idea rank (1 = modal idea)", "estimated probability pi_i",
           "Rank-probability curve  (dashed = pi_0, average scale of the unseen tail)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
