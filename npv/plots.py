"""Figures: idea-discovery curves and rank-probability curves.

Fixed categorical palette (assigned by condition order, never cycled past 8:
extra conditions fold into gray and a warning is logged)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
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


# --------------------------------------------------------------------------- #
# Value landscape: before value enters vs after the valuable filter
# --------------------------------------------------------------------------- #
GRAY = "#B0B0B0"


def _stats_text(v: dict) -> str:
    return (f"S_obs {v['S_obs']} -> S_V {v['S_V_obs']}\n"
            f"N_hat {v['N_hat']:.1f} -> N_V {v['N_V']:.1f} [{v['N_V_lower']:.1f}, {v['N_V_upper']:.1f}]\n"
            f"mass C {v['coverage']:.2f} -> Q_V {v['Q_V']:.2f} ({v['Q_V'] / v['coverage']:.0%} kept)")


def plot_rank_landscape(panels: dict[str, dict], out_path: Path):
    """panels[label] = {pi, valuable, stats}; pi in rank order of the FULL sample.

    Before value enters: every idea as a gray stem at its rank.  After: valuable
    ideas keep their exact pi (colored stems); dropped ideas become hollow markers.
    Shared y-axis so heights are comparable across conditions."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 1, 4.2), dpi=150, sharey=True, squeeze=False)
    ymax = max(float(np.max(p["pi"])) for p in panels.values()) * 1.15
    for i, (label, p) in enumerate(panels.items()):
        ax, col = axes[0, i], _color(i)
        pi, keep = np.asarray(p["pi"]), np.asarray(p["valuable"]).astype(bool)
        rank = np.arange(1, pi.size + 1)
        ax.vlines(rank, 0, pi, color=GRAY, lw=1.2, alpha=0.8, label="before value enters" if i == 0 else None)
        ax.scatter(rank[~keep], pi[~keep], s=14, facecolor="white", edgecolor=GRAY, lw=0.8, zorder=3,
                   label="dropped by the valuable filter" if i == 0 else None)
        ax.vlines(rank[keep], 0, pi[keep], color=col, lw=2.2, zorder=4)
        ax.scatter(rank[keep], pi[keep], s=18, color=col, zorder=5, label="valuable (pi unchanged)" if i == 0 else None)
        ax.set_ylim(0, ymax)
        ax.text(0.98, 0.97, _stats_text(p["stats"]), transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        _style(ax, "idea rank (by pi, full sample)", "pi_i" if i == 0 else "", label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Value landscape: before value enters vs valuable filter (rank view)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def weighted_kde_grid(xy: np.ndarray, w: np.ndarray, gx: np.ndarray, gy: np.ndarray, h: float) -> np.ndarray:
    """Isotropic Gaussian KDE on a grid, weights NOT renormalised: the surface integrates to sum(w)."""
    GX, GY = np.meshgrid(gx, gy)
    dens = np.zeros(GX.shape)
    norm = 1.0 / (2 * np.pi * h * h)
    for (x, y), wi in zip(xy, w):
        dens += wi * norm * np.exp(-((GX - x) ** 2 + (GY - y) ** 2) / (2 * h * h))
    return dens


def scott_bandwidth(xy: np.ndarray, w: np.ndarray) -> float:
    """Scott's rule with the effective sample size of the weights, isotropic (mean marginal sd)."""
    wn = w / w.sum()
    n_eff = 1.0 / float((wn ** 2).sum())
    mu = wn @ xy
    var = wn @ ((xy - mu) ** 2)
    return float(np.sqrt(var.mean()) * n_eff ** (-1.0 / 6.0))


def plot_density_landscape(panels: dict[str, dict], out_path: Path, grid_n: int = 160):
    """panels[label] = {xy (S_obs x 2), pi, valuable, stats}.

    Same recipe as the project's contour / 3D landscape figures, reduced to 2-D:
      before = KDE of ALL ideas weighted by pi           (integrates to C)
      after  = KDE of VALUABLE ideas weighted by their pi (integrates to Q_V)
    One bandwidth per condition (Scott's rule on the full sample) used for both,
    one grid for all conditions, one color scale for all panels - so a smaller or
    lower 'after' surface is real, not an artefact of renormalisation."""
    all_xy = np.vstack([np.asarray(p["xy"]) for p in panels.values()])
    pad = 0.16 * np.ptp(all_xy, axis=0)
    gx = np.linspace(all_xy[:, 0].min() - pad[0], all_xy[:, 0].max() + pad[0], grid_n)
    gy = np.linspace(all_xy[:, 1].min() - pad[1], all_xy[:, 1].max() + pad[1], grid_n)
    surfaces = {}
    for label, p in panels.items():
        xy, pi, keep = np.asarray(p["xy"], float), np.asarray(p["pi"], float), np.asarray(p["valuable"]).astype(bool)
        h = scott_bandwidth(xy, pi)
        before = weighted_kde_grid(xy, pi, gx, gy, h)
        after = weighted_kde_grid(xy[keep], pi[keep], gx, gy, h) if keep.any() else np.zeros_like(before)
        surfaces[label] = (before, after, h)
    zmax = max(max(b.max(), a.max()) for b, a, _ in surfaces.values())
    levels = np.linspace(0, zmax, 9)[1:]
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n + 0.8, 7.2), dpi=150, squeeze=False)
    for i, (label, p) in enumerate(panels.items()):
        before, after, h = surfaces[label]
        xy, keep = np.asarray(p["xy"], float), np.asarray(p["valuable"]).astype(bool)
        col = _color(i)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("v", ["#ffffff", col])
        ax_b, ax_a = axes[0, i], axes[1, i]
        ax_b.contourf(gx, gy, before, levels=levels, cmap="Greys", alpha=0.85)
        ax_b.contour(gx, gy, before, levels=levels, colors=GRAY, linewidths=0.5)
        ax_b.scatter(xy[:, 0], xy[:, 1], s=6, color="#52514e", alpha=0.6)
        ax_b.set_title(f"{label}   before value enters", fontsize=9, loc="left", color=INK)
        ax_a.contour(gx, gy, before, levels=levels, colors=GRAY, linewidths=0.5, linestyles="--")
        ax_a.contourf(gx, gy, after, levels=levels, cmap=cmap, alpha=0.9)
        ax_a.scatter(xy[~keep, 0], xy[~keep, 1], s=8, facecolor="white", edgecolor=GRAY, lw=0.6)
        ax_a.scatter(xy[keep, 0], xy[keep, 1], s=8, color=col)
        ax_a.set_title("after the valuable filter (pi unchanged)", fontsize=9, loc="left", color=INK)
        ax_a.text(0.02, 0.02, _stats_text(p["stats"]) + f"\nbandwidth {h:.3g}", transform=ax_a.transAxes,
                  ha="left", va="bottom", fontsize=6.5, color=INK,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        for ax in (ax_b, ax_a):
            ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="box")
            for s in ax.spines.values():
                s.set_color(GRID)
    axes[0, 0].set_ylabel("semantic dim 2", color=MUTED); axes[1, 0].set_ylabel("semantic dim 2", color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("semantic dim 1", color=MUTED)
    fig.suptitle("Value landscape: pi-weighted density before vs after the valuable filter",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.955, "one grid, one bandwidth per condition and one color scale for all panels; "
             "dashed contours = before; hollow dots = dropped ideas", fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
