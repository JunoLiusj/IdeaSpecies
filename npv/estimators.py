"""Pure-numpy estimators for the N-P-V framework.

Every function takes a 1-D integer array ``counts`` where ``counts[i] = x_i`` is
the number of times idea ``i`` appeared among ``n = counts.sum()`` independent
samples.  Nothing here reads files or knows about conditions; see ``io.py`` and
``cli.py`` for that.

Formulas (verbatim from the task description)
---------------------------------------------
N  (richness, bias-corrected Chao1 lower bound)
    N_hat = S_obs + f1 (f1 - 1) / (2 (f2 + 1))
P  (probability landscape, Good-Turing coverage)
    C      = 1 - f1 / n
    pi_i   = C x_i / n
    M0     = 1 - C = f1 / n            (mass of unseen ideas)
    f0     = N_hat - S_obs             (implied lower-bound unseen count)
    pi_0   = M0 / f0    when f0 > 0    (average scale of the unseen tail)
Discovery curve
    E[K_n] = sum_i 1 - (1 - pi_i)^n
V  (value landscape, threshold tau)
    z_i(tau) = 1[value_i >= tau]
    q_i      = 1 - (1 - pi_i)^n        (probability idea i appears at least once)
    r_V(tau) = sum_obs z_i / q_i  /  sum_obs 1 / q_i
    N_V(tau) = N_hat r_V(tau)
    Q_V(tau) = sum_obs z_i pi_i
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import lgamma

import numpy as np


# --------------------------------------------------------------------------- #
# Basic frequency quantities
# --------------------------------------------------------------------------- #
def _as_counts(counts) -> np.ndarray:
    c = np.asarray(counts)
    if c.ndim != 1:
        raise ValueError(f"counts must be 1-D, got shape {c.shape}")
    if c.size == 0:
        raise ValueError("counts is empty: no ideas observed")
    if not np.issubdtype(c.dtype, np.integer):
        if not np.all(np.equal(np.mod(c, 1), 0)):
            raise ValueError("counts must be integers (number of samples per idea)")
        c = c.astype(np.int64)
    if np.any(c <= 0):
        raise ValueError(
            "every observed idea must have count >= 1; drop unobserved ideas "
            "(count 0) before estimation - they are exactly what N estimates"
        )
    return c.astype(np.int64)


def frequency_counts(counts) -> tuple[int, int, int, int]:
    """Return (n, S_obs, f1, f2)."""
    c = _as_counts(counts)
    n = int(c.sum())
    s_obs = int(c.size)
    f1 = int((c == 1).sum())
    f2 = int((c == 2).sum())
    return n, s_obs, f1, f2


def chao1_bias_corrected(s_obs: int, f1: int, f2: int) -> float:
    """N_hat = S_obs + f1 (f1 - 1) / (2 (f2 + 1)).  Always finite (f2 + 1 > 0)."""
    return float(s_obs) + f1 * (f1 - 1) / (2.0 * (f2 + 1))


def sample_coverage(n: int, f1: int) -> float:
    """Good-Turing sample coverage C = 1 - f1 / n."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 - f1 / n


def expected_distinct(pi, n):
    """E[K_n] = sum_i 1 - (1 - pi_i)^n.  ``n`` may be a scalar or a 1-D grid."""
    pi = np.asarray(pi, dtype=float)
    n_arr = np.atleast_1d(np.asarray(n, dtype=float))
    out = (1.0 - (1.0 - pi[None, :]) ** n_arr[:, None]).sum(axis=1)
    return float(out[0]) if np.ndim(n) == 0 else out


def detection_probability(pi, n: int) -> np.ndarray:
    """q_i = 1 - (1 - pi_i)^n."""
    pi = np.asarray(pi, dtype=float)
    return 1.0 - (1.0 - pi) ** n


# --------------------------------------------------------------------------- #
# N: richness
# --------------------------------------------------------------------------- #
@dataclass
class NEstimate:
    n: int
    S_obs: int
    f1: int
    f2: int
    f0_hat: float
    N_hat: float
    coverage: float
    M0: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_N(counts) -> NEstimate:
    n, s_obs, f1, f2 = frequency_counts(counts)
    n_hat = chao1_bias_corrected(s_obs, f1, f2)
    cov = sample_coverage(n, f1)
    warnings = []
    if f1 == 0:
        warnings.append(
            "f1 = 0: no singletons, Chao1 collapses to S_obs and coverage is 1. "
            "Either the space is exhausted or ideas were over-merged."
        )
    if f1 == n:
        warnings.append(
            "f1 = n: every sample is a singleton, coverage is 0 and all pi_i = 0. "
            "The sample is far too small (or ideas were over-split) for P/V estimates."
        )
    if f2 == 0 and f1 > 0:
        warnings.append(
            "f2 = 0: Chao1 relies on the (f2 + 1) bias correction only; treat N_hat as very rough."
        )
    if cov < 0.5:
        warnings.append(
            f"low sample coverage C = {cov:.3f}: rare-idea probabilities are highly uncertain."
        )
    return NEstimate(
        n=n, S_obs=s_obs, f1=f1, f2=f2, f0_hat=n_hat - s_obs, N_hat=n_hat,
        coverage=cov, M0=1.0 - cov, warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# P: probability landscape
# --------------------------------------------------------------------------- #
@dataclass
class PEstimate:
    n: int
    coverage: float
    pi_hat: np.ndarray          # coverage-adjusted probability per observed idea (input order)
    rank: np.ndarray            # 1 = most probable (ties broken by input order)
    M0: float
    f0_hat: float
    pi0: float                  # nan when f0_hat == 0
    modal_index: int
    modal_pi: float
    top_k_mass: dict[int, float]
    observed_mass: float        # sum pi_hat == C

    def to_dict(self) -> dict:
        return {
            "n": self.n, "coverage": self.coverage, "M0": self.M0, "f0_hat": self.f0_hat,
            "pi0": self.pi0, "modal_index": self.modal_index, "modal_pi": self.modal_pi,
            "observed_mass": self.observed_mass,
            **{f"top{k}_mass": v for k, v in self.top_k_mass.items()},
        }


def estimate_P(counts, top_k=(1, 3, 5, 10)) -> PEstimate:
    c = _as_counts(counts)
    n_est = estimate_N(c)
    n, cov = n_est.n, n_est.coverage
    pi_hat = cov * c / n
    order = np.argsort(-c, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, c.size + 1)
    sorted_pi = pi_hat[order]
    cum = np.cumsum(sorted_pi)
    top_mass = {int(k): float(cum[min(int(k), c.size) - 1]) for k in top_k}
    pi0 = n_est.M0 / n_est.f0_hat if n_est.f0_hat > 0 else float("nan")
    return PEstimate(
        n=n, coverage=cov, pi_hat=pi_hat, rank=rank, M0=n_est.M0, f0_hat=n_est.f0_hat,
        pi0=pi0, modal_index=int(order[0]), modal_pi=float(sorted_pi[0]),
        top_k_mass=top_mass, observed_mass=float(pi_hat.sum()),
    )


# --------------------------------------------------------------------------- #
# V: value landscape
# --------------------------------------------------------------------------- #
@dataclass
class VEstimate:
    tau: float
    z: np.ndarray               # z_i(tau) per observed idea
    q: np.ndarray               # detection probability q_i per observed idea
    S_V_obs: int                # number of observed valuable ideas
    obs_valuable_share: float   # S_V_obs / S_obs (naive, not detection-weighted)
    r_V: float
    N_V: float
    Q_V: float
    mean_value_obs: float       # unweighted mean of idea values (observed ideas)
    mean_value_sample: float    # sample-weighted mean (each generation counts once)

    def to_dict(self) -> dict:
        return {
            "tau": self.tau, "S_V_obs": self.S_V_obs,
            "obs_valuable_share": self.obs_valuable_share,
            "r_V": self.r_V, "N_V": self.N_V, "Q_V": self.Q_V,
            "mean_value_obs": self.mean_value_obs, "mean_value_sample": self.mean_value_sample,
        }


def estimate_V(counts, values, tau: float) -> VEstimate:
    c = _as_counts(counts)
    v = np.asarray(values, dtype=float)
    if v.shape != c.shape:
        raise ValueError(f"values shape {v.shape} must match counts shape {c.shape}")
    if np.any(np.isnan(v)):
        raise ValueError("values contain NaN: every observed idea needs a value for V estimation")
    p = estimate_P(c)
    n_est = estimate_N(c)
    q = detection_probability(p.pi_hat, p.n)
    if np.any(q <= 0):
        raise ValueError(
            "some q_i = 0 (coverage C = 0, all samples are singletons); "
            "inverse-detection weighting is undefined - collect more samples"
        )
    z = (v >= tau).astype(float)
    w = 1.0 / q
    r_v = float((z * w).sum() / w.sum())
    return VEstimate(
        tau=float(tau), z=z, q=q, S_V_obs=int(z.sum()), obs_valuable_share=float(z.mean()),
        r_V=r_v, N_V=float(n_est.N_hat * r_v), Q_V=float((z * p.pi_hat).sum()),
        mean_value_obs=float(v.mean()), mean_value_sample=float((v * c).sum() / c.sum()),
    )


# --------------------------------------------------------------------------- #
# Discovery curves
# --------------------------------------------------------------------------- #
def _log_choose(a: float, b: float) -> float:
    return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)


def rarefaction_curve(counts, m_grid) -> np.ndarray:
    """Exact interpolation: E[S_m] = S_obs - sum_i C(n - x_i, m) / C(n, m), 1 <= m <= n.

    Expected number of distinct ideas in a random subsample of size m drawn
    without replacement from the observed sample (hypergeometric)."""
    c = _as_counts(counts)
    n = int(c.sum())
    m_grid = np.asarray(m_grid, dtype=np.int64)
    if np.any(m_grid < 1) or np.any(m_grid > n):
        raise ValueError(f"rarefaction grid must lie in [1, n={n}]")
    out = np.empty(m_grid.size, dtype=float)
    for j, m in enumerate(m_grid):
        m = int(m)
        log_denom = _log_choose(float(n), float(m))
        term = 0.0
        for x in c:
            rem = n - int(x)
            if rem >= m:
                term += np.exp(_log_choose(float(rem), float(m)) - log_denom)
        out[j] = c.size - term
    return out


def empirical_accumulation(idea_labels, n_perm: int = 200, seed: int = 0) -> np.ndarray:
    """Mean over random orderings of the *actual* samples: cumulative distinct count
    after m samples, m = 1..n.  Sanity check on rarefaction; needs sample-level labels."""
    labels = np.asarray(idea_labels)
    n = labels.size
    if n == 0:
        raise ValueError("no samples")
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    for _ in range(n_perm):
        perm = labels[rng.permutation(n)]
        _, first_idx = np.unique(perm, return_index=True)
        seen = np.zeros(n)
        seen[first_idx] = 1
        acc += np.cumsum(seen)
    return acc / n_perm


def reconstructed_landscape(counts) -> tuple[np.ndarray, int]:
    """Estimated full probability vector: observed pi_hat plus round(f0_hat) unseen
    ideas each at pi_0 (renormalised).  Returns (pi_full, n_pseudo_unseen)."""
    p = estimate_P(counts)
    n_pseudo = int(round(p.f0_hat)) if p.f0_hat > 0 else 0
    if n_pseudo > 0:
        pi_full = np.concatenate([p.pi_hat, np.full(n_pseudo, p.M0 / n_pseudo)])
    else:
        pi_full = p.pi_hat.copy()
    total = pi_full.sum()
    if total <= 0:
        raise ValueError("reconstructed landscape has zero mass (coverage 0 and no unseen ideas)")
    return pi_full / total, n_pseudo


def discovery_curves(counts, extrapolate_to: int | None = None, step: int | None = None) -> dict:
    """Idea-discovery curve on a common grid.

    Returns a dict with arrays on ``m_grid``:
      * ``rarefaction``   exact expected distinct ideas for m <= n (nan beyond n)
      * ``model``         E[K_m] = sum_i 1-(1-pi_i)^m on the reconstructed landscape
                          (observed pi_hat + f0 unseen ideas at pi_0), all m
    The model curve at m = n need not equal S_obs; the gap is a diagnostic of how
    well the (pi_hat, pi_0) reconstruction describes the sample."""
    c = _as_counts(counts)
    n = int(c.sum())
    if extrapolate_to is None:
        extrapolate_to = 2 * n
    if extrapolate_to < n:
        raise ValueError("extrapolate_to must be >= n")
    if step is None:
        step = max(1, extrapolate_to // 200)
    m_grid = np.unique(np.concatenate([np.arange(1, extrapolate_to + 1, step), [n, extrapolate_to]]))
    rare = np.full(m_grid.size, np.nan)
    inside = m_grid <= n
    rare[inside] = rarefaction_curve(c, m_grid[inside])
    pi_full, n_pseudo = reconstructed_landscape(c)
    model = expected_distinct(pi_full, m_grid)
    return {"m_grid": m_grid, "rarefaction": rare, "model": model, "n": n,
            "n_pseudo_unseen": n_pseudo, "S_obs": int(c.size)}


# --------------------------------------------------------------------------- #
# Bootstrap uncertainty for the Chao1 lower bound (optional extra)
# --------------------------------------------------------------------------- #
def chao1_bootstrap(counts, n_boot: int = 200, seed: int = 0) -> dict:
    """Population-reconstruction bootstrap (Chao et al. 2014, Appendix G style).

    Resample n draws from the reconstructed landscape (observed pi_hat + f0 unseen
    ideas at pi_0) so a resample *can* discover ideas outside S_obs, recompute
    Chao1 and coverage on each resample, and report percentile intervals.
    This quantifies uncertainty in the lower bound, not a two-sided interval for N.
    """
    c = _as_counts(counts)
    n = int(c.sum())
    pi_full, n_pseudo = reconstructed_landscape(c)
    rng = np.random.default_rng(seed)
    boot_n = np.empty(n_boot)
    boot_c = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.multinomial(n, pi_full)
        cb = draw[draw > 0]
        _, s_b, f1_b, f2_b = frequency_counts(cb)
        boot_n[b] = chao1_bias_corrected(s_b, f1_b, f2_b)
        boot_c[b] = sample_coverage(n, f1_b)
    return {
        "n_boot": n_boot, "seed": seed, "n_pseudo_unseen": n_pseudo,
        "N_hat_ci95": [float(np.percentile(boot_n, 2.5)), float(np.percentile(boot_n, 97.5))],
        "N_hat_boot_sd": float(boot_n.std(ddof=1)) if n_boot > 1 else float("nan"),
        "coverage_ci95": [float(np.percentile(boot_c, 2.5)), float(np.percentile(boot_c, 97.5))],
    }
