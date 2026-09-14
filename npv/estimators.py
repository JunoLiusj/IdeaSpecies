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
V  (value landscape; valuable is a FILTER, it never changes pi_i)
    z_i      = 1 if idea i is labelled valuable else 0   (final label from the RA's procedure,
                                                          after whatever calibration they designed)
    N_obs_V  = sum_obs z_i                          observed valuable ideas
    f0_hat   = N_hat - S_obs                        unseen ideas (Chao1 lower bound)
    frac_u   = mean z_i over singletons             declared assumption for the unseen part
    N_V      = N_obs_V + f0_hat * frac_u            point estimate ("rare ideas proxy the unseen")
    bounds   = [N_obs_V, N_obs_V + f0_hat]          no unseen valuable / all unseen valuable
    Q_V      = sum_obs pi_i z_i                     mass on observed valuable ideas (pi from full sample)
    P_V      = Q_V + M0 * frac_u                    incl. the unseen valuable mass
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
    z: np.ndarray                   # final valuable label z_i per observed idea (0/1)
    pi_within_valuable: np.ndarray  # pi_i / P_V for valuable ideas, nan otherwise
    S_V_obs: int                    # #{z_i = 1} = N_obs_valuable
    raw_ratio: float                # S_V_obs / S_obs   (Tier 1, no unseen assumption)
    f0_hat: float
    frac_unseen_singleton: float    # mean z_i over singletons (nan when f1 = 0)
    n_singleton_basis: int
    frac_unseen_observed: float     # mean z_i over all observed ideas
    N_V: float                      # S_V_obs + f0_hat * frac_unseen  (rule chosen by caller)
    N_V_observed_rule: float        # same with frac_unseen_observed
    N_V_lower: float                # S_V_obs            (no unseen idea is valuable)
    N_V_upper: float                # S_V_obs + f0_hat   (every unseen idea is valuable)
    valuable_fraction: float        # N_V / N_hat
    valuable_fraction_lower: float  # N_V_lower / N_hat
    valuable_fraction_upper: float  # N_V_upper / N_hat
    Q_V: float                      # sum pi_i z_i   (observed valuable mass, pi from full sample)
    P_V: float                      # Q_V + M0 * frac_unseen
    valuable_sample_share: float    # sum x_i z_i / n
    unseen_rule: str
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "S_V_obs": self.S_V_obs, "raw_ratio": self.raw_ratio,
            "frac_unseen_singleton": self.frac_unseen_singleton, "n_singleton_basis": self.n_singleton_basis,
            "frac_unseen_observed": self.frac_unseen_observed, "unseen_rule": self.unseen_rule,
            "N_V": self.N_V, "N_V_lower": self.N_V_lower, "N_V_upper": self.N_V_upper,
            "N_V_observed_rule": self.N_V_observed_rule,
            "valuable_fraction": self.valuable_fraction,
            "valuable_fraction_lower": self.valuable_fraction_lower,
            "valuable_fraction_upper": self.valuable_fraction_upper,
            "Q_V": self.Q_V, "P_V": self.P_V, "valuable_sample_share": self.valuable_sample_share,
        }


def estimate_V(counts, valuable, unseen_rule: str = "singleton") -> VEstimate:
    """V block.  ``valuable`` is the RA's FINAL 0/1 label per observed idea (any
    calibration against human labels happens upstream, in the RA's own pipeline).
    Value never touches pi_i: it is a filter over ideas whose probabilities come
    from the full sample."""
    c = _as_counts(counts)
    z = as_valuable_labels(valuable)
    if z.shape != c.shape:
        raise ValueError(f"valuable shape {z.shape} must match counts shape {c.shape}")
    if unseen_rule not in {"singleton", "observed"}:
        raise ValueError("unseen_rule must be 'singleton' or 'observed'")
    warnings: list[str] = []
    r = z

    n_est = estimate_N(c)
    p = estimate_P(c)
    n_obs_v = float(r.sum())
    singles = c == 1
    n_single = int(singles.sum())
    frac_single = float(r[singles].mean()) if n_single > 0 else float("nan")
    frac_obs = float(r.mean())
    if 0 < n_single < 10:
        warnings.append(f"only {n_single} singleton(s) support frac_unseen_singleton; the unseen "
                        "valuable fraction is unstable - read N_V with its bounds.")
    if n_single == 0:
        warnings.append("f1 = 0: no singletons, f0_hat = 0, so N_V = N_obs_valuable with no unseen part.")
    frac_used = frac_single if unseen_rule == "singleton" else frac_obs
    f0 = n_est.f0_hat
    unseen_part = f0 * frac_used if f0 > 0 else 0.0
    n_v = n_obs_v + unseen_part
    n_v_obs_rule = n_obs_v + (f0 * frac_obs if f0 > 0 else 0.0)
    q_v = float((p.pi_hat * r).sum())
    p_v = q_v + (n_est.M0 * frac_used if f0 > 0 else 0.0)
    within = np.where(r > 0, p.pi_hat * r / p_v, np.nan) if p_v > 0 else np.full(c.size, np.nan)
    return VEstimate(
        z=z, pi_within_valuable=within, S_V_obs=int(z.sum()),
        raw_ratio=n_obs_v / n_est.S_obs, f0_hat=f0,
        frac_unseen_singleton=frac_single, n_singleton_basis=n_single, frac_unseen_observed=frac_obs,
        N_V=n_v, N_V_observed_rule=n_v_obs_rule, N_V_lower=n_obs_v, N_V_upper=n_obs_v + f0,
        valuable_fraction=n_v / n_est.N_hat, valuable_fraction_lower=n_obs_v / n_est.N_hat,
        valuable_fraction_upper=(n_obs_v + f0) / n_est.N_hat,
        Q_V=q_v, P_V=p_v, valuable_sample_share=float((c * r).sum() / c.sum()),
        unseen_rule=unseen_rule, warnings=warnings,
    )


def as_valuable_labels(labels) -> np.ndarray:
    """Coerce a gold valuable label vector to a 0/1 float array.

    Accepts booleans, the integers 0/1, or the strings true/false, yes/no, 1/0
    (case-insensitive).  Anything else (2, 0.5, NaN, 'maybe') is an error: the
    task description's V block needs a binary label per observed idea."""
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"valuable labels must be 1-D, got shape {arr.shape}")
    if arr.dtype == bool:
        return arr.astype(float)
    if np.issubdtype(arr.dtype, np.number):
        if np.any(np.isnan(arr.astype(float))):
            raise ValueError("valuable labels contain NaN: every observed idea needs a label")
        if not np.all(np.isin(arr, [0, 1])):
            bad = np.unique(arr[~np.isin(arr, [0, 1])])[:5]
            raise ValueError(f"valuable labels must be 0/1, found {bad.tolist()}")
        return arr.astype(float)
    mapping = {"1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0,
               "true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0}
    out = np.empty(arr.size, dtype=float)
    for i, s in enumerate(arr):
        if isinstance(s, (bool, np.bool_)):
            out[i] = float(s)
            continue
        if isinstance(s, (int, float, np.integer, np.floating)):
            if s != s or s not in (0, 1):
                raise ValueError(f"valuable labels must be 0/1, found {s!r}")
            out[i] = float(s)
            continue
        key = str(s).strip().lower()
        if key not in mapping:
            raise ValueError(f"valuable label {s!r} is not one of 0/1, true/false, yes/no")
        out[i] = mapping[key]
    return out


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
