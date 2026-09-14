"""Hand-checked tests for every formula in the task description."""

import numpy as np
import pytest

from npv import (
    chao1_bias_corrected, chao1_bootstrap, detection_probability, discovery_curves,
    empirical_accumulation, estimate_N, estimate_P, estimate_V, expected_distinct,
    frequency_counts, rarefaction_curve, reconstructed_landscape, sample_coverage,
)

# Worked example: 6 observed ideas, counts x = [5, 3, 1, 1, 2, 1]
#   n = 13, S_obs = 6, f1 = 3, f2 = 1
#   N_hat = 6 + 3*2 / (2*(1+1)) = 6 + 1.5 = 7.5
#   C     = 1 - 3/13 = 10/13
#   pi_i  = C * x_i / 13
#   M0    = 3/13,  f0 = 1.5,  pi0 = (3/13)/1.5 = 2/13
COUNTS = np.array([5, 3, 1, 1, 2, 1])


def test_frequency_counts():
    assert frequency_counts(COUNTS) == (13, 6, 3, 1)


def test_chao1_bias_corrected():
    assert chao1_bias_corrected(6, 3, 1) == pytest.approx(7.5)
    assert chao1_bias_corrected(6, 0, 0) == 6.0           # f1 = 0 -> S_obs
    assert chao1_bias_corrected(6, 1, 0) == 6.0           # f1 = 1 -> f1(f1-1) = 0
    assert chao1_bias_corrected(10, 4, 0) == pytest.approx(10 + 4 * 3 / 2.0)  # f2 = 0 uses (f2+1)


def test_estimate_N_block():
    e = estimate_N(COUNTS)
    assert e.n == 13 and e.S_obs == 6 and e.f1 == 3 and e.f2 == 1
    assert e.N_hat == pytest.approx(7.5)
    assert e.f0_hat == pytest.approx(1.5)
    assert e.coverage == pytest.approx(10 / 13)
    assert e.M0 == pytest.approx(3 / 13)
    assert sample_coverage(13, 3) == pytest.approx(10 / 13)


def test_estimate_P_block():
    p = estimate_P(COUNTS, top_k=(1, 3, 100))
    C = 10 / 13
    np.testing.assert_allclose(p.pi_hat, C * COUNTS / 13)
    assert p.observed_mass == pytest.approx(C)             # sum pi_i = C, unseen mass = M0
    assert p.M0 == pytest.approx(3 / 13)
    assert p.pi0 == pytest.approx(2 / 13)
    assert p.modal_index == 0 and p.modal_pi == pytest.approx(C * 5 / 13)
    assert list(p.rank) == [1, 2, 4, 5, 3, 6]             # 5,3,2 then the singletons in input order
    assert p.top_k_mass[1] == pytest.approx(C * 5 / 13)
    assert p.top_k_mass[3] == pytest.approx(C * (5 + 3 + 2) / 13)
    assert p.top_k_mass[100] == pytest.approx(C)            # k > S_obs -> everything observed


def test_pi0_nan_when_no_unseen():
    p = estimate_P(np.array([4, 3, 2]))                     # f1 = 0 -> f0 = 0
    assert np.isnan(p.pi0) and p.M0 == 0 and p.coverage == 1


def test_expected_distinct_and_detection():
    pi = np.array([0.5, 0.3, 0.2])
    assert expected_distinct(pi, 1) == pytest.approx(1.0)
    assert expected_distinct(pi, 2) == pytest.approx(sum(1 - (1 - x) ** 2 for x in pi))
    grid = expected_distinct(pi, np.array([1, 2, 1000]))
    assert grid.shape == (3,) and grid[-1] == pytest.approx(3.0, abs=1e-9)
    np.testing.assert_allclose(detection_probability(pi, 2), 1 - (1 - pi) ** 2)


def test_estimate_V_block_by_hand():
    values = np.array([5.0, 1.0, 5.0, 1.0, 3.0, 3.0])
    tau = 3.0
    v = estimate_V(COUNTS, values, tau)
    C, n = 10 / 13, 13
    pi = C * COUNTS / n
    q = 1 - (1 - pi) ** n
    z = (values >= tau).astype(float)                        # [1,0,1,0,1,1]
    r_v = (z / q).sum() / (1 / q).sum()
    assert list(v.z) == list(z)
    np.testing.assert_allclose(v.q, q)
    assert v.S_V_obs == 4 and v.obs_valuable_share == pytest.approx(4 / 6)
    assert v.r_V == pytest.approx(r_v)
    assert v.N_V == pytest.approx(7.5 * r_v)
    assert v.Q_V == pytest.approx((z * pi).sum())
    assert v.mean_value_obs == pytest.approx(values.mean())
    assert v.mean_value_sample == pytest.approx((values * COUNTS).sum() / n)
    # inverse-detection weighting must up-weight rare (singleton) ideas relative to the naive share:
    # here valuable ideas include two singletons, so r_V > naive share
    assert v.r_V > v.obs_valuable_share


def test_estimate_V_rejects_zero_detection():
    with pytest.raises(ValueError, match="q_i = 0"):
        estimate_V(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0]), tau=2)   # C = 0


def test_estimate_V_rejects_nan_values():
    with pytest.raises(ValueError, match="NaN"):
        estimate_V(COUNTS, np.array([1, 2, np.nan, 4, 5, 6]), tau=2)


def test_rarefaction_endpoints_and_vs_permutation():
    n = int(COUNTS.sum())
    r = rarefaction_curve(COUNTS, np.array([1, n]))
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(6.0)
    labels = np.repeat(np.arange(COUNTS.size), COUNTS)
    emp = empirical_accumulation(labels, n_perm=4000, seed=1)
    full = rarefaction_curve(COUNTS, np.arange(1, n + 1))
    np.testing.assert_allclose(emp, full, atol=0.05)


def test_reconstructed_landscape_and_curves():
    pi_full, n_pseudo = reconstructed_landscape(COUNTS)
    assert n_pseudo == 2                                     # round(1.5) -> 2
    assert pi_full.sum() == pytest.approx(1.0)
    assert pi_full.size == 6 + 2
    cv = discovery_curves(COUNTS, extrapolate_to=26)
    assert cv["m_grid"][0] == 1 and cv["m_grid"][-1] == 26
    inside = cv["m_grid"] <= 13
    assert np.all(np.isnan(cv["rarefaction"][~inside])) and not np.any(np.isnan(cv["rarefaction"][inside]))
    assert np.all(np.diff(cv["model"]) >= 0)                 # monotone in m
    assert cv["model"][-1] <= 8.0                            # cannot exceed S_obs + pseudo unseen
    with pytest.raises(ValueError):
        discovery_curves(COUNTS, extrapolate_to=5)


def test_bootstrap_is_deterministic_and_brackets_point():
    b1 = chao1_bootstrap(COUNTS, n_boot=100, seed=3)
    b2 = chao1_bootstrap(COUNTS, n_boot=100, seed=3)
    assert b1 == b2
    lo, hi = b1["N_hat_ci95"]
    assert lo <= hi and lo >= 1
    assert b1["n_pseudo_unseen"] == 2


def test_input_validation():
    with pytest.raises(ValueError, match="count >= 1"):
        estimate_N(np.array([3, 0, 2]))
    with pytest.raises(ValueError, match="integers"):
        estimate_N(np.array([3.5, 2]))
    with pytest.raises(ValueError, match="empty"):
        estimate_N(np.array([]))


def test_chao1_recovers_known_N_on_simulation():
    """Uniform space of 50 ideas; with n = 400 Chao1 should land near 50 (lower bound, so <= ~50)."""
    rng = np.random.default_rng(0)
    true_pi = np.full(50, 1 / 50)
    draws = rng.multinomial(400, true_pi)
    e = estimate_N(draws[draws > 0])
    assert 44 <= e.N_hat <= 56
    assert e.coverage > 0.95
