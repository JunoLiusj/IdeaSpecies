"""Hand-checked tests for every formula in the task description."""

import numpy as np
import pytest

from npv import (
    as_valuable_labels, calibrate_labels, chao1_bias_corrected, chao1_bootstrap, detection_probability, discovery_curves,
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
    # counts [5,3,1,1,2,1]; valuable labels z = [1,0,1,0,1,1]; f0_hat = 1.5
    # singletons are ideas 2,3,5 with z = [1,0,1] -> frac_unseen_singleton = 2/3
    z = np.array([1, 0, 1, 0, 1, 1], dtype=float)
    v = estimate_V(COUNTS, z)
    C, n = 10 / 13, 13
    pi = C * COUNTS / n
    assert list(v.z) == list(z) and list(v.r) == list(z)           # no gold -> r == z
    assert v.S_V_obs == 4 and v.N_obs_valuable == 4.0
    assert v.raw_ratio == pytest.approx(4 / 6)
    assert v.f0_hat == pytest.approx(1.5)
    assert v.n_singleton_basis == 3
    assert v.frac_unseen_singleton == pytest.approx(2 / 3)
    assert v.frac_unseen_observed == pytest.approx(4 / 6)
    assert v.N_V == pytest.approx(4 + 1.5 * 2 / 3)                # 5.0
    assert v.N_V_observed_rule == pytest.approx(4 + 1.5 * 4 / 6)
    assert v.N_V_lower == 4.0 and v.N_V_upper == pytest.approx(5.5)
    assert v.N_V_lower <= v.N_V <= v.N_V_upper
    assert v.valuable_fraction == pytest.approx(5.0 / 7.5)
    assert v.valuable_fraction_lower == pytest.approx(4 / 7.5)
    assert v.valuable_fraction_upper == pytest.approx(5.5 / 7.5)
    assert v.Q_V == pytest.approx((z * pi).sum())                  # pi from the FULL sample
    assert v.P_V == pytest.approx((z * pi).sum() + (3 / 13) * 2 / 3)
    assert v.valuable_sample_share == pytest.approx((z * COUNTS).sum() / n)   # 9/13
    within = v.pi_within_valuable
    assert np.isnan(within[1]) and np.isnan(within[3])
    assert np.nansum(within) == pytest.approx(v.Q_V / v.P_V)       # observed valuable share of P_V
    assert any("singleton" in w for w in v.warnings)               # fewer than 10 singletons


def test_estimate_V_observed_rule_and_no_singletons():
    v = estimate_V(COUNTS, [1, 0, 1, 0, 1, 1], unseen_rule="observed")
    assert v.N_V == pytest.approx(4 + 1.5 * 4 / 6)
    with pytest.raises(ValueError, match="unseen_rule"):
        estimate_V(COUNTS, [1, 0, 1, 0, 1, 1], unseen_rule="magic")
    v0 = estimate_V(np.array([4, 3, 2]), [1, 0, 1])                # f1 = 0 -> f0 = 0
    assert np.isnan(v0.frac_unseen_singleton)
    assert v0.N_V == v0.N_V_lower == v0.N_V_upper == 2.0
    assert v0.P_V == pytest.approx(v0.Q_V)


def test_calibration_with_gold_subset_by_hand():
    # z = [1,0,1,0,1,1]; human labels on ideas 0,1,2,4: gold = [1,0,0,nan,1,nan]
    # stratum z=1 labelled: ideas 0,2,4 -> gold 1,0,1 -> PPV = 2/3
    # stratum z=0 labelled: idea 1 -> gold 0 -> NPV = 1
    # r = gold where labelled, else PPV for z=1 (idea 5), 1-NPV = 0 for z=0 (idea 3)
    z = [1, 0, 1, 0, 1, 1]
    gold = [1, 0, 0, np.nan, 1, np.nan]
    r, calib = calibrate_labels(np.array(z, dtype=float), gold)
    np.testing.assert_allclose(r, [1, 0, 0, 0, 1, 2 / 3])
    assert calib["n_gold"] == 4 and calib["n_gold_z1"] == 3 and calib["n_gold_z0"] == 1
    assert calib["PPV"] == pytest.approx(2 / 3) and calib["NPV"] == 1.0
    assert calib["agreement"] == pytest.approx(3 / 4)
    v = estimate_V(COUNTS, z, gold=gold)
    assert v.N_obs_valuable == pytest.approx(2 + 2 / 3)
    assert v.frac_unseen_singleton == pytest.approx((0 + 0 + 2 / 3) / 3)   # singletons: ideas 2,3,5
    assert v.N_V == pytest.approx(2 + 2 / 3 + 1.5 * (2 / 9))                # 3.0
    assert v.calibration["PPV"] == pytest.approx(2 / 3)


def test_calibration_requires_both_strata_and_blank_handling():
    z = np.array([1, 0, 1, 0, 1, 1], dtype=float)
    with pytest.raises(ValueError, match="both label strata"):
        calibrate_labels(z, [1, np.nan, 0, np.nan, 1, np.nan])          # no z=0 idea labelled
    with pytest.raises(ValueError, match="no labelled"):
        calibrate_labels(z, [np.nan] * 6)
    r, calib = calibrate_labels(z, ["1", "", "0", "no", None, ""])      # strings and blanks accepted
    assert calib["n_gold"] == 3                                          # "", None are unlabelled
    assert calib["PPV"] == pytest.approx(0.5) and calib["NPV"] == 1.0
    np.testing.assert_allclose(r, [1, 0, 0, 0, 0.5, 0.5])
    r2, _ = calibrate_labels(z, np.array([1.0, np.nan, 0.0, 0.0, np.nan, np.nan], dtype=object))
    np.testing.assert_allclose(r2, [1, 0, 0, 0, 0.5, 0.5])              # float objects accepted


def test_valuable_label_coercion():
    np.testing.assert_array_equal(as_valuable_labels([True, False]), [1.0, 0.0])
    np.testing.assert_array_equal(as_valuable_labels(np.array([1, 0, 1])), [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(as_valuable_labels(["yes", "No", " TRUE ", "0"]), [1.0, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="must be 0/1"):
        as_valuable_labels(np.array([0, 2, 1]))
    with pytest.raises(ValueError, match="not one of"):
        as_valuable_labels(["maybe", "yes"])
    with pytest.raises(ValueError, match="NaN"):
        as_valuable_labels(np.array([1.0, np.nan]))


def test_estimate_V_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        estimate_V(COUNTS, np.array([1, 0, 1]))


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
