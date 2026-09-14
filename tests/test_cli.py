"""End-to-end: samples and counts formats must give identical estimates and all outputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npv.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

EXPECTED_FILES = [
    "resolved_config.json", "run.log", "summary.csv", "summary.json",
    "idea_table_broad_flat.csv", "idea_table_narrow_peaked.csv",
    "discovery_curve_broad_flat.csv", "discovery_curve_narrow_peaked.csv",
    "fig_discovery_curve.png", "fig_rank_probability.png",
]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("npv")
    common = ["--condition-cols", "condition", "--valuable-col", "valuable", "--gold-col", "gold",
              "--n-boot", "20", "--n-perm", "5", "--out-root", str(out)]
    s = main(["--input", str(EXAMPLES / "example_samples.csv"), "--format", "samples",
              "--run-name", "s", *common])
    c = main(["--input", str(EXAMPLES / "example_counts.csv"), "--format", "counts",
              "--run-name", "c", *common])
    return s, c


def test_outputs_exist(runs):
    s, c = runs
    for f in EXPECTED_FILES:
        assert (s / f).exists(), f
        assert (c / f).exists(), f
    assert (s / "empirical_accumulation_broad_flat.csv").exists()
    assert not (c / "empirical_accumulation_broad_flat.csv").exists()   # counts format has no samples


def test_formats_agree_and_match_truth(runs):
    s, c = runs
    a = pd.read_csv(s / "summary.csv").set_index("condition").sort_index()
    b = pd.read_csv(c / "summary.csv").set_index("condition").sort_index()
    cols = ["n", "S_obs", "f1", "f2", "N_hat", "coverage", "pi0", "top1_mass",
            "N_obs_valuable", "N_V", "N_V_lower", "N_V_upper", "Q_V", "P_V", "calib_PPV", "calib_NPV"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    truth = json.loads((EXAMPLES / "example_truth.json").read_text())
    for cond in ["broad_flat", "narrow_peaked"]:
        assert a.loc[cond, "n"] == truth["n_per_condition"]
        assert abs(a.loc[cond, "N_hat"] - truth[cond]["true_N"]) <= 3          # Chao1 close to the truth here
        assert a.loc[cond, "N_V_lower"] <= a.loc[cond, "N_V"] <= a.loc[cond, "N_V_upper"]
        assert abs(a.loc[cond, "N_V"] - truth[cond]["true_valuable_N"]) <= 5   # calibrated, noisy auto label
    assert list(a.sort_values("N_hat").index) == list(pd.read_csv(s / "summary.csv")["condition"])  # worst first


def test_idea_table_consistency(runs):
    s, _ = runs
    tbl = pd.read_csv(s / "idea_table_narrow_peaked.csv")
    summ = pd.read_csv(s / "summary.csv").set_index("condition").loc["narrow_peaked"]
    assert tbl["count"].sum() == summ["n"]
    assert len(tbl) == summ["S_obs"]
    assert tbl["pi_hat"].sum() == pytest.approx(summ["coverage"])
    assert list(tbl["rank"]) == list(range(1, len(tbl) + 1))
    np.testing.assert_allclose(tbl["q_detect"], 1 - (1 - tbl["pi_hat"]) ** summ["n"])
    assert tbl["valuable"].sum() == summ["S_V_obs"]
    assert tbl["r_valuable"].sum() == pytest.approx(summ["N_obs_valuable"])
    labelled = tbl["gold"].notna()
    assert labelled.sum() == summ["calib_n_gold"]
    np.testing.assert_allclose(tbl.loc[labelled, "r_valuable"], tbl.loc[labelled, "gold"])   # human wins
    assert (tbl["pi_hat"] * tbl["r_valuable"]).sum() == pytest.approx(summ["Q_V"])
    assert np.nansum(tbl["pi_within_valuable"]) == pytest.approx(summ["Q_V"] / summ["P_V"])


def test_refuses_to_overwrite(runs):
    s, _ = runs
    with pytest.raises(SystemExit, match="already exists"):
        main(["--input", str(EXAMPLES / "example_counts.csv"), "--format", "counts",
              "--out-root", str(s.parent), "--run-name", "s"])


def test_inconsistent_labels_within_idea_are_rejected(tmp_path):
    df = pd.read_csv(EXAMPLES / "example_samples.csv")
    first_idea = df["idea_id"].iloc[0]
    rows = df.index[df["idea_id"] == first_idea]
    assert len(rows) >= 2
    df.loc[rows[0], "valuable"] = 1 - df.loc[rows[0], "valuable"]      # flip one sample's label
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="both valuable=1 and valuable=0"):
        main(["--input", str(bad), "--format", "samples", "--condition-cols", "condition",
              "--valuable-col", "valuable", "--n-boot", "0", "--n-perm", "0",
              "--out-root", str(tmp_path / "out")])
