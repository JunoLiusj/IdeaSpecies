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
    common = ["--condition-cols", "condition", "--value-col", "value", "--tau", "4",
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
    cols = ["n", "S_obs", "f1", "f2", "N_hat", "coverage", "pi0", "top1_mass", "r_V_tau4", "N_V_tau4", "Q_V_tau4"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    truth = json.loads((EXAMPLES / "example_truth.json").read_text())
    for cond in ["broad_flat", "narrow_peaked"]:
        assert a.loc[cond, "n"] == truth["n_per_condition"]
        assert abs(a.loc[cond, "N_hat"] - truth[cond]["true_N"]) <= 3          # Chao1 close to the truth here
        assert abs(a.loc[cond, "N_V_tau4"] - truth[cond]["true_valuable_N"]) <= 3
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
    assert tbl["z_tau4"].sum() == summ["S_V_obs_tau4"]


def test_refuses_to_overwrite(runs):
    s, _ = runs
    with pytest.raises(SystemExit, match="already exists"):
        main(["--input", str(EXAMPLES / "example_counts.csv"), "--format", "counts",
              "--out-root", str(s.parent), "--run-name", "s"])


def test_value_col_requires_tau(tmp_path):
    with pytest.raises(SystemExit, match="--tau is required"):
        main(["--input", str(EXAMPLES / "example_counts.csv"), "--format", "counts",
              "--value-col", "value", "--out-root", str(tmp_path)])
