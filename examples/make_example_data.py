"""Generate a synthetic example with KNOWN idea spaces so an RA can see what the
estimators recover.  Two conditions:

  broad_flat   : N = 80 ideas, near-uniform probabilities  -> discovery keeps going
  narrow_peaked: N = 40 ideas, Zipf-like concentration      -> fast saturation, fat tail

Each condition gets n = 300 independent samples.  A latent 1-5 quality score is
drawn once per idea and turned into a binary gold label valuable = 1[score >= 4],
standing in for whatever labelling procedure the RA uses.  Writes:

  examples/example_samples.csv   one row per sample (samples format)
  examples/example_counts.csv    one row per observed idea (counts format)
  examples/example_truth.json    true N, true valuable count, true top-1 mass

Run:  uv run python examples/make_example_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
TAU = 4.0
N_SAMPLES = 300


def make_condition(rng, name, n_ideas, zipf_s):
    ranks = np.arange(1, n_ideas + 1)
    pi = ranks ** (-zipf_s)
    pi = pi / pi.sum()
    values = np.clip(np.round(rng.normal(3.0, 1.0, n_ideas)), 1, 5)
    valuable = (values >= TAU).astype(int)
    draws = rng.choice(n_ideas, size=N_SAMPLES, replace=True, p=pi)
    samples = pd.DataFrame({
        "sample_id": [f"{name}_{i:04d}" for i in range(N_SAMPLES)],
        "condition": name,
        "idea_id": [f"{name}_idea{j:03d}" for j in draws],
        "valuable": valuable[draws],
    })
    truth = {
        "true_N": int(n_ideas),
        "true_valuable_N": int(valuable.sum()),
        "true_valuable_share": float(valuable.mean()),
        "true_valuable_mass": float(pi[valuable == 1].sum()),
        "true_top1_mass": float(pi.max()),
        "true_top3_mass": float(np.sort(pi)[::-1][:3].sum()),
        "true_expected_distinct_at_n": float((1 - (1 - pi) ** N_SAMPLES).sum()),
    }
    return samples, truth


def main():
    rng = np.random.default_rng(2026)
    parts, truth = [], {}
    for name, n_ideas, s in [("broad_flat", 80, 0.3), ("narrow_peaked", 40, 1.2)]:
        df, t = make_condition(rng, name, n_ideas, s)
        parts.append(df)
        truth[name] = t
    samples = pd.concat(parts, ignore_index=True)
    samples.to_csv(HERE / "example_samples.csv", index=False)
    counts = (samples.groupby(["condition", "idea_id"], as_index=False)
              .agg(count=("sample_id", "size"), valuable=("valuable", "first")))
    counts.to_csv(HERE / "example_counts.csv", index=False)
    (HERE / "example_truth.json").write_text(json.dumps({"latent_score_threshold": TAU,
                                                         "n_per_condition": N_SAMPLES,
                                                         **truth}, indent=2))
    print(samples.groupby("condition").idea_id.nunique().rename("S_obs"))
    print(json.dumps(truth, indent=2))


if __name__ == "__main__":
    main()
