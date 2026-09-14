# RA N-P-V Estimation Toolkit

A small, dependency-light tool that computes every estimator written in the
**Stage 2 Task for RAs** document ("The N-P-V framework" section) from an idea-level
or sample-level table. It does *not* identify ideas, embed text, or call any model:
you bring a table where each independent generation has already been assigned to
one idea category (and, optionally, each idea carries a binary gold label saying
whether it is valuable, obtained however your study defines value).

```
N  richness             bias-corrected Chao1 lower bound, sample coverage, discovery curve
P  probability landscape coverage-adjusted pi_i, unseen mass M0, f0, pi_0, modal idea,
                         top-k mass, rank-probability curve
V  value landscape      valuable filter z_i (optionally calibrated on a human-labelled subset),
                         N_V with bounds, valuable mass Q_V / P_V, pi within the valuable space
```

## 1. Install and run (uv)

```bash
cd ra_npv_toolkit
uv sync                                  # creates .venv with numpy / pandas / matplotlib
uv run pytest -q                         # 14 hand-checked tests should pass

# synthetic example with known truth (two conditions, 300 samples each)
uv run python examples/make_example_data.py
uv run npv-estimate --input examples/example_samples.csv --format samples \
    --condition-cols condition --valuable-col valuable --gold-col gold \
    --out-root examples/results --run-name demo
```

Each run writes a **new** folder `<out-root>/<run-name>/` and refuses to overwrite an
existing one. Without `--run-name` the folder is `npv_<timestamp>`.

## 2. Input formats

Both are CSV (or TSV by extension). Column names are configurable.

**`--format samples`** — one row per independent generation (the recommended form;
it also enables the empirical accumulation curve).

| column (default name) | required | meaning |
|---|---|---|
| `idea_id` (`--idea-col`) | yes | idea category the sample was assigned to |
| condition columns (`--condition-cols model,prompt`) | no | one estimate per unique combination |
| `valuable` (`--valuable-col`) | no | the idea's valuable label from your procedure: `1`/`0`, `true`/`false` or `yes`/`no`; must be identical for all samples of one idea (the run stops otherwise) |
| `gold` (`--gold-col`) | no | HUMAN `1`/`0` label on a subset of ideas, blank elsewhere; calibrates `valuable` (section 3.1) |

**`--format counts`** — one row per *observed* idea.

| column | required | meaning |
|---|---|---|
| `idea_id` | yes | idea id (unique within a condition) |
| `count` (`--count-col`) | yes | x_i, number of samples that produced the idea (>= 1) |
| condition columns | no | as above |
| `valuable` | no | idea-level label, `1`/`0` (or `true`/`false`, `yes`/`no`) |
| `gold` | no | human label on a subset, blank elsewhere |

Do **not** include rows with count 0: unobserved ideas are exactly what N estimates.
Missing values, non-integer counts, duplicate idea rows, or labels other than 0/1
stop the run with an explicit error (no silent defaults).

How the valuable label is produced is up to your design (blinded human rating with a
preregistered threshold, an LLM judge, an existing evaluator, ...). Record that
procedure; the tool consumes the resulting 0/1 label and, if you have one, a small
human-labelled subset to correct it.

## 3. What is computed (formulas)

With `x_i` the count of idea `i`, `n = sum x_i`, `S_obs` the number of observed ideas,
`f1` / `f2` the number of ideas seen exactly once / twice:

| quantity | formula | column in `summary.csv` |
|---|---|---|
| Chao1 lower bound | `N_hat = S_obs + f1 (f1 - 1) / (2 (f2 + 1))` | `N_hat` (and `f0_hat = N_hat - S_obs`) |
| sample coverage | `C = 1 - f1 / n` | `coverage` |
| idea probability | `pi_i = C x_i / n` | `pi_hat` in `idea_table_*.csv` |
| unseen mass | `M0 = 1 - C = f1 / n` | `M0` |
| unseen tail scale | `pi_0 = M0 / f0_hat` (NaN when `f0_hat = 0`) | `pi0` |
| modal idea | `argmax pi_i` | `modal_idea`, `modal_pi` |
| top-k mass | `sum of the k largest pi_i` | `top1_mass`, `top3_mass`, ... (`--top-k`) |
| detection probability | `q_i = 1 - (1 - pi_i)^n` | `q_detect` in idea table |
| discovery curve | `E[K_m] = sum_i 1 - (1 - pi_i)^m` | `discovery_curve_*.csv`, `fig_discovery_curve.png` |
| valuable indicator | `z_i` = your label (1 valuable, 0 not); `r_i` = calibrated `Pr(valuable)` (= `z_i` without gold) | `valuable`, `r_valuable` in idea table |
| observed valuable ideas | `N_obs_V = sum_obs r_i` (and the hard count `S_V_obs = #{z_i = 1}`) | `N_obs_valuable`, `S_V_obs` |
| observed valuable ratio | `raw_ratio = N_obs_V / S_obs` (no unseen assumption) | `raw_ratio` |
| unseen valuable fraction | `frac_u = mean r_i over singletons` (rare ideas proxy the unseen) | `frac_unseen_singleton` (+ `frac_unseen_observed`) |
| valuable breadth | `N_V = N_obs_V + f0_hat * frac_u` | `N_V` |
| bounds | `N_V_lower = N_obs_V` (no unseen idea valuable); `N_V_upper = N_obs_V + f0_hat` (all unseen valuable) | `N_V_lower`, `N_V_upper` |
| valuable fraction of the space | `N_V / N_hat` (and the same for the bounds) | `valuable_fraction`, `_lower`, `_upper` |
| observed valuable mass | `Q_V = sum_obs pi_i r_i` (pi from the full sample, never renormalised) | `Q_V` |
| total valuable mass | `P_V = Q_V + M0 * frac_u` | `P_V` |
| pi within the valuable space | `pi_i r_i / P_V` for valuable ideas, NaN otherwise | `pi_within_valuable` in idea table |

**Value is a filter, not a re-estimation.** Capture probabilities `pi_i`, coverage `C`,
`N_hat` and `f0_hat` are all computed once on the full, unfiltered sample; the value
label only selects which ideas count. Do *not* re-run Chao1 on the valuable subset
alone: when value correlates with generation probability the filtered frequency counts
no longer describe a random sample from the valuable sub-space and the estimate is biased.
`N_V` is therefore additive: what you saw, plus a declared assumption about the unseen part.
Always report `N_V` together with its bounds and `raw_ratio`; the bounds are the only
assumption-free statement about unseen valuable ideas. `--unseen-rule observed` swaps the
singleton fraction for the share among all observed ideas; both are always in the table.

### 3.1 Correcting an automatic label with a small human-labelled subset

If `valuable` comes from an LLM judge or another automatic evaluator, label a small
random subset of ideas by hand and pass it as `--gold-col`. Sample within each label
stratum (some ideas the judge called 1, some it called 0); a stratified random sample is
fine, a random sample of all ideas is fine, a hand-picked sample is not. The tool then:

1. estimates the judge's precision in each stratum on the labelled ideas:
   `PPV = Pr(human = 1 | z = 1)`, `NPV = Pr(human = 0 | z = 0)`;
2. sets `r_i = gold_i` where a human looked, else `r_i = PPV` for `z_i = 1` and
   `r_i = 1 - NPV` for `z_i = 0`;
3. runs every V formula above with `r_i` in place of `z_i`.

`summary.csv` reports `calib_n_gold`, `calib_n_gold_z1`, `calib_n_gold_z0`, `calib_PPV`,
`calib_NPV` and `calib_agreement`; both strata need at least one human label or the run
stops. With few human labels the PPV/NPV are themselves noisy: say how many you used.
Without `--gold-col`, `r_i = z_i` and the tool trusts your label as given.

Extras that are *not* in the task description but help reporting:

* `valuable_sample_share` — `sum x_i r_i / n`, the fraction of generations that produced
  a valuable idea (the "hit rate" a user experiences; compare with `Q_V`, which is
  coverage-adjusted).
* `N_V_observed_rule` — `N_V` under the alternative unseen assumption, so the two rules
  can be compared without a second run.
* `N_hat_ci95_lo/hi`, `coverage_ci95_lo/hi` — percentile intervals from a
  population-reconstruction bootstrap (`--n-boot`, default 200; `0` disables). The
  interval describes uncertainty *in the lower bound*, not a two-sided interval for N.
* `rarefaction` column of the discovery curve — exact expected distinct ideas in a
  random subsample of size m <= n (hypergeometric); the `model_EK` column is the
  task-description formula evaluated on the reconstructed landscape
  (observed `pi_i` plus `round(f0_hat)` unseen ideas each at `pi_0`) and is the only
  curve that extends beyond n (`--extrapolate-to`, default 2n).
* `model_EK_at_n_minus_S_obs` — gap between the model curve at m = n and the observed
  richness; a large gap means the (pi_i, pi_0) reconstruction describes the sample poorly.
* `empirical_accumulation_*.csv` (samples format) — mean cumulative distinct count over
  `--n-perm` random orderings of the actual samples; should track `rarefaction`.

## 4. Output files per run

```
resolved_config.json            all arguments as actually used + toolkit version
run.log                         same messages as the console, incl. warnings
summary.csv / summary.json      one row per condition, sorted by N_hat ascending (worst first)
idea_table_<condition>.csv      idea_id, count, pi_hat, rank, q_detect [, valuable, gold, r_valuable, pi_within_valuable]
discovery_curve_<condition>.csv m, rarefaction (m <= n), model_EK (all m)
empirical_accumulation_<c>.csv  (samples format only)
fig_discovery_curve.png         all conditions overlaid; dot = S_obs at n, dash-dot = N_hat
fig_rank_probability.png        rank vs pi_i (log y); dashed line = pi_0
```

`idea_table_<condition>.csv` is the "idea-level table containing counts, pi_i ... and
value" deliverable; join your raw value ratings and semantic columns (cluster / region /
embedding coordinates) onto it downstream by `idea_id`.

## 5. Reading the numbers (and what they cannot tell you)

* **N_hat is a lower bound.** Report it together with `S_obs`, `coverage`, and the
  discovery curve, never alone. When `f1 = 0` the estimate collapses to `S_obs`.
* **pi_i sums to C, not 1.** The remaining mass `M0` belongs to unseen ideas; `pi_0` is
  only an average scale for that tail, not a claim that unseen ideas are equally likely.
* **Low coverage (C < 0.5) is flagged** in `run.log`; rare-idea probabilities and
  everything built on them (q_i, r_V, N_V) are then very uncertain.
* **N_V rests on one declared assumption**: unseen ideas are valuable at the rate seen
  among singletons (`frac_unseen_singleton`). With fewer than 10 singletons the log
  warns that this fraction is unstable; lean on `N_V_lower` / `N_V_upper` then.
* **Unseen ideas are never given a measured value.** `N_V_upper - N_V_lower = f0_hat`
  is the honest width of what the sample cannot tell you.
* **Independence is assumed.** Each row in samples format must be one complete,
  independently reset replication (see "Sampling" in the task description).
* **Idea identification errors propagate.** False splits inflate `f1` (and hence
  `N_hat`, `M0`); false merges deflate them. Validate the idea labels first.
* Semantic distance (within/between-region spread) is out of scope for this tool.

## 6. Using the functions directly

```python
import numpy as np
from npv import estimate_N, estimate_P, estimate_V, discovery_curves

counts = np.array([5, 3, 1, 1, 2, 1])          # x_i for 6 observed ideas, n = 13
estimate_N(counts).N_hat                         # 7.5
estimate_P(counts).pi_hat                        # C * x_i / n with C = 10/13
v = estimate_V(counts, valuable=[1, 0, 1, 0, 1, 1])      # labels, one per idea
v.N_V, v.N_V_lower, v.N_V_upper                          # 5.0, 4.0, 5.5
estimate_V(counts, [1, 0, 1, 0, 1, 1], gold=[1, 0, 0, None, 1, None]).calibration   # PPV/NPV
discovery_curves(counts, extrapolate_to=50)["model"]
```

`tests/test_estimators.py` contains the same worked example with every number checked
by hand — read it if you want to verify a formula.

## 7. References

* Chao, A. (1984). Nonparametric estimation of the number of classes in a population. *Scand. J. Stat.* 11, 265-270.
* Chao, A. & Shen, T.-J. (2003). Nonparametric estimation of Shannon's index of diversity when there are unseen species in sample. *Environ. Ecol. Stat.* 10, 429-443.
* Good, I. J. (1953). The population frequencies of species and the estimation of population parameters. *Biometrika* 40, 237-264.
* Chao, A. et al. (2014). Rarefaction and extrapolation with Hill numbers. *Ecol. Monogr.* 84, 45-67 (bootstrap and rarefaction formulas).
