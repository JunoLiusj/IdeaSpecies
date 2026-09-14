# RA N-P-V Estimation Toolkit

A small, dependency-light tool that computes every estimator written in the
**Stage 2 Task for RAs** document ("The N-P-V framework" section) from an idea-level
table. It does *not* identify ideas, embed text, or call any model:
you bring a table where each independent generation has already been assigned to
one idea category (and, optionally, each idea carries a binary label saying whether
it is valuable, obtained however your study defines value).

**This toolkit is a reference implementation, not the analysis.** It fixes the
formulas so that every RA computes N, P and V the same way; everything around them
(the hypothesis, the conditions you compare, the idea-identification and value
procedures, the robustness checks, and above all the figures) is yours to design. Fork
it, add columns, change the plots, or reimplement the formulas in your own pipeline.
The figures it writes are deliberately plain starting points: tailor them to the
regularity you want the reader to see.

```
N  richness             bias-corrected Chao1 lower bound, sample coverage, discovery curve
P  probability landscape coverage-adjusted pi_i, unseen mass M0, f0, pi_0, modal idea,
                         top-k mass, rank-probability curve
V  value landscape      valuable filter z_i, N_V with bounds, valuable mass Q_V / P_V,
                         pi within the valuable space
```

## 1. Install and run (uv)

```bash
git clone https://github.com/JunoLiusj/IdeaSpecies && cd IdeaSpecies
uv sync                                  # creates .venv with numpy / pandas / matplotlib
uv run pytest -q                         # 14 hand-checked tests should pass

# synthetic example with known truth (two conditions, 300 samples each)
uv run python examples/make_example_data.py
uv run npv-estimate --input examples/example_samples.csv --format samples \
    --condition-cols condition --valuable-col valuable --coord-cols x,y \
    --out-root examples/results --run-name demo
```

Each run writes a **new** folder `<out-root>/<run-name>/` and refuses to overwrite an
existing one. Without `--run-name` the folder is `npv_<timestamp>`.

## 2. Input requirements

One CSV (or TSV, by file extension) with a header row, UTF-8, one table per run. The
table may hold several conditions; the tool splits it by the condition columns you name.
Column names are free, you pass them on the command line.

**What a row must be.** In `samples` format each row is **one complete, independently
reset replication** of your sampling unit (one completion, one conversation, one
generate-critique-revise trajectory, ...), exactly as the task description defines it.
Do not put the turns of one conversation, or the retries of one call, in separate rows.
Sampling budgets should be comparable across conditions (same `n`, or report `n`).

**Idea ids.** Every row carries the id of the substantive idea category it was assigned
to, from *your* idea identification step (LLM judge or embedding clustering, checked on
a human-labelled subset). For comparisons across conditions the ids must come from one
common idea definition built on the pooled, blinded outputs: the same string means the
same idea in every condition. Ids are treated as opaque strings.

**Minimal `samples` table** (`--format samples --idea-col idea_id --condition-cols model`):

```csv
sample_id,model,idea_id,valuable,x,y
s0001,llama70b,I-017,1,0.42,-1.10
s0002,llama70b,I-003,0,-0.85,0.20
s0003,gemini25,I-017,1,0.42,-1.10
```

**Minimal `counts` table** (`--format counts --count-col count --condition-cols model`):

```csv
model,idea_id,count,valuable,x,y
llama70b,I-017,12,1,0.42,-1.10
llama70b,I-003,1,0,-0.85,0.20
gemini25,I-017,4,1,0.42,-1.10
```

Only `idea_id` (plus `count` in counts format) is required; `valuable`, `x`, `y` and the
condition columns are optional and switch on the corresponding analyses.

**Checklist before running**

- [ ] one row = one independent replication (samples) or one observed idea (counts)
- [ ] no missing values in any column you name on the command line
- [ ] `count` values are positive integers; ideas with count 0 are not listed
- [ ] the same idea id never appears twice within a condition in counts format
- [ ] `valuable` is 0/1 (or true/false, yes/no) and identical on every row of an idea
- [ ] `x`, `y` are identical on every row of an idea and were computed on the pooled ideas of all conditions
- [ ] condition columns identify the frozen condition (model, prompt, sampling setting, protocol); anything that varies within a condition must be part of the treatment

Anything that violates the checklist stops the run with an explicit error rather than
being silently repaired.

**`--format samples`** — one row per independent generation (the recommended form;
it also enables the empirical accumulation curve).

| column (default name) | required | meaning |
|---|---|---|
| `idea_id` (`--idea-col`) | yes | idea category the sample was assigned to |
| condition columns (`--condition-cols model,prompt`) | no | one estimate per unique combination |
| `valuable` (`--valuable-col`) | no | the idea's FINAL valuable label from your procedure (section 3.1): `1`/`0`, `true`/`false` or `yes`/`no`; must be identical for all samples of one idea (the run stops otherwise) |
| `x`, `y` (`--coord-cols x,y`) | no | 2-D semantic coordinates of the idea (e.g. MDS / UMAP / PCA of the idea embedding, computed by you on the pooled ideas of all conditions); identical within an idea; enables the density landscape figure |

**`--format counts`** — one row per *observed* idea.

| column | required | meaning |
|---|---|---|
| `idea_id` | yes | idea id (unique within a condition) |
| `count` (`--count-col`) | yes | x_i, number of samples that produced the idea (>= 1) |
| condition columns | no | as above |
| `valuable` | no | idea-level FINAL label, `1`/`0` (or `true`/`false`, `yes`/`no`) |
| `x`, `y` | no | 2-D semantic coordinates of the canonical idea of this idea species |

Do **not** include rows with count 0: unobserved ideas are exactly what N estimates.
Missing values, non-integer counts, duplicate idea rows, or labels other than 0/1
stop the run with an explicit error (no silent defaults).

The tool takes the valuable label as given. Producing it, checking it against human
judgement, and correcting it are part of your study design (section 3.1).

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
| valuable indicator | `z_i` = your final label (1 valuable, 0 not) | `valuable` in idea table |
| observed valuable ideas | `S_V_obs = #{z_i = 1}` | `S_V_obs` |
| observed valuable ratio | `raw_ratio = S_V_obs / S_obs` (no unseen assumption) | `raw_ratio` |
| unseen valuable fraction | `frac_u = mean z_i over singletons` (rare ideas proxy the unseen) | `frac_unseen_singleton` (+ `frac_unseen_observed`) |
| valuable breadth | `N_V = S_V_obs + f0_hat * frac_u` | `N_V` |
| bounds | `N_V_lower = S_V_obs` (no unseen idea valuable); `N_V_upper = S_V_obs + f0_hat` (all unseen valuable) | `N_V_lower`, `N_V_upper` |
| valuable fraction of the space | `N_V / N_hat` (and the same for the bounds) | `valuable_fraction`, `_lower`, `_upper` |
| observed valuable mass | `Q_V = sum_obs pi_i z_i` (pi from the full sample, never renormalised) | `Q_V` |
| total valuable mass | `P_V = Q_V + M0 * frac_u` | `P_V` |
| pi within the valuable space | `pi_i / P_V` for valuable ideas, NaN otherwise | `pi_within_valuable` in idea table |

**Value is a filter, not a re-estimation.** Capture probabilities `pi_i`, coverage `C`,
`N_hat` and `f0_hat` are all computed once on the full, unfiltered sample; the value
label only selects which ideas count. Do *not* re-run Chao1 on the valuable subset
alone: when value correlates with generation probability the filtered frequency counts
no longer describe a random sample from the valuable sub-space and the estimate is biased.
`N_V` is therefore additive: what you saw, plus a declared assumption about the unseen part.
Always report `N_V` together with its bounds and `raw_ratio`; the bounds are the only
assumption-free statement about unseen valuable ideas. `--unseen-rule observed` swaps the
singleton fraction for the share among all observed ideas; both are always in the table.

### 3.1 The valuable label, its human check, and its correction are yours to design

The tool does not calibrate labels. Whatever produces `valuable` (blinded human ratings
with a preregistered threshold, an LLM judge, an existing evaluator), you are expected to:

1. **Label a human subset.** Draw a random subset of *ideas* (not samples) before
   looking at results, have blinded humans label it with the same rubric, and freeze it.
   A hand-picked subset is not acceptable.
2. **Correct the automatic label against the human labels**, in a way you decide and
   document (replace the automatic label wherever a human looked, adjust the labels or
   the counts by what the human subset shows, fit a calibration model, ...). Preregister
   the choice if you can.
3. **Report how reliable the corrected label is**, following the same reliability
   reporting you used at the Stage 1 checkpoint (the human-labelled subset, its size, the
   agreement between the automatic and the human labels, and where they systematically
   disagree). Show the uncorrected and the corrected results side by side.
4. **Feed the corrected 0/1 label back** into `--valuable-col` and rerun. If your
   correction yields corrected *counts* rather than per-idea labels, plug them into the
   same formulas by hand using `summary.csv` (`f0_hat`, `S_obs`, `N_hat`, `M0`):

   ```
   N_V       = S_V_obs_corrected + f0_hat * frac_u_corrected
   N_V_lower = S_V_obs_corrected,   N_V_upper = S_V_obs_corrected + f0_hat
   ```

   and keep `pi_i` untouched: the correction changes which ideas count as valuable,
   never their generation probability.

Extras that are *not* in the task description but help reporting:

* `valuable_sample_share` — `sum x_i z_i / n`, the fraction of generations that produced
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
idea_table_<condition>.csv      idea_id, count, pi_hat, rank, q_detect [, valuable, pi_within_valuable]
discovery_curve_<condition>.csv m, rarefaction (m <= n), model_EK (all m)
empirical_accumulation_<c>.csv  (samples format only)
fig_discovery_curve.png         all conditions overlaid; dot = S_obs at n, dash-dot = N_hat
fig_rank_probability.png        rank vs pi_i (log y); dashed line = pi_0
fig_value_landscape_rank.png    (with --valuable-col) before vs after the valuable filter, rank view
fig_value_landscape_2d.png      (with --valuable-col and --coord-cols) before vs after, density view
```

### 4.1 The value-landscape figures (before value enters vs after the valuable filter)

Both figures follow the convention of the project's own landscape figures: **value is a
filter**. Kept ideas keep their exact `pi_i`; dropped ideas are removed whole; nothing is
renormalised, so a lower or smaller "after" surface means that probability mass really
left the valuable space. Each panel prints `S_obs -> S_V_obs`, `N_hat -> N_V [lower, upper]`
and `C -> Q_V` (the share of observed mass that is valuable).

* **Rank view** (always available): every idea as a gray stem at its rank in the full
  sample; valuable ideas are recolored at the same height; dropped ideas become hollow
  markers. Shared y-axis across conditions.
* **Density view** (needs `--coord-cols`): a Gaussian kernel density on the 2-D semantic
  plane, weighted by `pi_i`. "Before" uses all ideas (integrates to `C`); "after" uses
  valuable ideas only (integrates to `Q_V`). One grid for all conditions, one bandwidth per
  condition (Scott's rule on the full sample, reused unchanged for "after"), one color
  scale for every panel. The "before" contours are repeated as dashed lines under the
  "after" surface so the eye can compare footprint and height directly.

These are starting points. Two conditions side by side already show a pattern; to make
a regularity convincing put *several* models or conditions in one figure, order them by
the factor you manipulate, and let the figure carry the comparison (a 3-D surface, a
ridge plot of `pi` by semantic region, a difference map "after minus before", a
small-multiple grid...). The tables give you every number the figures use.

`idea_table_<condition>.csv` is the "idea-level table containing counts, pi_i ... and
value" deliverable; join your raw value ratings and semantic columns (cluster / region /
embedding coordinates) onto it downstream by `idea_id`.

## 5. Reading the numbers (and what they cannot tell you)

* **One model or one example is not a regularity.** A pattern that holds in a single
  condition can be an accident of that model, prompt or task. Test every hypothesis on
  several models (or several comparable conditions of the factor you manipulate) and on
  at least two tasks; report where the direction holds and where it does not. The tool
  runs any number of conditions in one call precisely so that the comparison table and
  the overlaid figures are the default output, not an afterthought.

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
v = estimate_V(counts, valuable=[1, 0, 1, 0, 1, 1])      # final labels, one per idea
v.N_V, v.N_V_lower, v.N_V_upper                          # 5.0, 4.0, 5.5
discovery_curves(counts, extrapolate_to=50)["model"]
```

`tests/test_estimators.py` contains the same worked example with every number checked
by hand — read it if you want to verify a formula.

## 7. References

* Chao, A. (1984). Nonparametric estimation of the number of classes in a population. *Scand. J. Stat.* 11, 265-270.
* Chao, A. & Shen, T.-J. (2003). Nonparametric estimation of Shannon's index of diversity when there are unseen species in sample. *Environ. Ecol. Stat.* 10, 429-443.
* Good, I. J. (1953). The population frequencies of species and the estimation of population parameters. *Biometrika* 40, 237-264.
* Chao, A. et al. (2014). Rarefaction and extrapolation with Hill numbers. *Ecol. Monogr.* 84, 45-67 (bootstrap and rarefaction formulas).
