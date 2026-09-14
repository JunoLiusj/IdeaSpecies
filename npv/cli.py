"""Command-line entry point: ``uv run npv-estimate --input ... ``.

Writes a fresh, timestamped run directory (never overwrites) containing
resolved_config.json, run.log, summary.csv / summary.json, and per-condition
idea tables, discovery curves, rank-probability tables and figures."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .estimators import (
    chao1_bootstrap, discovery_curves, empirical_accumulation, estimate_N, estimate_P, estimate_V,
)
from .io import ConditionData, load_conditions
from .plots import plot_discovery, plot_rank_probability

log = logging.getLogger("npv")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="npv-estimate",
        description="Estimate N (richness), P (probability landscape) and V (value landscape) "
                    "for one or more conditions, using the formulas in the Stage 2 RA task description.",
    )
    p.add_argument("--input", required=True, help="CSV/TSV file")
    p.add_argument("--format", required=True, choices=["samples", "counts"],
                   help="samples: one row per generation; counts: one row per observed idea")
    p.add_argument("--idea-col", default="idea_id", help="column holding the idea category id")
    p.add_argument("--count-col", default="count", help="(counts format) column with x_i")
    p.add_argument("--condition-cols", default="",
                   help="comma-separated columns defining a condition, e.g. model,prompt (empty = single condition)")
    p.add_argument("--valuable-col", default=None,
                   help="column with the 0/1 (or true/false) gold valuable label of each idea; enables the V block")
    p.add_argument("--top-k", default="1,3,5,10", help="k values for top-k probability mass")
    p.add_argument("--extrapolate-to", type=int, default=None,
                   help="extend the model discovery curve to this many samples (default 2n per condition)")
    p.add_argument("--n-boot", type=int, default=200,
                   help="bootstrap replicates for the N_hat / coverage interval (0 disables)")
    p.add_argument("--n-perm", type=int, default=100,
                   help="(samples format) random orderings for the empirical accumulation curve (0 disables)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root", default="results", help="parent directory for run folders")
    p.add_argument("--run-name", default=None, help="run folder name (default: npv_<timestamp>)")
    return p


def _setup_logging(run_dir: Path) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(run_dir / "run.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def _safe(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)


def analyse_condition(cd: ConditionData, args, top_k: list[int]) -> tuple[dict, pd.DataFrame, dict, pd.DataFrame, np.ndarray | None]:
    n_est = estimate_N(cd.counts)
    for w in n_est.warnings:
        log.warning("[%s] %s", cd.label, w)
    p_est = estimate_P(cd.counts, top_k=top_k)

    row: dict = {**cd.keys, "condition": cd.label}
    row.update({k: v for k, v in n_est.to_dict().items() if k != "warnings"})
    row.update({k: v for k, v in p_est.to_dict().items()
                if k not in {"n", "coverage", "M0", "f0_hat", "modal_index", "observed_mass"}})
    row["modal_idea"] = str(cd.idea_ids[p_est.modal_index])
    row["n_warnings"] = len(n_est.warnings)

    idea_tbl = pd.DataFrame({
        "idea_id": cd.idea_ids, "count": cd.counts, "pi_hat": p_est.pi_hat, "rank": p_est.rank,
        "q_detect": 1.0 - (1.0 - p_est.pi_hat) ** p_est.n,
    })
    if cd.valuable is not None:
        v_est = estimate_V(cd.counts, cd.valuable)
        idea_tbl["valuable"] = v_est.z.astype(int)
        idea_tbl["idw_weight"] = 1.0 / v_est.q          # inverse-detection weight used in r_V
        row.update(v_est.to_dict())
    idea_tbl = idea_tbl.sort_values("rank").reset_index(drop=True)

    if args.n_boot > 0:
        boot = chao1_bootstrap(cd.counts, n_boot=args.n_boot, seed=args.seed)
        row["N_hat_ci95_lo"], row["N_hat_ci95_hi"] = boot["N_hat_ci95"]
        row["coverage_ci95_lo"], row["coverage_ci95_hi"] = boot["coverage_ci95"]
        row["N_hat_boot_sd"] = boot["N_hat_boot_sd"]

    curves = discovery_curves(cd.counts, extrapolate_to=args.extrapolate_to)
    curve_tbl = pd.DataFrame({"m": curves["m_grid"], "rarefaction": curves["rarefaction"],
                              "model_EK": curves["model"]})
    row["model_EK_at_n"] = float(np.interp(curves["n"], curves["m_grid"], curves["model"]))
    row["model_EK_at_n_minus_S_obs"] = row["model_EK_at_n"] - n_est.S_obs

    empirical = None
    if cd.sample_labels is not None and args.n_perm > 0:
        empirical = empirical_accumulation(cd.sample_labels, n_perm=args.n_perm, seed=args.seed)
    return row, idea_tbl, curves, curve_tbl, empirical


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    condition_cols = [c.strip() for c in args.condition_cols.split(",") if c.strip()]
    top_k = [int(k) for k in args.top_k.split(",") if k.strip()]

    run_name = args.run_name or f"npv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out_root) / run_name
    if run_dir.exists():
        raise SystemExit(f"run directory {run_dir} already exists; choose another --run-name (never overwrite)")
    run_dir.mkdir(parents=True)
    _setup_logging(run_dir)
    resolved = {**vars(args), "condition_cols": condition_cols, "top_k": top_k,
                "toolkit_version": __version__, "started": datetime.now().isoformat()}
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2))
    log.info("run dir %s", run_dir)

    conditions = load_conditions(args.input, args.format, args.idea_col, condition_cols,
                                 args.valuable_col, args.count_col)
    rows, all_curves, all_rank, all_emp = [], {}, {}, {}
    for cd in conditions:
        row, idea_tbl, curves, curve_tbl, emp = analyse_condition(cd, args, top_k)
        rows.append(row)
        tag = _safe(cd.label)
        idea_tbl.to_csv(run_dir / f"idea_table_{tag}.csv", index=False)
        curve_tbl.to_csv(run_dir / f"discovery_curve_{tag}.csv", index=False)
        rank_tbl = idea_tbl[["rank", "idea_id", "count", "pi_hat"]].copy()
        rank_tbl.attrs["pi0"] = row["pi0"]
        all_curves[cd.label], all_rank[cd.label] = curves, rank_tbl
        if emp is not None:
            all_emp[cd.label] = emp
            pd.DataFrame({"m": np.arange(1, len(emp) + 1), "empirical_distinct": emp}).to_csv(
                run_dir / f"empirical_accumulation_{tag}.csv", index=False)
        log.info("[%s] n=%d S_obs=%d f1=%d f2=%d N_hat=%.2f C=%.3f pi0=%s",
                 cd.label, row["n"], row["S_obs"], row["f1"], row["f2"], row["N_hat"], row["coverage"],
                 f"{row['pi0']:.4g}" if row["pi0"] == row["pi0"] else "nan")

    summary = pd.DataFrame(rows).sort_values("N_hat", ascending=True).reset_index(drop=True)
    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=float))
    plot_discovery(all_curves, run_dir / "fig_discovery_curve.png", empirical=all_emp or None)
    plot_rank_probability(all_rank, run_dir / "fig_rank_probability.png")
    log.info("wrote %d condition(s); summary.csv sorted by N_hat ascending (worst first)", len(rows))
    return run_dir


if __name__ == "__main__":
    main()
