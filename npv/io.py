"""Loading idea tables and splitting them by condition.

Two input formats are accepted (both CSV / TSV):

``samples``  one row per independent generation
             required: idea column      (the idea category the sample was assigned to)
             optional: condition columns, value column (per-sample value; aggregated
             to the idea level by the mean - every sample of one idea should carry
             the same idea-level value, a warning is logged otherwise)

``counts``   one row per observed idea
             required: idea column, count column (x_i >= 1)
             optional: condition columns, value column (idea-level value)

Every condition group becomes one ``ConditionData`` with aligned arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("npv")


@dataclass
class ConditionData:
    label: str                       # "modelA|promptX" (joined with |)
    keys: dict[str, str]             # {condition_col: value}
    idea_ids: np.ndarray             # object array, one per observed idea
    counts: np.ndarray               # int64, x_i
    values: np.ndarray | None        # float per idea, or None
    sample_labels: np.ndarray | None # per-sample idea ids (samples format only)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep)
    if df.empty:
        raise ValueError(f"{path} has no rows")
    return df


def _require_columns(df: pd.DataFrame, cols: list[str], path: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}; available: {list(df.columns)}"
        )


def _check_no_missing(df: pd.DataFrame, cols: list[str], path: str) -> None:
    for c in cols:
        n_missing = int(df[c].isna().sum())
        if n_missing:
            raise ValueError(f"{path}: column '{c}' has {n_missing} missing value(s); fix the data")


def load_conditions(
    path: str | Path,
    fmt: str,
    idea_col: str,
    condition_cols: list[str],
    value_col: str | None,
    count_col: str = "count",
) -> list[ConditionData]:
    path = Path(path)
    df = read_table(path)
    needed = [idea_col] + condition_cols + ([value_col] if value_col else [])
    if fmt == "counts":
        needed.append(count_col)
    elif fmt != "samples":
        raise ValueError(f"unknown format '{fmt}' (expected 'samples' or 'counts')")
    _require_columns(df, needed, str(path))
    _check_no_missing(df, needed, str(path))

    if condition_cols:
        groups = list(df.groupby(condition_cols, sort=True))
    else:
        groups = [((), df)]

    out: list[ConditionData] = []
    for key, g in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        keys = {c: str(k) for c, k in zip(condition_cols, key_tuple)}
        label = "|".join(keys.values()) if keys else "all"
        if fmt == "samples":
            out.append(_from_samples(g, label, keys, idea_col, value_col, str(path)))
        else:
            out.append(_from_counts(g, label, keys, idea_col, count_col, value_col, str(path)))
    log.info("loaded %d condition(s) from %s (%s format)", len(out), path, fmt)
    return out


def _from_samples(g, label, keys, idea_col, value_col, path) -> ConditionData:
    ideas = g[idea_col].astype(str).to_numpy()
    counts_s = pd.Series(ideas).value_counts(sort=False)
    idea_ids = counts_s.index.to_numpy(dtype=object)
    counts = counts_s.to_numpy(dtype=np.int64)
    values = None
    if value_col:
        v = pd.to_numeric(g[value_col], errors="raise")
        per_idea = v.groupby(ideas).agg(["mean", "min", "max"])
        inconsistent = per_idea[(per_idea["max"] - per_idea["min"]) > 1e-12]
        if len(inconsistent):
            log.warning(
                "[%s] %d idea(s) have varying per-sample values; using the per-idea MEAN. "
                "The task description defines value at the idea level - consider "
                "rating each idea once. Example ideas: %s",
                label, len(inconsistent), list(inconsistent.index[:5]),
            )
        values = per_idea.loc[idea_ids, "mean"].to_numpy(dtype=float)
    return ConditionData(label, keys, idea_ids, counts, values, ideas)


def _from_counts(g, label, keys, idea_col, count_col, value_col, path) -> ConditionData:
    ideas = g[idea_col].astype(str).to_numpy(dtype=object)
    if len(set(ideas)) != len(ideas):
        dup = pd.Series(ideas)[pd.Series(ideas).duplicated()].unique()[:5]
        raise ValueError(f"{path} [{label}]: duplicate idea rows in counts format, e.g. {list(dup)}")
    raw = pd.to_numeric(g[count_col], errors="raise").to_numpy()
    if np.any(np.mod(raw, 1) != 0) or np.any(raw < 1):
        raise ValueError(f"{path} [{label}]: '{count_col}' must be positive integers")
    counts = raw.astype(np.int64)
    values = pd.to_numeric(g[value_col], errors="raise").to_numpy(dtype=float) if value_col else None
    return ConditionData(label, keys, ideas, counts, values, None)
