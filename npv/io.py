"""Loading idea tables and splitting them by condition.

Two input formats are accepted (both CSV / TSV):

``samples``  one row per independent generation
             required: idea column      (the idea category the sample was assigned to)
             optional: condition columns; valuable column (final 0/1 label of the idea,
             must be identical on every sample of the same idea - the run stops otherwise)

``counts``   one row per observed idea
             required: idea column, count column (x_i >= 1)
             optional: condition columns, valuable column (final 0/1 label per idea)

Every condition group becomes one ``ConditionData`` with aligned arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .estimators import as_valuable_labels

log = logging.getLogger("npv")


@dataclass
class ConditionData:
    label: str                       # "modelA|promptX" (joined with |)
    keys: dict[str, str]             # {condition_col: value}
    idea_ids: np.ndarray             # object array, one per observed idea
    counts: np.ndarray               # int64, x_i
    valuable: np.ndarray | None      # 0/1 float per idea, or None
    sample_labels: np.ndarray | None # per-sample idea ids (samples format only)
    coords: np.ndarray | None = None # (S_obs, 2) semantic coordinates per idea, or None


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
    valuable_col: str | None,
    count_col: str = "count",
    coord_cols: list[str] | None = None,
) -> list[ConditionData]:
    path = Path(path)
    df = read_table(path)
    coord_cols = list(coord_cols or [])
    if coord_cols and len(coord_cols) != 2:
        raise ValueError(f"coord columns must be exactly two (x, y), got {coord_cols}")
    needed = [idea_col] + condition_cols + ([valuable_col] if valuable_col else []) + coord_cols
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
            cd = _from_samples(g, label, keys, idea_col, valuable_col, str(path))
        else:
            cd = _from_counts(g, label, keys, idea_col, count_col, valuable_col, str(path))
        if coord_cols:
            cd.coords = _idea_coords(g, cd.idea_ids, idea_col, coord_cols, label, str(path))
        out.append(cd)
    log.info("loaded %d condition(s) from %s (%s format)", len(out), path, fmt)
    return out


def _idea_coords(g, idea_ids, idea_col, coord_cols, label, path) -> np.ndarray:
    """One (x, y) per idea; in samples format every sample of an idea must carry the same point."""
    ideas = g[idea_col].astype(str).to_numpy()
    xy = g[coord_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    agg = pd.DataFrame(xy, columns=["x", "y"]).groupby(ideas).agg(["min", "max"])
    spread = (agg[("x", "max")] - agg[("x", "min")]).abs() + (agg[("y", "max")] - agg[("y", "min")]).abs()
    bad = spread[spread > 1e-9]
    if len(bad):
        raise ValueError(f"{path} [{label}]: {len(bad)} idea(s) have differing coordinates across samples, "
                         f"e.g. {list(bad.index[:5])}. Coordinates are a property of the idea.")
    return np.column_stack([agg.loc[idea_ids, ("x", "min")].to_numpy(), agg.loc[idea_ids, ("y", "min")].to_numpy()])


def _from_samples(g, label, keys, idea_col, valuable_col, path) -> ConditionData:
    ideas = g[idea_col].astype(str).to_numpy()
    counts_s = pd.Series(ideas).value_counts(sort=False)
    idea_ids = counts_s.index.to_numpy(dtype=object)
    counts = counts_s.to_numpy(dtype=np.int64)
    valuable = None
    if valuable_col:
        z = as_valuable_labels(g[valuable_col].to_numpy())
        per_idea = pd.Series(z).groupby(ideas).agg(["min", "max"])
        inconsistent = per_idea[per_idea["max"] != per_idea["min"]]
        if len(inconsistent):
            raise ValueError(
                f"{path} [{label}]: {len(inconsistent)} idea(s) carry both valuable=1 and "
                f"valuable=0 across their samples, e.g. {list(inconsistent.index[:5])}. "
                "The valuable label is a property of the idea - label each idea once."
            )
        valuable = per_idea.loc[idea_ids, "max"].to_numpy(dtype=float)
    return ConditionData(label, keys, idea_ids, counts, valuable, ideas)


def _from_counts(g, label, keys, idea_col, count_col, valuable_col, path) -> ConditionData:
    ideas = g[idea_col].astype(str).to_numpy(dtype=object)
    if len(set(ideas)) != len(ideas):
        dup = pd.Series(ideas)[pd.Series(ideas).duplicated()].unique()[:5]
        raise ValueError(f"{path} [{label}]: duplicate idea rows in counts format, e.g. {list(dup)}")
    raw = pd.to_numeric(g[count_col], errors="raise").to_numpy()
    if np.any(np.mod(raw, 1) != 0) or np.any(raw < 1):
        raise ValueError(f"{path} [{label}]: '{count_col}' must be positive integers")
    counts = raw.astype(np.int64)
    valuable = as_valuable_labels(g[valuable_col].to_numpy()) if valuable_col else None
    return ConditionData(label, keys, ideas, counts, valuable, None)
