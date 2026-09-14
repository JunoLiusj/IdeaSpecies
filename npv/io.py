"""Loading idea tables and splitting them by condition.

Two input formats are accepted (both CSV / TSV):

``samples``  one row per independent generation
             required: idea column      (the idea category the sample was assigned to)
             optional: condition columns; valuable column (0/1 label of the idea, must be
             identical on every sample of the same idea); gold column (human 0/1 label on
             a subset of ideas, blank elsewhere, also identical within an idea)

``counts``   one row per observed idea
             required: idea column, count column (x_i >= 1)
             optional: condition columns, valuable column, gold column (blank = unlabelled)

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
    gold: np.ndarray | None          # 0/1 float per idea with nan where unlabelled, or None
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


def _partial_labels(series: pd.Series) -> np.ndarray:
    """0/1 float array with nan where the entry is blank."""
    out = np.full(len(series), np.nan)
    present = series.notna() & (series.astype(str).str.strip() != "")
    if present.any():
        out[present.to_numpy()] = as_valuable_labels(series[present].to_numpy())
    return out


def load_conditions(
    path: str | Path,
    fmt: str,
    idea_col: str,
    condition_cols: list[str],
    valuable_col: str | None,
    count_col: str = "count",
    gold_col: str | None = None,
) -> list[ConditionData]:
    path = Path(path)
    df = read_table(path)
    if gold_col and not valuable_col:
        raise ValueError("--gold-col needs --valuable-col: the gold subset calibrates the valuable label")
    needed = [idea_col] + condition_cols + ([valuable_col] if valuable_col else [])
    if fmt == "counts":
        needed.append(count_col)
    elif fmt != "samples":
        raise ValueError(f"unknown format '{fmt}' (expected 'samples' or 'counts')")
    _require_columns(df, needed + ([gold_col] if gold_col else []), str(path))
    _check_no_missing(df, needed, str(path))          # gold may be blank: it is a partial column

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
            out.append(_from_samples(g, label, keys, idea_col, valuable_col, gold_col, str(path)))
        else:
            out.append(_from_counts(g, label, keys, idea_col, count_col, valuable_col, gold_col, str(path)))
    log.info("loaded %d condition(s) from %s (%s format)", len(out), path, fmt)
    return out


def _per_idea_label(values: np.ndarray, ideas: np.ndarray, idea_ids: np.ndarray, name: str,
                    label: str, path: str) -> np.ndarray:
    """Collapse per-sample 0/1 (or nan) labels to one label per idea; nan stays nan only
    when no sample of the idea was labelled; mixed 0 and 1 within an idea is an error."""
    s = pd.Series(values).groupby(ideas).agg(["min", "max"])
    mixed = s[(s["max"] != s["min"]) & s["min"].notna()]
    if len(mixed):
        raise ValueError(
            f"{path} [{label}]: {len(mixed)} idea(s) carry both {name}=1 and {name}=0 across their "
            f"samples, e.g. {list(mixed.index[:5])}. The label is a property of the idea - label each idea once."
        )
    return s.loc[idea_ids, "max"].to_numpy(dtype=float)


def _from_samples(g, label, keys, idea_col, valuable_col, gold_col, path) -> ConditionData:
    ideas = g[idea_col].astype(str).to_numpy()
    counts_s = pd.Series(ideas).value_counts(sort=False)
    idea_ids = counts_s.index.to_numpy(dtype=object)
    counts = counts_s.to_numpy(dtype=np.int64)
    valuable = gold = None
    if valuable_col:
        valuable = _per_idea_label(as_valuable_labels(g[valuable_col].to_numpy()), ideas, idea_ids,
                                   "valuable", label, path)
    if gold_col:
        gold = _per_idea_label(_partial_labels(g[gold_col]), ideas, idea_ids, "gold", label, path)
    return ConditionData(label, keys, idea_ids, counts, valuable, gold, ideas)


def _from_counts(g, label, keys, idea_col, count_col, valuable_col, gold_col, path) -> ConditionData:
    ideas = g[idea_col].astype(str).to_numpy(dtype=object)
    if len(set(ideas)) != len(ideas):
        dup = pd.Series(ideas)[pd.Series(ideas).duplicated()].unique()[:5]
        raise ValueError(f"{path} [{label}]: duplicate idea rows in counts format, e.g. {list(dup)}")
    raw = pd.to_numeric(g[count_col], errors="raise").to_numpy()
    if np.any(np.mod(raw, 1) != 0) or np.any(raw < 1):
        raise ValueError(f"{path} [{label}]: '{count_col}' must be positive integers")
    counts = raw.astype(np.int64)
    valuable = as_valuable_labels(g[valuable_col].to_numpy()) if valuable_col else None
    gold = _partial_labels(g[gold_col]) if gold_col else None
    return ConditionData(label, keys, ideas, counts, valuable, gold, None)
