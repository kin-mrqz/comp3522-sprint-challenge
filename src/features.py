"""Column drops, entity ids built from descriptors, and train-only aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .entity import normalize_key


def drop_low_information(
    df: pd.DataFrame,
    missing_thresh: float = 0.95,
    protect: list[str] | tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop constant columns and columns that are almost entirely missing."""
    protected = set(protect)
    dropped = []
    for column in df.columns:
        if column in protected:
            continue
        series = df[column]
        if series.nunique(dropna=True) <= 1:
            dropped.append({"column": column, "reason": "constant"})
        elif float(series.isna().mean()) > missing_thresh:
            dropped.append({"column": column, "reason": f"missing>{missing_thresh}"})
    log = pd.DataFrame(dropped, columns=["column", "reason"])
    keep = [column for column in df.columns if column not in set(log["column"])]
    return df.loc[:, keep].copy(), log


def id_from_descriptors(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Build one entity key by joining normalized descriptor fields.

    ``A|B`` and ``a-b`` style spellings of the same parts land on the same key.
    A row whose parts are all missing gets a missing key, not the string ``"|"``.
    """
    if not columns:
        raise ValueError("id_from_descriptors needs at least one column")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"columns not in the frame: {missing}")
    parts = [df[column].map(normalize_key) for column in columns]
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    blank = key.str.replace("|", "", regex=False).str.len().eq(0)
    return key.mask(blank).astype("string")


def add_time_parts(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Month, hour, and day-of-week parts for two-way tables. Original column stays."""
    stamps = pd.to_datetime(df[time_column], errors="coerce")
    if isinstance(stamps.dtype, pd.DatetimeTZDtype):
        stamps = stamps.dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
    return pd.DataFrame(
        {
            f"{time_column}_month": stamps.dt.month,
            f"{time_column}_hour": stamps.dt.hour,
            f"{time_column}_dow": stamps.dt.dayofweek,
        },
        index=df.index,
    )


def suggest_two_way_columns(df: pd.DataFrame, roles: dict) -> tuple[str | None, str | None]:
    """Two low-cardinality descriptor, time, or geo columns for a rate table."""
    tagged = roles["columns"] if isinstance(roles, dict) and "columns" in roles else roles
    candidates = []
    for column, role in tagged.items():
        if column not in df.columns or role not in {"descriptor", "time", "geo"}:
            continue
        nunique = int(df[column].nunique(dropna=True))
        if 2 <= nunique <= 40:
            candidates.append((nunique, column))
    candidates.sort()
    if len(candidates) < 2:
        return None, None
    return candidates[0][1], candidates[1][1]


def two_way(
    df: pd.DataFrame,
    row_column: str,
    col_column: str,
    target: str | None = None,
) -> pd.DataFrame:
    """Counts, or the mean of a target, across two columns."""
    if target is None:
        return pd.crosstab(df[row_column], df[col_column])
    return pd.crosstab(df[row_column], df[col_column], values=df[target], aggfunc="mean")


def aggregate_measures(
    df: pd.DataFrame,
    entity_column: str,
    measure_columns: list[str],
    time_column: str | None = None,
    train_positions: np.ndarray | list[int] | None = None,
) -> pd.DataFrame:
    """Count, nunique, mean, sum, and last-time features from training rows only.

    The returned frame is aligned to ``df`` and does not include ``entity_column``.
    Entities that never appear in training get missing aggregates.
    """
    if train_positions is None:
        raise ValueError(
            "Pass train_positions. Aggregates are computed on training rows only."
        )
    positions = np.asarray(train_positions, dtype=int)
    if len(positions) == 0:
        raise ValueError("train_positions is empty")
    base = df.iloc[positions]
    grouped = base.groupby(entity_column, dropna=False)
    pieces = [grouped.size().rename("agg_count")]
    for column in measure_columns:
        if column not in df.columns or column == entity_column:
            continue
        series = base[column]
        pieces.append(grouped[column].nunique(dropna=True).rename(f"agg_nunique_{column}"))
        if pd.api.types.is_numeric_dtype(series):
            pieces.append(grouped[column].mean().rename(f"agg_mean_{column}"))
            pieces.append(grouped[column].sum().rename(f"agg_sum_{column}"))
    if time_column is not None and time_column in df.columns:
        stamps = pd.to_datetime(base[time_column], errors="coerce")
        if isinstance(stamps.dtype, pd.DatetimeTZDtype):
            stamps = stamps.dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
        hours = (stamps - pd.Timestamp("1970-01-01")).dt.total_seconds() / 3600.0
        last_time = hours.groupby(base[entity_column], dropna=False).max()
        pieces.append(last_time.rename("agg_last_time_hours"))
    stats = pd.concat(pieces, axis=1)
    stats.index.name = entity_column
    stats = stats.reset_index()
    aligned = df[[entity_column]].merge(stats, on=entity_column, how="left")
    if len(aligned) != len(df):
        raise ValueError(
            "aggregates changed the number of rows; the entity key is not unique on the stats side"
        )
    return aligned.drop(columns=[entity_column]).reset_index(drop=True)
