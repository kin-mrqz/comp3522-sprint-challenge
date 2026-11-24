"""Frozen train, validation, and test positions. The file is written once."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import SPLIT_PATH, ensure_dirs

PARTS = ("train", "val", "test")


def _as_naive(time: pd.Series) -> pd.Series:
    """Hong Kong local time, stored as a naive clock, matching the project tables."""
    parsed = pd.to_datetime(time, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
    return parsed


def stratified_split(
    y,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Stratified split of labelled rows. Positions are integer locations, not index labels."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    labels = pd.Series(np.asarray(y)).reset_index(drop=True)
    labelled = np.flatnonzero(labels.notna().to_numpy())
    if len(labelled) < 3:
        raise ValueError("need at least 3 labelled rows to make train, val, and test")
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[np.ndarray]] = {part: [] for part in PARTS}
    for cls in pd.unique(labels.iloc[labelled]):
        idx = labelled[labels.iloc[labelled].eq(cls).to_numpy()]
        idx = rng.permutation(idx)
        n = len(idx)
        if n >= 3:
            n_train = max(1, int(round(fractions[0] * n)))
            n_val = max(1, int(round(fractions[1] * n)))
            if n_train + n_val >= n:
                n_val = 1
                n_train = n - 2
            buckets["train"].append(idx[:n_train])
            buckets["val"].append(idx[n_train : n_train + n_val])
            buckets["test"].append(idx[n_train + n_val :])
        elif n == 2:
            buckets["train"].append(idx[:1])
            buckets["val"].append(idx[1:])
        elif n == 1:
            buckets["train"].append(idx)
    split = {}
    for part in PARTS:
        if not buckets[part]:
            raise ValueError(
                f"stratified split produced an empty {part} part; the labelled set is too small"
            )
        split[part] = np.sort(np.concatenate(buckets[part])).astype(np.int64)
    return split


def time_split(
    time: pd.Series,
    train_end: str,
    val_end: str,
    labelled: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Out-of-time split: train <= train_end < val <= val_end < test."""
    stamps = _as_naive(time).reset_index(drop=True)
    train_cut = pd.Timestamp(train_end)
    val_cut = pd.Timestamp(val_end)
    if train_cut >= val_cut:
        raise ValueError("train_end must be earlier than val_end")
    ok = stamps.notna().to_numpy()
    if labelled is not None:
        labelled = np.asarray(labelled, dtype=bool)
        if len(labelled) != len(stamps):
            raise ValueError("labelled mask length must match the time column")
        ok = ok & labelled
    train = np.flatnonzero(ok & (stamps <= train_cut).to_numpy())
    val = np.flatnonzero(ok & (stamps > train_cut).to_numpy() & (stamps <= val_cut).to_numpy())
    test = np.flatnonzero(ok & (stamps > val_cut).to_numpy())
    return {
        "train": train.astype(np.int64),
        "val": val.astype(np.int64),
        "test": test.astype(np.int64),
    }


def assert_valid_split(split: dict[str, np.ndarray], n: int) -> None:
    for part in PARTS:
        if part not in split:
            raise AssertionError(f"split missing part {part!r}")
        idx = np.asarray(split[part])
        if len(idx) == 0:
            raise AssertionError(f"split part {part!r} is empty")
        if idx.min() < 0 or idx.max() >= n:
            raise AssertionError(f"split part {part!r} has a position outside 0..{n - 1}")
    train, val, test = (set(np.asarray(split[part]).tolist()) for part in PARTS)
    if train & val or train & test or val & test:
        raise AssertionError("split parts overlap")


def masks(split: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    out = {}
    for part in PARTS:
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(split[part], dtype=int)] = True
        out[part] = mask
    return out


def save_split(
    split: dict[str, np.ndarray],
    n: int,
    path: Path | None = None,
    note: str = "",
    force: bool = False,
) -> Path:
    """Write ``splits/split.npz`` once. A second different split is refused."""
    ensure_dirs()
    path = Path(path) if path else SPLIT_PATH
    cleaned = {part: np.asarray(split[part], dtype=np.int64) for part in PARTS}
    assert_valid_split(cleaned, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        existing = load_split(path)
        if all(np.array_equal(existing[part], cleaned[part]) for part in PARTS):
            return path
        raise FileExistsError(
            f"Refusing to overwrite {path}. The frozen split differs from this one. "
            "Pass force=True only if the whole team agrees the first split was wrong."
        )
    np.savez_compressed(path, **cleaned)
    meta = {
        "sizes": {part: int(len(cleaned[part])) for part in PARTS},
        "n": int(n),
        "note": note,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_split(path: Path | None = None) -> dict[str, np.ndarray]:
    path = Path(path) if path else SPLIT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No frozen split at {path}. Notebook 04 writes it, and only once."
        )
    with np.load(path) as stored:
        return {part: stored[part].astype(np.int64) for part in PARTS}


def split_exists(path: Path | None = None) -> bool:
    path = Path(path) if path else SPLIT_PATH
    return path.exists()
