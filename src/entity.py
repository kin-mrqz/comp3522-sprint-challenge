"""Identifier cleanup and exact overlap. Fuzzy pairs are listed and not merged."""

from __future__ import annotations

import difflib
import re

import pandas as pd


def clean_id(value) -> str:
    """Strip an identifier the same way the HKO project does. Case is kept."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_key(value) -> str:
    """Case-fold and drop punctuation so ``RF024`` and ``rf-024`` can meet.

    The alias table keeps the original ``clean_id`` beside this key. A blank
    string means the value was missing.
    """
    return re.sub(r"[^0-9a-z]+", "", clean_id(value).casefold())


def with_keys(df: pd.DataFrame, id_column: str, prefix: str = "entity") -> pd.DataFrame:
    """Add clean and normalized id columns. Does not drop the original."""
    out = df.copy()
    out[f"{prefix}_clean"] = out[id_column].map(clean_id)
    out[f"{prefix}_key"] = out[id_column].map(normalize_key)
    return out


def alias_table(series: pd.Series, source: str) -> pd.DataFrame:
    """One row per raw spelling, with the normalized key and how often it appears."""
    frame = pd.DataFrame(
        {
            "source": source,
            "raw_id": series.map(clean_id),
            "normalized_id": series.map(normalize_key),
        }
    )
    grouped = (
        frame.groupby(["source", "raw_id", "normalized_id"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["n_rows", "normalized_id"], ascending=[False, True])
    )
    return grouped.reset_index(drop=True)


def key_overlap(left: pd.Series, right: pd.Series) -> dict:
    """Exact overlap of normalized keys. Empty keys are ignored."""
    left_keys = {key for key in left.map(normalize_key) if key}
    right_keys = {key for key in right.map(normalize_key) if key}
    both = left_keys & right_keys
    return {
        "n_left": len(left_keys),
        "n_right": len(right_keys),
        "n_both": len(both),
        "n_left_only": len(left_keys - right_keys),
        "n_right_only": len(right_keys - left_keys),
        "match_rate_left": (len(both) / len(left_keys)) if left_keys else float("nan"),
        "match_rate_right": (len(both) / len(right_keys)) if right_keys else float("nan"),
    }


def unmatched_keys(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    """Normalized keys that appear on only one side."""
    left_keys = {key for key in left.map(normalize_key) if key}
    right_keys = {key for key in right.map(normalize_key) if key}
    rows = [{"normalized_id": key, "side": "left"} for key in sorted(left_keys - right_keys)]
    rows += [{"normalized_id": key, "side": "right"} for key in sorted(right_keys - left_keys)]
    return pd.DataFrame(rows, columns=["normalized_id", "side"])


def join_on_key(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_key: str,
    right_key: str,
    how: str = "left",
) -> pd.DataFrame:
    """Left-join on the normalized id. Blank keys do not match each other."""
    a = left.copy()
    b = right.copy()
    a["_join_key"] = a[left_key].map(normalize_key)
    b["_join_key"] = b[right_key].map(normalize_key)
    b = b[b["_join_key"] != ""]
    merged = a.merge(b, on="_join_key", how=how, suffixes=("", "_right"))
    return merged


def fuzzy_candidates(
    left_ids: pd.Series,
    right_ids: pd.Series,
    cutoff: float = 0.88,
    limit: int = 30,
) -> pd.DataFrame:
    """List near-spellings among keys that failed an exact match.

    This does not merge anything. A person looks at the pairs and, if a pair
    is the same entity, adds it to the alias table by hand.
    """
    left = sorted({key for key in left_ids.map(normalize_key) if key})
    right = sorted({key for key in right_ids.map(normalize_key) if key})
    right_set = set(right)
    left_set = set(left)
    left_only = [key for key in left if key not in right_set]
    right_only = [key for key in right if key not in left_set]
    columns = ["left_id", "right_id", "score", "note"]
    if len(left_only) * len(right_only) > 40000:
        return pd.DataFrame(
            [
                {
                    "left_id": "",
                    "right_id": "",
                    "score": float("nan"),
                    "note": "too many unmatched keys for a fuzzy pass; review the unmatched list instead",
                }
            ],
            columns=columns,
        )
    rows = []
    for left_key in left_only:
        best_score = -1.0
        best_right = ""
        for right_key in right_only:
            score = difflib.SequenceMatcher(None, left_key, right_key).ratio()
            if score > best_score:
                best_score = score
                best_right = right_key
        if best_score >= cutoff:
            rows.append(
                {
                    "left_id": left_key,
                    "right_id": best_right,
                    "score": round(best_score, 3),
                    "note": "review before merge",
                }
            )
    rows.sort(key=lambda row: -row["score"])
    return pd.DataFrame(rows[:limit], columns=columns)
