"""Column roles: id, time, geo, target, descriptor, measure, or drop."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .paths import ROLES_PATH, ensure_dirs

ROLES = ("id", "time", "geo", "target", "descriptor", "measure", "drop")

_ID = re.compile(r"(^|_)(id|code|key|station|gauge|site)(_|$)", re.I)
_TIME = re.compile(r"time|date|hour|timestamp", re.I)
_GEO = re.compile(r"lat|lon|lng|easting|northing|coord", re.I)
_TARGET = re.compile(r"(^|_)(target|label)(_|$)", re.I)


def suggest_roles(df: pd.DataFrame) -> dict:
    """Draft a role for every column. The team edits this before cleaning."""
    roles: dict[str, str] = {}
    n_rows = len(df)
    for column in df.columns:
        series = df[column]
        name = str(column)
        missing = float(series.isna().mean()) if n_rows else 0.0
        nunique = int(series.nunique(dropna=True))
        if nunique <= 1 or missing > 0.95:
            roles[name] = "drop"
            continue
        if _TARGET.search(name):
            roles[name] = "target"
            continue
        if _GEO.search(name):
            roles[name] = "geo"
            continue
        if _TIME.search(name) or pd.api.types.is_datetime64_any_dtype(series):
            roles[name] = "time"
            continue
        if _ID.search(name):
            roles[name] = "id"
            continue
        if pd.api.types.is_bool_dtype(series) or (
            pd.api.types.is_numeric_dtype(series) and nunique > 40
        ):
            roles[name] = "measure"
            continue
        if pd.api.types.is_numeric_dtype(series):
            roles[name] = "descriptor" if nunique <= 12 else "measure"
            continue
        if n_rows and nunique / n_rows > 0.9:
            roles[name] = "id"
        else:
            roles[name] = "descriptor"
    return {
        "columns": roles,
        "notes": "Draft from names and dtypes. Edit before notebook 02 uses it.",
    }


def validate_roles(payload: dict, columns) -> dict:
    """Check that every tag is a known role and report columns the file missed."""
    tagged = payload["columns"] if isinstance(payload, dict) and "columns" in payload else payload
    bad_roles = sorted({role for role in tagged.values() if role not in ROLES})
    if bad_roles:
        raise ValueError(f"unknown roles: {bad_roles}")
    column_set = {str(column) for column in columns}
    return {
        "unknown_columns": sorted(set(tagged) - column_set),
        "untagged_columns": sorted(column_set - set(tagged)),
    }


def columns_with_role(payload: dict, role: str) -> list[str]:
    tagged = payload["columns"] if isinstance(payload, dict) and "columns" in payload else payload
    return [column for column, tagged_role in tagged.items() if tagged_role == role]


def write_roles(payload: dict, path: Path | None = None) -> Path:
    ensure_dirs()
    path = Path(path) if path else ROLES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_roles(path: Path | None = None) -> dict:
    path = Path(path) if path else ROLES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No column roles at {path}. Notebook 00 writes a draft; edit it before notebook 02."
        )
    return json.loads(path.read_text(encoding="utf-8"))
