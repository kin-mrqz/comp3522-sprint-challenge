"""Load the unseen tables and write a one-page profile of each."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import RAW, TABLES, ensure_dirs

TABULAR = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}


def list_raw_files(directory: Path | None = None) -> list[Path]:
    """Tabular files under data/raw, including one unzipped folder."""
    directory = Path(directory) if directory else RAW
    if not directory.exists():
        return []
    files = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.startswith("~$"):
            continue
        if path.suffix.lower() in TABULAR:
            files.append(path)
    return sorted(files)


def load_table(path: Path | str) -> pd.DataFrame:
    """Read one csv, tsv, spreadsheet, parquet, or json table."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported table type: {path.name}")


def profile(df: pd.DataFrame, sample_n: int = 3) -> pd.DataFrame:
    """One row per column: dtype, missing fraction, cardinality, and a few values."""
    rows = []
    for column in df.columns:
        series = df[column]
        samples = (
            series.dropna()
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.slice(0, 40)
            .head(sample_n)
            .tolist()
        )
        rows.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "rows": int(len(series)),
                "missing_pct": float(series.isna().mean()) if len(series) else 0.0,
                "nunique": int(series.nunique(dropna=True)),
                "sample": " | ".join(samples),
            }
        )
    return pd.DataFrame(rows)


def save_frame(df: pd.DataFrame, path: Path | str) -> Path:
    """Write parquet, and fall back to csv if a column cannot be stored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as exc:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        path.with_suffix(".parquet-error.txt").write_text(str(exc), encoding="utf-8")
        return csv_path


def read_frame(path: Path | str) -> pd.DataFrame:
    """Read a parquet path, or the csv written when parquet was refused."""
    path = Path(path)
    if path.exists():
        return load_table(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(path)


def write_profile(df: pd.DataFrame, stem: str) -> Path:
    """Save a profile table under outputs/tables."""
    ensure_dirs()
    path = TABLES / f"profile_{stem}.csv"
    profile(df).to_csv(path, index=False)
    return path


def save_figure(fig, name: str):
    """Save a matplotlib figure under outputs/figures and close it."""
    from .paths import FIGURES

    ensure_dirs()
    path = FIGURES / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
