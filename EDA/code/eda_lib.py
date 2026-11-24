"""Shared helpers for the COMP3522 AQHI sprint EDA.

Run scripts from the repository root:
    .venv/bin/python EDA/code/01_profile.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "aqhi_data.csv"
PROC = ROOT / "data" / "processed"
EDA = ROOT / "EDA"
RES = EDA / "results"
FIG = RES / "figures"
TAB = RES / "tables"
for _p in (FIG, TAB):
    _p.mkdir(parents=True, exist_ok=True)

# station, slug, region, station type (EPD: 15 general + 3 roadside)
STATIONS = [
    ("Central/Western", "central_western", "Hong Kong Island", "general"),
    ("Southern", "southern", "Hong Kong Island", "general"),
    ("Eastern", "eastern", "Hong Kong Island", "general"),
    ("Kwun Tong", "kwun_tong", "Kowloon", "general"),
    ("Sham Shui Po", "sham_shui_po", "Kowloon", "general"),
    ("Kwai Chung", "kwai_chung", "New Territories", "general"),
    ("Tsuen Wan", "tsuen_wan", "New Territories", "general"),
    ("Tseung Kwan O", "tseung_kwan_o", "New Territories", "general"),
    ("Yuen Long", "yuen_long", "New Territories", "general"),
    ("Tuen Mun", "tuen_mun", "New Territories", "general"),
    ("Tung Chung", "tung_chung", "New Territories", "general"),
    ("Tai Po", "tai_po", "New Territories", "general"),
    ("Sha Tin", "sha_tin", "New Territories", "general"),
    ("North", "north", "New Territories", "general"),
    ("Tap Mun", "tap_mun", "New Territories", "general"),
    ("Causeway Bay", "causeway_bay", "Hong Kong Island", "roadside"),
    ("Central", "central", "Hong Kong Island", "roadside"),
    ("Mong Kok", "mong_kok", "Kowloon", "roadside"),
]
META = pd.DataFrame(STATIONS, columns=["station", "slug", "region", "station_type"])
META["is_background"] = META["station"].eq("Tap Mun")

REGION_ORDER = ["Hong Kong Island", "Kowloon", "New Territories"]
REGION_COLOR = {
    "Hong Kong Island": "#1f77b4",
    "Kowloon": "#d62728",
    "New Territories": "#2ca02c",
}
TYPE_COLOR = {"general": "#4c72b0", "roadside": "#c44e52"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
BAND_ORDER = ["Low (1-3)", "Moderate (4-6)", "High (7)", "Very High (8-10)", "Serious (10+)"]
BAND_COLOR = {
    "Low (1-3)": "#2ca02c",
    "Moderate (4-6)": "#ffbf00",
    "High (7)": "#ff7f0e",
    "Very High (8-10)": "#d62728",
    "Serious (10+)": "#6a0dad",
}
# AQHI compilation was updated on 2025-03-22 (ozone risk factor). Bands are EPD's.
METHOD_CHANGE = pd.Timestamp("2025-03-22")
FULL_YEARS = [2021, 2022, 2023, 2024, 2025]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
})


def save_fig(fig, name: str) -> Path:
    path = FIG / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = TAB / name
    df.to_csv(path, index=index, float_format="%.6g")
    return path


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / name)


def band(value: float) -> str:
    if pd.isna(value):
        return ""
    if value >= 10:
        return "Serious (10+)"
    if value >= 8:
        return "Very High (8-10)"
    if value >= 7:
        return "High (7)"
    if value >= 4:
        return "Moderate (4-6)"
    return "Low (1-3)"


def add_bands(df: pd.DataFrame, column: str = "aqhi", out: str = "band") -> pd.DataFrame:
    df = df.copy()
    df[out] = pd.cut(
        df[column].clip(lower=1, upper=10),
        bins=[0, 3, 6, 7, 9, 10],
        labels=BAND_ORDER,
        include_lowest=True,
    )
    return df


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{100 * value:.{digits}f}%"


def stamp(ax, text: str, loc: str = "upper left", size: int = 9) -> None:
    """Put a number-bearing caption inside an axes."""
    positions = {
        "upper left": (0.015, 0.97, "left", "top"),
        "upper right": (0.985, 0.97, "right", "top"),
        "lower left": (0.015, 0.03, "left", "bottom"),
        "lower right": (0.985, 0.03, "right", "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.7"))


def ci95(series: pd.Series) -> float:
    """Approximate 95% half-width from the standard error of the values given."""
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return np.nan
    return float(1.96 * s.std(ddof=1) / np.sqrt(len(s)))


def write_findings(rows: list[dict], name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(TAB / name, index=False, float_format="%.6g")
    return df
