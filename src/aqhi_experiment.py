"""AQHI sprint experiment: tidy the EPD file, describe 2021-2025, join project weather.

Poor air means AQHI >= 7 (EPD High or worse). Safer means AQHI 1-3 (Low).
Substituted hours (a trailing asterisk) are kept in the file and left out of the rates.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .graph_gate import write_gate
from .paths import (
    ANALYSIS_PATH,
    ANSWERS_PATH,
    CONTRACT_PATH,
    FIGURES,
    RAW,
    TABLES,
    ensure_dirs,
)

PROJECT = Path(r"C:\Users\User\OneDrive\Uni\Y4 Sem1\COMP3522\COMP3522-HKO")
CACHE = PROJECT / "graph" / "hf_cache"
SOURCE_CSV = Path(__file__).resolve().parents[1] / "task specifics" / "aqhi_data.csv"
SLIDES = Path(__file__).resolve().parents[1] / "slides" / "aqhi_sprint_deck.pdf"

REGIONS = {
    "Central/Western": "Hong Kong Island",
    "Central": "Hong Kong Island",
    "Causeway Bay": "Hong Kong Island",
    "Eastern": "Hong Kong Island",
    "Southern": "Hong Kong Island",
    "Mong Kok": "Kowloon",
    "Sham Shui Po": "Kowloon",
    "Kwun Tong": "Kowloon",
    "Kwai Chung": "New Territories",
    "Tsuen Wan": "New Territories",
    "Tuen Mun": "New Territories",
    "Yuen Long": "New Territories",
    "Sha Tin": "New Territories",
    "Tai Po": "New Territories",
    "North": "New Territories",
    "Tseung Kwan O": "New Territories",
    "Tung Chung": "New Territories",
    "Tap Mun": "New Territories",
}
ROADSIDE = {"Causeway Bay", "Central", "Mong Kok"}
WEATHER_SITES = {
    "Sham Shui Po": {"temp_code": "SSP", "rain_id": "RF022", "lat": 22.3333, "lon": 114.1500},
    "Kwun Tong": {"temp_code": "KTG", "rain_id": "K03", "lat": 22.3083, "lon": 114.2233},
    "Sha Tin": {"temp_code": "SHA", "rain_id": "RF020", "lat": 22.3833, "lon": 114.1833},
}
FPR_CAPS = (0.10, 0.20, 0.30)
SLIDE_FPR = 0.20
SEASON_OF = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
POOR = 7
LOW = 3
RAIN_DAY_HOURS = 18


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def copy_raw() -> Path:
    ensure_dirs()
    dest = RAW / "aqhi_data.csv"
    shutil.copyfile(SOURCE_CSV, dest)
    return dest


def load_aqhi(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Melt the wide EPD file. Blanks and dashes are missing. Asterisks are flagged."""
    path = Path(path) if path else RAW / "aqhi_data.csv"
    wide = pd.read_csv(path, skiprows=7)
    stations = [column for column in wide.columns if column not in {"Date", "Hour"}]
    unknown = [station for station in stations if station not in REGIONS]
    if unknown:
        raise ValueError(f"stations missing from the scenario list: {unknown}")
    long = wide.melt(id_vars=["Date", "Hour"], value_vars=stations, var_name="station", value_name="aqhi_raw")
    text = long["aqhi_raw"].astype("string").str.strip()
    missing_token = text.isna() | text.isin({"", "-", "nan", "<NA>"})
    substituted = text.str.endswith("*", na=False)
    number = pd.to_numeric(text.str.replace("*", "", regex=False), errors="coerce")
    long["substituted"] = substituted.fillna(False)
    long["aqhi"] = number
    long["date"] = pd.to_datetime(long["Date"], dayfirst=True, format="%d/%m/%Y")
    hour_number = pd.to_numeric(long["Hour"], errors="coerce")
    long["is_daily_max"] = long["Hour"].astype(str).str.strip().eq("Daily Max")
    long["hour"] = hour_number
    long["region"] = long["station"].map(REGIONS)
    long["roadside"] = long["station"].isin(ROADSIDE)
    long["year"] = long["date"].dt.year
    long["month"] = long["date"].dt.month
    hourly = long.loc[~long["is_daily_max"]].copy()
    hourly_missing = missing_token.loc[hourly.index]
    audit = {
        "source_rows": int(len(wide)),
        "stations": stations,
        "wide_cells": int(long.shape[0]),
        "daily_max_cells": int(long["is_daily_max"].sum()),
        "hourly_cells": int(len(hourly)),
        "blank_or_dash": int(hourly_missing.sum()),
        "substituted_hourly": int(hourly["substituted"].sum()),
        "numeric_hourly": int(hourly["aqhi"].notna().sum()),
        "unparsed_hourly": int((hourly["aqhi"].isna() & ~hourly_missing & ~hourly["substituted"]).sum()),
        "years": sorted(int(year) for year in hourly["year"].dropna().unique()),
    }
    return hourly, audit


def analysis_hours(hourly: pd.DataFrame) -> pd.DataFrame:
    """2021-2025 hourly values that are numeric and not a substituted station."""
    frame = hourly.loc[
        hourly["aqhi"].notna()
        & ~hourly["substituted"]
        & hourly["year"].between(2021, 2025)
        & hourly["hour"].between(1, 24)
    ].copy()
    frame["poor"] = frame["aqhi"] >= POOR
    frame["low"] = frame["aqhi"] <= LOW
    frame["clock"] = pd.to_datetime(frame["date"]) + pd.to_timedelta(frame["hour"].astype(int), unit="h")
    frame["clock"] = frame["clock"].where(frame["hour"] < 24, frame["date"] + pd.to_timedelta(1, unit="D"))
    return frame.reset_index(drop=True)


def _rate_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    table = (
        frame.groupby(column, dropna=False)
        .agg(n=("aqhi", "size"), mean_aqhi=("aqhi", "mean"), poor_share=("poor", "mean"), low_share=("low", "mean"))
        .reset_index()
    )
    return table


def task1_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_hour = _rate_table(frame, "hour").sort_values("hour")
    by_month = _rate_table(frame, "month").sort_values("month")
    by_year = _rate_table(frame, "year").sort_values("year")
    by_region = _rate_table(frame, "region")
    by_road = _rate_table(frame, "roadside")
    by_station = _rate_table(frame, "station").sort_values("poor_share", ascending=False)
    heat = (
        frame.pivot_table(index="month", columns="hour", values="poor", aggfunc="mean")
        .reindex(index=range(1, 13), columns=range(1, 25))
        .astype(float)
    )
    return {
        "hour": by_hour,
        "month": by_month,
        "year": by_year,
        "region": by_region,
        "roadside": by_road,
        "station": by_station,
        "hour_month": heat,
    }


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
        }
    )


def plot_task1(frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    ensure_dirs()
    _style()
    paths = {}
    hour = tables["hour"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    poor_line = ax.plot(
        hour["hour"], hour["poor_share"] * 100, marker="o", color="#b45309", label="AQHI ≥ 7"
    )
    ax.set_xlabel("Hour (24 is midnight at the end of the date)")
    ax.set_ylabel("AQHI ≥ 7 (%)", color="#b45309")
    ax.set_ylim(0, 8)
    ax.set_title("2021–2025, own-station readings")
    ax.set_xticks(range(1, 25, 2))
    ax2 = ax.twinx()
    low_line = ax2.plot(
        hour["hour"], hour["low_share"] * 100, marker="o", color="#0f766e", label="AQHI 1–3"
    )
    ax2.set_ylabel("AQHI 1–3 (%)", color="#0f766e")
    ax2.set_ylim(0, 100)
    ax2.grid(False)
    lines = poor_line + low_line
    ax.legend(lines, [line.get_label() for line in lines], loc="center left", frameon=True)
    fig.tight_layout()
    paths["hour"] = FIGURES / "aqhi_by_hour.png"
    fig.savefig(paths["hour"], dpi=140)
    plt.close(fig)

    month = tables["month"]
    road = (
        frame.groupby(["month", "roadside"])
        .agg(poor_share=("poor", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.38
    general = road.loc[~road["roadside"]]
    side = road.loc[road["roadside"]]
    ax.bar(general["month"] - width / 2, general["poor_share"] * 100, width, label="General", color="#0369a1")
    ax.bar(side["month"] + width / 2, side["poor_share"] * 100, width, label="Roadside", color="#b45309")
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_ylabel("Share with AQHI ≥ 7 (%)")
    ax.set_title("Poor hours by month and station type, 2021–2025")
    ax.legend()
    fig.tight_layout()
    paths["month_road"] = FIGURES / "aqhi_month_roadside.png"
    fig.savefig(paths["month_road"], dpi=140)
    plt.close(fig)

    heat = tables["hour_month"]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    image = ax.imshow(heat.to_numpy(dtype=float) * 100, aspect="auto", cmap="YlOrBr")
    ax.set_yticks(range(12), MONTHS)
    ax.set_xticks(range(0, 24, 2), list(range(1, 25, 2)))
    ax.set_xlabel("Hour")
    ax.set_title("Share with AQHI ≥ 7 (%)")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    paths["heatmap"] = FIGURES / "aqhi_hour_month.png"
    fig.savefig(paths["heatmap"], dpi=140)
    plt.close(fig)
    return paths


def _nearest_gauges(stations: pd.DataFrame) -> pd.DataFrame:
    latest = stations.sort_values("year").groupby("station_id", as_index=False).tail(1)
    rows = []
    for name, site in WEATHER_SITES.items():
        distance = latest.apply(
            lambda row: _haversine_km(site["lat"], site["lon"], row["latitude"], row["longitude"]),
            axis=1,
        )
        pick = latest.loc[distance.idxmin()]
        rows.append(
            {
                "aqhi_station": name,
                "temp_code": site["temp_code"],
                "gauge_id": str(pick["station_id"]),
                "gauge_name": pick["station_name"],
                "distance_km": float(distance.min()),
            }
        )
    return pd.DataFrame(rows)


def join_weather(frame: pd.DataFrame) -> dict:
    """Daily max temperature at SSP, KTG, and SHA, plus hourly rain from the nearest project gauge."""
    if not CACHE.exists():
        raise FileNotFoundError(f"project weather cache is not at {CACHE}")
    sites = list(WEATHER_SITES)
    local = frame.loc[frame["station"].isin(sites) & frame["year"].between(2023, 2025)].copy()
    temp = pd.read_csv(CACHE / "hko" / "hko_daily_temp_wide.csv", parse_dates=["date"])
    temp = temp.loc[temp["station_code"].isin([site["temp_code"] for site in WEATHER_SITES.values()])]
    code_to_name = {site["temp_code"]: name for name, site in WEATHER_SITES.items()}
    temp["station"] = temp["station_code"].map(code_to_name)
    daily_aqhi = (
        local.groupby(["date", "station"], as_index=False)
        .agg(daily_max_aqhi=("aqhi", "max"), n_hours=("aqhi", "size"), poor_share=("poor", "mean"))
    )
    daily = daily_aqhi.merge(
        temp[["date", "station", "max_temp_c", "mean_temp_c"]],
        on=["date", "station"],
        how="inner",
    )
    daily["month"] = daily["date"].dt.month

    gauges = _nearest_gauges(pd.read_parquet(CACHE / "cowin" / "cowin_stations.parquet"))
    rain = pd.read_parquet(CACHE / "cowin" / "cowin_hourly_rainfall.parquet")
    rain = rain.loc[rain["qcscore"].eq("G") & rain["station_id"].astype(str).isin(gauges["gauge_id"])].copy()
    rain["clock"] = pd.to_datetime(rain["obs_time_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
    rain["gauge_id"] = rain["station_id"].astype(str)
    rain = rain[["clock", "gauge_id", "rain_mm_1h"]]
    keyed = local.merge(gauges[["aqhi_station", "gauge_id"]], left_on="station", right_on="aqhi_station", how="left")
    hourly = keyed.merge(rain, on=["clock", "gauge_id"], how="left")
    matched = hourly.dropna(subset=["rain_mm_1h"]).copy()
    matched["month"] = matched["date"].dt.month

    def pack(df, x, y):
        use = df.dropna(subset=[x, y])
        overall = float(use[x].corr(use[y], method="spearman")) if len(use) > 5 else float("nan")
        within, n_within = _residual_corr(use, x, y, "month")
        return {"n": int(len(use)), "spearman": overall, "spearman_within_month": within, "n_within": n_within}

    rain_bins = matched.copy()
    rain_bins["rain_bin"] = pd.cut(
        rain_bins["rain_mm_1h"],
        [-0.01, 0, 1, 5, 300],
        labels=["0 mm", "0–1 mm", "1–5 mm", ">5 mm"],
    )
    by_bin = (
        rain_bins.groupby("rain_bin", observed=True)
        .agg(n=("aqhi", "size"), mean_aqhi=("aqhi", "mean"), poor_share=("poor", "mean"), low_share=("low", "mean"))
        .reset_index()
    )
    rain_day = rain.copy()
    rain_day["date"] = rain_day["clock"].dt.normalize()
    daily_rain = (
        rain_day.groupby(["date", "gauge_id"], as_index=False)
        .agg(rain_mm=("rain_mm_1h", "sum"), rain_hours=("rain_mm_1h", "size"))
    )
    # A short gauge day is left out. It is not written in as a dry day.
    daily_rain = daily_rain.loc[daily_rain["rain_hours"] >= RAIN_DAY_HOURS]
    daily_rain = daily_rain.merge(gauges[["aqhi_station", "gauge_id"]], on="gauge_id", how="inner")
    daily_rain = daily_rain.rename(columns={"aqhi_station": "station"})
    daily_rain = daily.merge(
        daily_rain[["date", "station", "rain_mm", "rain_hours"]],
        on=["date", "station"],
        how="inner",
    )
    return {
        "gauges": gauges,
        "daily": daily,
        "hourly_matched": matched,
        "temp_corr": pack(daily, "max_temp_c", "daily_max_aqhi"),
        "rain_corr": pack(matched, "rain_mm_1h", "aqhi"),
        "rain_daily_corr": pack(daily_rain, "rain_mm", "daily_max_aqhi"),
        "rain_bins": by_bin,
        "hourly_coverage": {
            "aqhi_hours": int(len(local)),
            "matched_rain_hours": int(len(matched)),
            "daily_rain_days": int(len(daily_rain)),
            "min_rain_hours": RAIN_DAY_HOURS,
        },
    }


def _residual_corr(df: pd.DataFrame, x: str, y: str, by: str) -> tuple[float, int]:
    use = df.dropna(subset=[x, y, by]).copy()
    if use[by].nunique() < 2 or len(use) < 10:
        return float("nan"), int(len(use))
    use["_x"] = use[x] - use.groupby(by)[x].transform("mean")
    use["_y"] = use[y] - use.groupby(by)[y].transform("mean")
    return float(use["_x"].corr(use["_y"], method="spearman")), int(len(use))


def plot_weather(weather: dict) -> dict[str, Path]:
    _style()
    paths = {}
    daily = weather["daily"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for station, color in zip(WEATHER_SITES, ["#0369a1", "#b45309", "#0f766e"]):
        part = daily.loc[daily["station"].eq(station)]
        ax.scatter(part["max_temp_c"], part["daily_max_aqhi"], s=10, alpha=0.35, label=station, color=color)
    ax.axhline(POOR, color="#7f1d1d", lw=1, ls="--")
    ax.set_xlabel("Daily maximum temperature (°C)")
    ax.set_ylabel("Daily maximum AQHI")
    ax.set_title("2023–2025, three shared stations")
    ax.legend(markerscale=2, frameon=False)
    fig.tight_layout()
    paths["temp"] = FIGURES / "aqhi_vs_temperature.png"
    fig.savefig(paths["temp"], dpi=140)
    plt.close(fig)

    bins = weather["rain_bins"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(bins["rain_bin"].astype(str), bins["mean_aqhi"], color="#0369a1")
    ax.set_ylabel("Mean AQHI")
    ax.set_xlabel("Hourly rainfall at the nearest project gauge")
    ax.set_title("Same three stations, QC = G hours only")
    fig.tight_layout()
    paths["rain"] = FIGURES / "aqhi_vs_rain.png"
    fig.savefig(paths["rain"], dpi=140)
    plt.close(fig)
    return paths


def advice_rule(frame: pd.DataFrame) -> dict:
    """Learn a three-flag rule on 2021–2023 and check it on 2024–2025.

    An hour is flagged when its poor-hour share is above the 2021–2023 average.
    A month is flagged when that poor share is above the average and its Low
    share is below the average. The second cut keeps out a month such as July,
    whose poor share is only a hair above the average while most of its hours
    are Low. Roadside is a flag because those stations have fewer Low hours.
    Go outside when all three flags are off.
    """
    train = frame.loc[frame["year"].between(2021, 2023)]
    test = frame.loc[frame["year"].between(2024, 2025)]
    base = float(train["poor"].mean())
    base_low = float(train["low"].mean())
    month_rate = train.groupby("month").agg(poor=("poor", "mean"), low=("low", "mean"))
    hour_rate = train.groupby("hour")["poor"].mean()
    bad_months = sorted(
        int(month)
        for month, row in month_rate.iterrows()
        if row["poor"] > base and row["low"] < base_low
    )
    bad_hours = sorted(int(hour) for hour, rate in hour_rate.items() if rate > base)

    def apply(part: pd.DataFrame) -> pd.DataFrame:
        out = part.copy()
        out["flag_road"] = out["roadside"]
        out["flag_month"] = out["month"].isin(bad_months)
        out["flag_hour"] = out["hour"].isin(bad_hours)
        out["risk"] = out[["flag_road", "flag_month", "flag_hour"]].sum(axis=1).astype(int)
        out["go_out"] = out["risk"].eq(0)
        return out

    def summary(part: pd.DataFrame) -> dict:
        scored = apply(part)
        go = scored.loc[scored["go_out"]]
        stay = scored.loc[~scored["go_out"]]
        return {
            "n": int(len(scored)),
            "n_go": int(len(go)),
            "coverage": float(len(go) / len(scored)) if len(scored) else float("nan"),
            "poor_go": float(go["poor"].mean()) if len(go) else float("nan"),
            "poor_stay": float(stay["poor"].mean()) if len(stay) else float("nan"),
            "poor_all": float(scored["poor"].mean()) if len(scored) else float("nan"),
            "low_go": float(go["low"].mean()) if len(go) else float("nan"),
            "low_all": float(scored["low"].mean()) if len(scored) else float("nan"),
        }

    return {
        "bad_months": bad_months,
        "bad_hours": bad_hours,
        "train_poor": base,
        "train_low": base_low,
        "july_poor": float(month_rate.loc[7, "poor"]),
        "july_low": float(month_rate.loc[7, "low"]),
        "train": summary(train),
        "test": summary(test),
        "all": summary(frame),
    }


def _pct(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{100 * value:.1f}%"


def _num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def _month_list(months: list[int]) -> str:
    return ", ".join(MONTHS[month - 1] for month in months) if months else "none"


def write_contract(audit: dict) -> None:
    contract = {
        "questions": [
            {
                "id": 1,
                "text": "Key trends in hourly AQHI, 2021–2025.",
                "answer_sentence": "Poor hours follow the afternoon and September. Roadside stations have fewer Low hours, not more hours at or above 7.",
                "owner": "A",
            },
            {
                "id": 2,
                "text": "Relationship between project weather and AQHI.",
                "answer_sentence": "Hotter days have a lower daily maximum AQHI because the hot months are cleaner. Within a month that link flips and is weak. Hourly rain disappears; daily rain shrinks but stays negative.",
                "owner": "B",
            },
            {
                "id": 3,
                "text": "Conditions that raise the chance of going outside safely.",
                "answer_sentence": "At Sham Shui Po, Kwun Tong, and Sha Tin, stay in when the score is at or above the fit-year threshold that holds the false-positive rate to 20%.",
                "owner": "C",
            },
        ],
        "row_unit": "station-hour",
        "target": "aqhi",
        "decision_time": "the clock hour of the reading",
        "useless_columns": ["Daily Max rows are held out of the hourly rates", "2026 rows are outside Task 1"],
        "join_keys": ["Sham Shui Po=SSP", "Kwun Tong=KTG", "Sha Tin=SHA"],
        "entity_risk": "A trailing asterisk means the station was down and a similar station was substituted.",
        "native_graph": False,
        "graph_kind": None,
        "graph_reason": "No coordinates and no edge list in aqhi_data.csv. Region and roadside are labels, not a graph.",
        "split_rule": "Describe 2021–2025. Fit the three-station score on the years with hourly weather and check it on 2024–2025. The threshold is not chosen on the check years.",
        "train_end": "2023-12-31",
        "val_end": "2025-12-31",
        "model_needed": False,
        "audit_numeric_hours_all_years": audit.get("numeric_hourly"),
    }
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    write_gate(False, contract["graph_reason"])


def _within_phrase(overall: float, within: float) -> str:
    if pd.isna(within):
        return "cannot be checked within the month"
    if abs(within) < 0.05:
        return f"disappears once the month is removed (Spearman {_num(within)})"
    if overall * within < 0:
        return f"changes sign once the month is removed (Spearman {_num(within)})"
    if abs(within) + 0.05 < abs(overall):
        return f"shrinks once the month is removed but does not disappear (Spearman {_num(within)})"
    return f"remains once the month is removed (Spearman {_num(within)})"


def _as_hkt_naive(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series)
    if getattr(parsed.dt, "tz", None) is not None:
        return parsed.dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
    return parsed


def _mean_table(frame: pd.DataFrame, columns: list[str], value: str) -> pd.DataFrame:
    return (
        frame.groupby(columns, dropna=False)
        .agg(n=(value, "size"), **{f"mean_{value}": (value, "mean")})
        .reset_index()
    )


def weather_season_place() -> dict:
    """Month and place means for HKO temperature and cached rainfall. No AQHI."""
    ensure_dirs()
    _style()
    stations = pd.read_csv(CACHE / "hko" / "stations.csv")
    temp = pd.read_csv(CACHE / "hko" / "hko_daily_temp_wide.csv", parse_dates=["date"])
    temp = temp.loc[temp["date"].between("2023-01-01", "2025-12-31")].copy()
    temp_meta = stations.dropna(subset=["temperature_station_code"]).drop_duplicates("temperature_station_code")
    temp = temp.merge(
        temp_meta[["temperature_station_code", "station_name", "region"]],
        left_on="station_code",
        right_on="temperature_station_code",
        how="left",
    )
    temp["month"] = temp["date"].dt.month

    hourly = pd.read_parquet(
        CACHE / "open_meteo" / "om_historical_weather_hourly.parquet",
        columns=["station_id", "time_hkt", "precipitation", "temperature_2m"],
    )
    hourly["clock"] = _as_hkt_naive(hourly["time_hkt"])
    hourly = hourly.loc[hourly["clock"].between("2023-01-01", "2025-12-31 23:00:00")].copy()
    hourly = hourly.drop_duplicates(["station_id", "clock"])
    hourly["date"] = hourly["clock"].dt.normalize()
    daily = hourly.groupby(["station_id", "date"], as_index=False).agg(
        rain_mm=("precipitation", "sum"),
        n_hours=("precipitation", "size"),
        n_obs=("precipitation", "count"),
    )
    # An incomplete day is left out. Missing hours are not written in as zero.
    daily = daily.loc[daily["n_obs"].eq(24) & daily["n_hours"].eq(24)].copy()
    rain_meta = stations.dropna(subset=["rainfall_station_id"]).drop_duplicates("rainfall_station_id")
    daily = daily.merge(
        rain_meta[["rainfall_station_id", "station_name", "region"]],
        left_on="station_id",
        right_on="rainfall_station_id",
        how="left",
    )
    daily["month"] = daily["date"].dt.month

    tables = {
        "temp_month": _mean_table(temp, ["month"], "max_temp_c"),
        "temp_station": _mean_table(temp, ["station_code", "station_name", "region"], "max_temp_c"),
        "temp_month_region": _mean_table(temp, ["region", "month"], "max_temp_c"),
        "rain_month": _mean_table(daily, ["month"], "rain_mm"),
        "rain_station": _mean_table(daily, ["station_id", "station_name", "region"], "rain_mm"),
        "rain_month_region": _mean_table(daily, ["region", "month"], "rain_mm"),
    }
    for name, table in tables.items():
        table.to_csv(TABLES / f"hko_{name}.csv", index=False)

    focus_temp_codes = [site["temp_code"] for site in WEATHER_SITES.values()]
    focus_rain_ids = [site["rain_id"] for site in WEATHER_SITES.values()]
    temp_focus = temp.loc[temp["station_code"].isin(focus_temp_codes)]
    rain_focus = daily.loc[daily["station_id"].isin(focus_rain_ids)]
    temp_lines = (
        temp_focus.groupby(["station_code", "month"], as_index=False)["max_temp_c"].mean()
    )
    rain_lines = rain_focus.groupby(["station_id", "month"], as_index=False)["rain_mm"].mean()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, site in WEATHER_SITES.items():
        part = temp_lines.loc[temp_lines["station_code"].eq(site["temp_code"])]
        ax.plot(part["month"], part["max_temp_c"], marker="o", label=name)
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_ylabel("Mean daily maximum (°C)")
    ax.set_title("HKO daily maximum, 2023–2025")
    ax.legend(frameon=False)
    fig.tight_layout()
    temp_path = FIGURES / "hko_temp_by_month.png"
    fig.savefig(temp_path, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, site in WEATHER_SITES.items():
        part = rain_lines.loc[rain_lines["station_id"].eq(site["rain_id"])]
        ax.plot(part["month"], part["rain_mm"], marker="o", label=name)
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_ylabel("Mean daily rainfall (mm)")
    ax.set_title("Complete days, Open-Meteo at the HKO sites, 2023–2025")
    ax.legend(frameon=False)
    fig.tight_layout()
    rain_path = FIGURES / "hko_rain_by_month.png"
    fig.savefig(rain_path, dpi=140)
    plt.close(fig)

    temp_month = temp_focus.groupby("month")["max_temp_c"].mean()
    rain_month = rain_focus.groupby("month")["rain_mm"].mean()
    temp_named = temp_focus.groupby("station_code")["max_temp_c"].mean().rename(
        index={site["temp_code"]: name for name, site in WEATHER_SITES.items()}
    )
    temp_station = {name: float(temp_named[name]) for name in WEATHER_SITES if name in temp_named}
    summary = {
        "temp_hottest_month": int(temp_month.idxmax()),
        "temp_hottest_c": float(temp_month.max()),
        "temp_coolest_month": int(temp_month.idxmin()),
        "temp_coolest_c": float(temp_month.min()),
        "temp_by_station": temp_station,
        "rain_wettest_month": int(rain_month.idxmax()),
        "rain_wettest_mm": float(rain_month.max()),
        "rain_driest_month": int(rain_month.idxmin()),
        "rain_driest_mm": float(rain_month.min()),
        "n_temp_stations": int(temp["station_code"].nunique()),
        "n_rain_stations": int(daily["station_id"].nunique()),
        "n_complete_rain_days": int(len(daily)),
    }
    keep = hourly.loc[hourly["station_id"].isin(focus_rain_ids), ["station_id", "clock", "temperature_2m", "precipitation"]]
    return {
        "summary": summary,
        "hourly": keep,
        "figures": {"hko_temp": temp_path, "hko_rain": rain_path},
    }


def _detection(y_poor: np.ndarray, stay: np.ndarray) -> dict:
    y = np.asarray(y_poor, dtype=bool)
    called = np.asarray(stay, dtype=bool)
    n_poor = int(y.sum())
    n_safe = int((~y).sum())
    caught = int((called & y).sum())
    false_pos = int((called & ~y).sum())
    return {
        "n": int(len(y)),
        "n_poor": n_poor,
        "tpr": float(caught / n_poor) if n_poor else float("nan"),
        "fpr": float(false_pos / n_safe) if n_safe else float("nan"),
        "go_share": float((~called).mean()) if len(y) else float("nan"),
    }


def _curve(y_poor: np.ndarray, proba: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        row = _detection(y_poor, proba >= threshold)
        row["threshold"] = float(threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def _pick_threshold(curve: pd.DataFrame, cap: float) -> tuple[pd.Series, bool]:
    under = curve.loc[curve["fpr"] <= cap + 1e-12]
    if under.empty:
        best = curve.sort_values(["fpr", "tpr"], ascending=[True, False]).iloc[0]
        return best, False
    best = under.sort_values(["tpr", "go_share", "threshold"], ascending=[False, False, False]).iloc[0]
    return best, True


def _score_frame(frame: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    id_to_name = {site["rain_id"]: name for name, site in WEATHER_SITES.items()}
    local = frame.loc[frame["station"].isin(WEATHER_SITES) & frame["year"].between(2021, 2025)].copy()
    local["rain_id"] = local["station"].map(lambda name: WEATHER_SITES[name]["rain_id"])
    weather = hourly.rename(columns={"station_id": "rain_id"})
    scored = local.merge(weather, on=["clock", "rain_id"], how="left")
    scored = scored.dropna(subset=["temperature_2m", "precipitation", "aqhi"]).copy()
    scored["hour"] = scored["hour"].astype(int)
    scored["month"] = scored["month"].astype(int)
    scored["season"] = scored["month"].map(SEASON_OF)
    scored["place"] = scored["rain_id"].map(id_to_name)
    return scored


def fit_risk(frame: pd.DataFrame, means: dict, rule: dict) -> dict:
    """One logistic score. The stay-in threshold is chosen on the training years only."""
    ensure_dirs()
    _style()
    scored = _score_frame(frame, means["hourly"])
    train = scored.loc[scored["year"].between(2021, 2023)].copy()
    test = scored.loc[scored["year"].between(2024, 2025)].copy()
    if train.empty or test.empty:
        raise RuntimeError("the weather join produced an empty train or test set")
    features = ["hour", "month", "station", "temperature_2m", "precipitation"]
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    model = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        ("cat", encoder, ["hour", "month", "station"]),
                        ("num", "passthrough", ["temperature_2m", "precipitation"]),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )
    model.fit(train[features], train["poor"].astype(int))
    positive = list(model.named_steps["clf"].classes_).index(1)
    train = train.copy()
    test = test.copy()
    train["proba"] = model.predict_proba(train[features])[:, positive]
    test["proba"] = model.predict_proba(test[features])[:, positive]
    thresholds = np.unique(
        np.concatenate(
            [
                [0.0, 1.0],
                np.quantile(train["proba"].to_numpy(), np.linspace(0, 1, 401)),
            ]
        )
    )
    train_curve = _curve(train["poor"].to_numpy(), train["proba"].to_numpy(), thresholds)
    test_curve = _curve(test["poor"].to_numpy(), test["proba"].to_numpy(), thresholds)
    curve = train_curve.rename(
        columns={"tpr": "train_tpr", "fpr": "train_fpr", "go_share": "train_go_share", "n": "train_n", "n_poor": "train_n_poor"}
    ).merge(
        test_curve.rename(
            columns={"tpr": "test_tpr", "fpr": "test_fpr", "go_share": "test_go_share", "n": "test_n", "n_poor": "test_n_poor"}
        ),
        on="threshold",
        how="inner",
    )
    curve.to_csv(TABLES / "aqhi_threshold_curve.csv", index=False)

    caps = []
    for cap in FPR_CAPS:
        chosen, met = _pick_threshold(train_curve, cap)
        test_at = test_curve.loc[np.isclose(test_curve["threshold"], chosen["threshold"])].iloc[0]
        caps.append(
            {
                "cap": cap,
                "met_on_train": bool(met),
                "threshold": float(chosen["threshold"]),
                "train_tpr": float(chosen["tpr"]),
                "train_fpr": float(chosen["fpr"]),
                "train_go_share": float(chosen["go_share"]),
                "test_tpr": float(test_at["tpr"]),
                "test_fpr": float(test_at["fpr"]),
                "test_go_share": float(test_at["go_share"]),
            }
        )
    cap_table = pd.DataFrame(caps)
    cap_table.to_csv(TABLES / "aqhi_threshold_caps.csv", index=False)
    locked = next(row for row in caps if abs(row["cap"] - SLIDE_FPR) < 1e-9)

    names = model.named_steps["pre"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_.ravel()
    coef_table = pd.DataFrame({"feature": names, "coef": coefs}).sort_values("coef", ascending=False)
    coef_table.to_csv(TABLES / "aqhi_risk_coefficients.csv", index=False)
    temp_coef = float(coef_table.loc[coef_table["feature"].eq("num__temperature_2m"), "coef"].iloc[0])
    rain_coef = float(coef_table.loc[coef_table["feature"].eq("num__precipitation"), "coef"].iloc[0])

    test["stay"] = test["proba"] >= locked["threshold"]
    pieces = []
    for column in ("season", "station"):
        for label, part in test.groupby(column, sort=False):
            row = _detection(part["poor"].to_numpy(), part["stay"].to_numpy())
            row["slice"] = column
            row["level"] = str(label)
            pieces.append(row)
    by_condition = pd.DataFrame(pieces)
    by_condition.to_csv(TABLES / "aqhi_threshold_by_condition.csv", index=False)

    old_stay = (
        test["month"].isin(rule["bad_months"])
        | test["hour"].isin(rule["bad_hours"])
        | test["roadside"]
    )
    old_test = _detection(test["poor"].to_numpy(), old_stay.to_numpy())
    new_test = _detection(test["poor"].to_numpy(), test["stay"].to_numpy())
    new_train = {
        "n": int(len(train)),
        "tpr": locked["train_tpr"],
        "fpr": locked["train_fpr"],
        "go_share": locked["train_go_share"],
        "year_counts": {str(int(year)): int(count) for year, count in train.groupby("year").size().items()},
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(curve["test_fpr"], curve["test_tpr"], color="#0369a1", label="2024–2025")
    ax.plot(curve["train_fpr"], curve["train_tpr"], color="#94a3b8", label="Fit years")
    for row, marker in zip(caps, ["o", "s", "D"]):
        ax.scatter(row["test_fpr"], row["test_tpr"], marker=marker, s=40, zorder=3, label=f"{int(row['cap'] * 100)}% cap")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Share of AQHI ≥ 7 hours caught")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Thresholds chosen on the fit years")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    curve_path = FIGURES / "aqhi_threshold_curve.png"
    fig.savefig(curve_path, dpi=140)
    plt.close(fig)

    return {
        "summary": means["summary"],
        "figures": {"curve": curve_path},
        "caps": caps,
        "locked": locked,
        "train": new_train,
        "test": new_test,
        "old_rule_test": old_test,
        "by_condition": by_condition,
        "temp_coef": temp_coef,
        "rain_coef": rain_coef,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_scored": int(len(scored)),
    }


def _question3_lines(risk: dict) -> list[str]:
    locked = risk["locked"]
    train = risk["train"]
    test = risk["test"]
    old = risk["old_rule_test"]
    summary = risk["summary"]
    year_bits = ", ".join(
        f"{year}: {count:,} hours" for year, count in sorted(train["year_counts"].items())
    )
    hotter = "raises" if risk["temp_coef"] > 0 else "lowers"
    wetter = "raises" if risk["rain_coef"] > 0 else "lowers"
    station_means = ", ".join(
        f"{name} {value:.1f}°C" for name, value in summary["temp_by_station"].items()
    )
    cap_bits = []
    for row in risk["caps"]:
        status = "met" if row["met_on_train"] else "not met"
        cap_bits.append(
            f"{int(row['cap'] * 100)}% ({status}): threshold {row['threshold']:.3f}, "
            f"test TPR {_pct(row['test_tpr'])}, test FPR {_pct(row['test_fpr'])}, "
            f"outdoors {_pct(row['test_go_share'])}"
        )
    season = risk["by_condition"].loc[risk["by_condition"]["slice"].eq("season")].copy()
    place = risk["by_condition"].loc[risk["by_condition"]["slice"].eq("station")].copy()
    season["_order"] = season["level"].map({"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3})
    place["_order"] = place["level"].map({name: i for i, name in enumerate(WEATHER_SITES)})
    season = season.sort_values("_order")
    place = place.sort_values("_order")
    season_bits = ", ".join(
        f"{row.level} TPR {_pct(float(row.tpr))} FPR {_pct(float(row.fpr))}" for row in season.itertuples()
    )
    place_bits = ", ".join(
        f"{row.level} TPR {_pct(float(row.tpr))} FPR {_pct(float(row.fpr))}" for row in place.itertuples()
    )
    return [
        "- Claim: Go outside in the morning, and not in September or April. Poor means AQHI at or above 7. The worst hour and the worst month are in Question 1. Roadside stations have fewer Low hours and do not have more poor hours.",
        "- Weather condition: do not wait for rain. Once the month is removed, the hourly rain link disappears. Hotter days in the year line up with a lower daily maximum AQHI because the hot months are cleaner. Within a month that temperature link is weak.",
        f"- Optional formula, three stations only: stay in at Sham Shui Po, Kwun Tong, or Sha Tin when the score is at least {locked['threshold']:.3f}. That cutoff is the fit-year point with the highest share of poor hours caught whose false-positive rate is at or under 20%. It is not the citywide rule.",
        f"- Weather first, with no AQHI in the average: at those three stations the hottest month is {MONTHS[summary['temp_hottest_month'] - 1]} ({summary['temp_hottest_c']:.1f}°C) and the coolest is {MONTHS[summary['temp_coolest_month'] - 1]} ({summary['temp_coolest_c']:.1f}°C). Mean daily maximum is {station_means}. The wettest month is {MONTHS[summary['rain_wettest_month'] - 1]} ({summary['rain_wettest_mm']:.1f} mm) and the driest is {MONTHS[summary['rain_driest_month'] - 1]} ({summary['rain_driest_mm']:.1f} mm).",
        "- Chart: `outputs/figures/hko_temp_by_month.png`, `outputs/figures/hko_rain_by_month.png`.",
        f"- Number: On the fit hours ({year_bits}) TPR is {_pct(train['tpr'])} and FPR is {_pct(train['fpr'])}. On 2024–2025 the false-positive rate stays at {_pct(test['fpr'])}, the share of poor hours caught falls to {_pct(test['tpr'])}, and he is sent out for {_pct(test['go_share'])} of hours.",
        "- Caps, frozen on the fit years and read on 2024–2025: " + "; ".join(cap_bits) + ".",
        f"- Same hours, old month-and-hour flags: test TPR {_pct(old['tpr'])}, test FPR {_pct(old['fpr'])}, outdoors {_pct(old['go_share'])}.",
        f"- 2024–2025 at this one threshold, not new cutoffs: {season_bits}. {place_bits}.",
        f"- Formula: one logistic score on hour, month, station, same-hour temperature, and same-hour precipitation. After month and hour are in the score, a hotter hour {hotter} the chance of AQHI ≥ 7 (log-odds {risk['temp_coef']:+.3f}) and a wetter hour {wetter} it (log-odds {risk['rain_coef']:+.3f}).",
        "- Chart: `outputs/figures/aqhi_threshold_curve.png`.",
        "- Limitation: Hourly temperature and rain start on 2023-01-01, so 2021 is absent and a 2022 date appears only when its hour 24 falls at that midnight. Open-Meteo is reanalysis at the HKO coordinates, not the EPD inlet and not an official rain gauge. The threshold was not chosen on 2024–2025. The fit-year catch rate does not carry over to those later years. 2026 is unused. The other 15 stations are not given this score.",
    ]


def write_answers(audit: dict, frame: pd.DataFrame, tables: dict, weather: dict, risk: dict) -> str:
    overall_poor = float(frame["poor"].mean())
    overall_low = float(frame["low"].mean())
    hour = tables["hour"].sort_values("poor_share", ascending=False).iloc[0]
    month = tables["month"].sort_values("poor_share", ascending=False).iloc[0]
    calm_month = tables["month"].sort_values("poor_share", ascending=True).iloc[0]
    year = tables["year"].sort_values("poor_share", ascending=False)
    road = tables["roadside"].set_index("roadside")
    region = tables["region"].sort_values("poor_share", ascending=False)
    station = tables["station"]
    top3 = ", ".join(
        f"{row.station} {_pct(float(row.poor_share))}" for row in station.head(3).itertuples()
    )
    temp = weather["temp_corr"]
    rain = weather["rain_corr"]
    rain_day = weather["rain_daily_corr"]
    gauges = weather["gauges"]
    cover = weather["hourly_coverage"]
    temp_claim = (
        "hotter days have a lower daily maximum AQHI"
        if temp["spearman"] < 0
        else "hotter days have a higher daily maximum AQHI"
    )
    rain_claim = (
        "wetter hours have a lower AQHI"
        if rain["spearman"] < 0
        else "wetter hours have a higher AQHI"
    )
    lines = [
        "# Answers",
        "",
        "Poor air means AQHI at or above 7 (EPD High, Very High, or Serious). Safer means AQHI 1–3 (Low). The bands are 1–3 Low, 4–6 Moderate, 7 High, 8–10 Very High, and 10+ Serious. Rates use 2021–2025 hours whose value is numeric and not marked with an asterisk.",
        "",
        "## Question 1",
        "",
        "- Claim: Poor hours follow the clock and the month. They rise through the afternoon and peak in September. They do not pile up at the roadside stations.",
        f"- Number: {_pct(overall_poor)} of {len(frame):,} station-hours have AQHI ≥ 7. The worst hour is {int(hour['hour'])} ({_pct(float(hour['poor_share']))}). The worst month is {MONTHS[int(month['month']) - 1]} ({_pct(float(month['poor_share']))}); the calmest is {MONTHS[int(calm_month['month']) - 1]} ({_pct(float(calm_month['poor_share']))}). The highest year is {int(year.iloc[0]['year'])} ({_pct(float(year.iloc[0]['poor_share']))}); the lowest is {int(year.iloc[-1]['year'])} ({_pct(float(year.iloc[-1]['poor_share']))}).",
        "- Chart: `outputs/figures/aqhi_by_hour.png`, `outputs/figures/aqhi_month_roadside.png`, `outputs/figures/aqhi_hour_month.png`.",
        f"- Roadside vs general: mean AQHI {float(road.loc[True, 'mean_aqhi']):.2f} at roadside versus {float(road.loc[False, 'mean_aqhi']):.2f} at general stations. Poor share {_pct(float(road.loc[True, 'poor_share']))} versus {_pct(float(road.loc[False, 'poor_share']))}. Low share {_pct(float(road.loc[True, 'low_share']))} versus {_pct(float(road.loc[False, 'low_share']))}. Highest region on the poor share: {region.iloc[0]['region']} at {_pct(float(region.iloc[0]['poor_share']))}. Highest stations: {top3}.",
        "- Limitation: 2026 is in the file and is not in these rates. Asterisk hours are excluded, so a station looks missing rather than copied from its neighbour. New Territories mixes Tap Mun, which has the most poor hours, with stations that are often Low.",
        "",
        "## Question 2",
        "",
        f"- Claim: At Sham Shui Po, Kwun Tong, and Sha Tin, {temp_claim}. That link {_within_phrase(temp['spearman'], temp['spearman_within_month'])}. At the same stations, {rain_claim}. That hourly link {_within_phrase(rain['spearman'], rain['spearman_within_month'])}. On days with at least {cover['min_rain_hours']} good gauge hours, a wetter day has a {'lower' if rain_day['spearman'] < 0 else 'higher'} daily maximum AQHI. That daily link {_within_phrase(rain_day['spearman'], rain_day['spearman_within_month'])}.",
        f"- Number: Temperature Spearman {_num(temp['spearman'])} on {temp['n']:,} station-days (within month {_num(temp['spearman_within_month'])}). Hourly rain Spearman {_num(rain['spearman'])} on {rain['n']:,} matched hours (within month {_num(rain['spearman_within_month'])}). Daily rain, summed only on days with at least {cover['min_rain_hours']} QC = G hours, Spearman {_num(rain_day['spearman'])} on {rain_day['n']:,} station-days (within month {_num(rain_day['spearman_within_month'])}).",
        "- Chart: `outputs/figures/aqhi_vs_temperature.png`, `outputs/figures/aqhi_vs_rain.png`.",
        "- Gauges: "
        + "; ".join(
            f"{row.aqhi_station} temperature {row.temp_code}, rain gauge {row.gauge_name} ({row.gauge_id}, {row.distance_km:.1f} km)"
            for row in gauges.itertuples()
        )
        + f". Rain match covers {cover['matched_rain_hours']:,} of {cover['aqhi_hours']:,} AQHI hours at these three stations, and {cover['daily_rain_days']:,} station-days clear the {cover['min_rain_hours']}-hour gauge bar.",
        "- Limitation: The project has no AQHI series. Rain is the nearest CoWIN gauge with QC = G, not the EPD inlet. A gauge day with fewer than "
        + f"{cover['min_rain_hours']} good hours is left out. Missing rain is not filled with zero.",
        "",
        "## Question 3",
        "",
        *_question3_lines(risk),
        "",
        "## Graph gate",
        "",
        "- Decision: no.",
        "- Why: The file is one column per station. It has no coordinates and no edge list. Region and roadside are the scenario's labels.",
        "",
        "## Project link",
        "",
        "- The project did not score AQHI, so the rainstorm CSI numbers are not the comparison.",
        "- What carries over: clocks in Hong Kong time, rainfall in millimetres, and a missing gauge hour is not filled with zero. The weather join uses the project window 2023-01-01 to 2025-12-31 and the station codes SSP, KTG, and SHA.",
        "",
        "## Data handling",
        "",
        f"- Wide file: {audit['source_rows']:,} rows, {len(audit['stations'])} stations, header after the EPD notes.",
        f"- Hourly cells: {audit['hourly_cells']:,}. Daily Max cells held out: {audit['daily_max_cells']:,}.",
        f"- Blank or dash hourly cells: {audit['blank_or_dash']:,}. Substituted hourly cells (asterisk): {audit['substituted_hourly']:,}.",
        f"- Rates above use {len(frame):,} numeric, non-substituted station-hours in 2021–2025. Low-band share overall is {_pct(overall_low)}.",
        "",
    ]
    text = "\n".join(lines)
    ANSWERS_PATH.write_text(text, encoding="utf-8")
    return text


def write_slides(frame, tables, weather, figure_paths: dict, risk: dict) -> Path:
    """Landscape PDF, one claim per page, using the figures already saved."""
    SLIDES.parent.mkdir(parents=True, exist_ok=True)
    overall = float(frame["poor"].mean())
    road = tables["roadside"].set_index("roadside")
    hour = tables["hour"].sort_values("poor_share", ascending=False).iloc[0]
    month = tables["month"].sort_values("poor_share", ascending=False).iloc[0]
    calm = tables["month"].sort_values("poor_share", ascending=True).iloc[0]
    year = tables["year"].sort_values("poor_share", ascending=False)
    station = tables["station"]
    temp = weather["temp_corr"]
    rain = weather["rain_corr"]
    rain_day = weather["rain_daily_corr"]
    cover = weather["hourly_coverage"]
    locked = risk["locked"]
    summary = risk["summary"]
    old = risk["old_rule_test"]
    if temp["spearman"] < 0:
        temp_line = "Hotter days have a lower daily max AQHI. Hot months are cleaner."
    else:
        temp_line = "Hotter days have a higher daily max AQHI."
    pages = [
        (
            "Smoking Baby and the AQHI",
            [
                "When can he go outside with the lowest risk?",
                "Data: EPD hourly AQHI, 18 stations. Task 1 uses 2021–2025.",
                "Weather: project temperature and rainfall, 2023–2025 only.",
                "Bands: 1–3 Low, 4–6 Moderate, 7 High, 8–10 Very High, 10+ Serious.",
                "Poor = AQHI ≥ 7. Safer = AQHI 1–3.",
            ],
            None,
        ),
        (
            "What we kept",
            [
                f"{len(frame):,} station-hours enter the rates.",
                "Dropped from the rates: blanks, dashes, asterisks, Daily Max rows, and 2026.",
                "An asterisk means the station was down and a neighbour was copied in.",
                "Those hours stay in the file and stay out of the percentages.",
            ],
            None,
        ),
        (
            "Poor hours follow the clock",
            [
                f"{_pct(overall)} of station-hours are at or above 7.",
                f"Hour {int(hour['hour'])} peaks at {_pct(float(hour['poor_share']))}.",
                "Amber, left axis: AQHI ≥ 7. Teal, right axis: AQHI 1–3.",
                "The morning is mostly Low. The afternoon is not.",
            ],
            figure_paths.get("hour"),
        ),
        (
            "Month, not roadside",
            [
                f"{MONTHS[int(month['month']) - 1]} worst at {_pct(float(month['poor_share']))}. {MONTHS[int(calm['month']) - 1]} calmest at {_pct(float(calm['poor_share']))}.",
                f"{int(year.iloc[0]['year'])} highest at {_pct(float(year.iloc[0]['poor_share']))}. {int(year.iloc[-1]['year'])} lowest at {_pct(float(year.iloc[-1]['poor_share']))}.",
                f"Poor share: roadside {_pct(float(road.loc[True, 'poor_share']))}, general {_pct(float(road.loc[False, 'poor_share']))}.",
                f"Low share: roadside {_pct(float(road.loc[True, 'low_share']))}, general {_pct(float(road.loc[False, 'low_share']))}.",
                f"Highest: {station.iloc[0]['station']}, {station.iloc[1]['station']}, {station.iloc[2]['station']}.",
            ],
            figure_paths.get("month_road"),
        ),
        (
            "Hour and month together",
            [
                "Each cell is the share of station-hours with AQHI ≥ 7.",
                "September afternoons are the dark band.",
                "Mornings stay light, including in September.",
            ],
            figure_paths.get("heatmap"),
        ),
        (
            "Temperature at three stations",
            [
                "Sham Shui Po, Kwun Tong, and Sha Tin, 2023–2025.",
                f"Spearman {_num(temp['spearman'])} on {temp['n']:,} station-days.",
                f"After removing the month mean: {_num(temp['spearman_within_month'])}.",
                temp_line,
            ],
            figure_paths.get("temp"),
        ),
        (
            "Rain at the nearest project gauge",
            [
                f"Hourly Spearman {_num(rain['spearman'])} on {rain['n']:,} hours; within month {_num(rain['spearman_within_month'])}.",
                "The hourly link disappears once the month is removed.",
                f"Daily rain, at least {cover['min_rain_hours']} good hours: {_num(rain_day['spearman'])} overall, {_num(rain_day['spearman_within_month'])} within month (n = {rain_day['n']:,}).",
                f"Matched {cover['matched_rain_hours']:,} of {cover['aqhi_hours']:,} hours. Missing rain stays missing.",
            ],
            figure_paths.get("rain"),
        ),
        (
            "Season in the HKO temperature",
            [
                f"Daily maximum at {summary['n_temp_stations']} HKO stations, 2023–2025.",
                f"Hottest month at the three shared stations: {MONTHS[summary['temp_hottest_month'] - 1]}, {summary['temp_hottest_c']:.1f}°C.",
                f"Coolest month: {MONTHS[summary['temp_coolest_month'] - 1]}, {summary['temp_coolest_c']:.1f}°C.",
                "Lines: Sham Shui Po, Kwun Tong, Sha Tin. No AQHI in this average.",
            ],
            figure_paths.get("hko_temp"),
        ),
        (
            "Season in the rainfall",
            [
                f"Complete days at {summary['n_rain_stations']} HKO sites.",
                f"Wettest month at the three: {MONTHS[summary['rain_wettest_month'] - 1]}, {summary['rain_wettest_mm']:.0f} mm.",
                f"Driest month: {MONTHS[summary['rain_driest_month'] - 1]}, {summary['rain_driest_mm']:.1f} mm.",
                "Open-Meteo at the HKO coordinates. Incomplete days stay out.",
            ],
            figure_paths.get("hko_rain"),
        ),
        (
            "When he can go outside",
            [
                "Poor means AQHI at or above 7.",
                "Go in the morning. Hour 17 is the worst hour.",
                "Avoid September and April. August is the calmest month.",
                "Roadside stations have fewer Low hours, not more poor hours.",
                "Do not wait for rain. That hourly link is gone once the month is removed.",
            ],
            figure_paths.get("hour"),
        ),
        (
            "Optional score, three stations",
            [
                "Sham Shui Po, Kwun Tong, and Sha Tin only. Not the citywide rule.",
                f"Stay in when that score is at least {locked['threshold']:.3f}.",
                f"2024–2025: TPR {_pct(risk['test']['tpr'])}, FPR {_pct(risk['test']['fpr'])}.",
                f"Sent outside for {_pct(risk['test']['go_share'])} of these hours.",
                "The false-positive rate holds. The catch rate falls.",
            ],
            figure_paths.get("curve"),
        ),
        (
            "What this does not say",
            [
                "No graph: the file has station columns, not edges or coordinates.",
                "One logistic score, no model search. CSI is not an AQHI comparison.",
                "Open-Meteo is reanalysis, not the EPD inlet or an official rain gauge.",
                "The threshold was not chosen on 2024–2025. 2026 is unused.",
            ],
            None,
        ),
    ]
    with PdfPages(SLIDES) as pdf:
        for title, bullets, image in pages:
            fig = plt.figure(figsize=(13.33, 7.5))
            fig.patch.set_facecolor("white")
            fig.text(0.06, 0.90, title, fontsize=26, fontweight="bold", color="#111827")
            fig.add_artist(plt.Line2D([0.06, 0.94], [0.84, 0.84], transform=fig.transFigure, color="#b45309", lw=2))
            if image and Path(image).exists():
                ax = fig.add_axes([0.50, 0.12, 0.44, 0.66])
                ax.imshow(plt.imread(image))
                ax.axis("off")
                y = 0.76
                chars = 42
                step = 0.048
            else:
                y = 0.74
                chars = 88
                step = 0.07
            for bullet in bullets:
                lines = textwrap.wrap(bullet, width=chars) or [""]
                for i, line in enumerate(lines):
                    fig.text(0.06, y, ("• " if i == 0 else "  ") + line, fontsize=15, color="#1f2937")
                    y -= step
            pdf.savefig(fig)
            plt.close(fig)
    return SLIDES


def save_tables(tables: dict, weather: dict, rule: dict) -> None:
    ensure_dirs()
    for name in ("hour", "month", "year", "region", "roadside", "station"):
        tables[name].to_csv(TABLES / f"aqhi_by_{name}.csv", index=False)
    tables["hour_month"].to_csv(TABLES / "aqhi_hour_month.csv")
    weather["gauges"].to_csv(TABLES / "aqhi_weather_gauges.csv", index=False)
    weather["rain_bins"].to_csv(TABLES / "aqhi_rain_bins.csv", index=False)
    pd.DataFrame(
        [
            {"link": "temperature", **weather["temp_corr"]},
            {"link": "rainfall_hourly", **weather["rain_corr"]},
            {"link": "rainfall_daily", **weather["rain_daily_corr"]},
        ]
    ).to_csv(TABLES / "aqhi_weather_corr.csv", index=False)
    (TABLES / "aqhi_rule.json").write_text(json.dumps(rule, indent=2), encoding="utf-8")


def run() -> dict:
    copy_raw()
    hourly, audit = load_aqhi()
    frame = analysis_hours(hourly)
    frame.to_parquet(ANALYSIS_PATH, index=False)
    tables = task1_tables(frame)
    figures = plot_task1(frame, tables)
    weather = join_weather(frame)
    figures.update(plot_weather(weather))
    rule = advice_rule(frame)
    save_tables(tables, weather, rule)
    means = weather_season_place()
    figures.update(means["figures"])
    risk = fit_risk(frame, means, rule)
    figures.update(risk["figures"])
    write_contract(audit)
    text = write_answers(audit, frame, tables, weather, risk)
    deck = write_slides(frame, tables, weather, figures, risk)
    print(text)
    print("deck", deck)
    return {"audit": audit, "n": len(frame), "rule": rule, "risk": risk["locked"], "deck": str(deck)}
