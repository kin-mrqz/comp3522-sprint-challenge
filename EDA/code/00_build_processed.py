"""00 - raw AQHI csv -> processed analysis tables plus a cleaning log.

Reads data/raw/aqhi_data.csv (unchanged EPD export).
Writes:
  data/processed/aqhi_hourly.parquet          one row per timestamp x station
  data/processed/aqhi_daily_station.parquet   one row per date x station
  data/processed/aqhi_daily_territory.parquet one row per date (all stations)
  data/processed/aqhi_wide_hourly.parquet     timestamp x station matrix
  data/processed/station_metadata.csv
  data/processed/data_quality_station_year.csv
  data/processed/cleaning_log.csv
  data/processed/t00_file_audit.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from eda_lib import META, PROC, RAW, SEASON, TAB, save_table

HEADER_ROW = 7  # line 8 of the file; lines 1-7 are the EPD notes


def parse_value(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (aqhi, substituted, censored). '10+' is 10 and flagged."""
    text = series.astype("string").str.strip()
    substituted = text.str.contains(r"\*", regex=True, na=False)
    clean = text.str.replace("*", "", regex=False).str.strip()
    censored = clean.str.startswith("10+", na=False)
    clean = clean.str.replace("+", "", regex=False)
    aqhi = pd.to_numeric(clean, errors="coerce")
    aqhi = aqhi.where(aqhi.between(1, 10))
    aqhi.loc[censored] = 10.0
    return aqhi, substituted, censored


def main() -> None:
    print(f"reading {RAW}")
    raw = pd.read_csv(RAW, header=HEADER_ROW, dtype=str, skip_blank_lines=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    stations = [c for c in raw.columns if c not in ("Date", "Hour")]
    expected = META["station"].tolist()
    if stations != expected:
        missing = [s for s in expected if s not in stations]
        extra = [s for s in stations if s not in expected]
        raise SystemExit(f"station columns differ from metadata; missing={missing} extra={extra}")
    print(f"raw rows={len(raw):,}  columns={len(raw.columns)}  stations={len(stations)}")

    hour_label = raw["Hour"].astype("string").str.strip()
    is_daily_max = hour_label.eq("Daily Max")
    hourly_raw = raw.loc[~is_daily_max].copy()
    daily_max_raw = raw.loc[is_daily_max].copy()

    hourly_raw["date"] = pd.to_datetime(hourly_raw["Date"], format="%d/%m/%Y", errors="raise")
    hour_num = pd.to_numeric(hourly_raw["Hour"], errors="raise").astype(int)
    if not hour_num.between(1, 24).all():
        raise SystemExit("hour outside 1..24")
    hourly_raw["hour_of_day"] = hour_num - 1  # hour 1 -> 00:00-00:59 (hour beginning)
    hourly_raw["timestamp"] = hourly_raw["date"] + pd.to_timedelta(hourly_raw["hour_of_day"], unit="h")

    long = hourly_raw.melt(
        id_vars=["date", "timestamp", "hour_of_day", "Hour"],
        value_vars=stations,
        var_name="station",
        value_name="raw_value",
    )
    long["raw_value"] = long["raw_value"].astype("string").str.strip()
    long["aqhi"], long["substituted"], long["censored"] = parse_value(long["raw_value"])
    long["is_missing_input"] = long["aqhi"].isna()

    long = long.merge(META, on="station", how="left", validate="many_to_one")
    long["year"] = long["timestamp"].dt.year
    long["month"] = long["timestamp"].dt.month
    long["day"] = long["timestamp"].dt.day
    long["dow"] = long["timestamp"].dt.day_of_week  # 0=Monday
    long["season"] = long["month"].map(SEASON)
    long["hour_label"] = long["hour_of_day"].map(lambda h: f"{h:02d}:00")
    long["band"] = pd.cut(
        long["aqhi"].clip(lower=1, upper=10),
        bins=[0, 3, 6, 7, 9, 10],
        labels=["Low (1-3)", "Moderate (4-6)", "High (7)", "Very High (8-10)", "Serious (10+)"],
        include_lowest=True,
    )
    long = long[[
        "timestamp", "date", "year", "month", "day", "hour_of_day", "hour_label", "dow", "season",
        "station", "slug", "region", "station_type", "is_background",
        "raw_value", "aqhi", "substituted", "censored", "is_missing_input", "band",
    ]].sort_values(["station", "timestamp"]).reset_index(drop=True)

    # ---- daily max rows ----------------------------------------------------
    daily_max_raw["date"] = pd.to_datetime(daily_max_raw["Date"], format="%d/%m/%Y", errors="raise")
    dm = daily_max_raw.melt(
        id_vars=["date"], value_vars=stations, var_name="station", value_name="raw_value"
    )
    dm["raw_value"] = dm["raw_value"].astype("string").str.strip()
    dm["aqhi_file_daily_max"], dm["dm_substituted"], dm["dm_censored"] = parse_value(dm["raw_value"])

    # ---- station-day aggregation ------------------------------------------
    good = long.dropna(subset=["aqhi"])
    grouped = good.groupby(["date", "station"], observed=True)
    daily = grouped.agg(
        n_obs=("aqhi", "size"),
        mean=("aqhi", "mean"),
        median=("aqhi", "median"),
        max=("aqhi", "max"),
        min=("aqhi", "min"),
        sd=("aqhi", "std"),
        p90=("aqhi", lambda s: s.quantile(0.90)),
        n_ge7=("aqhi", lambda s: int((s >= 7).sum())),
        n_ge8=("aqhi", lambda s: int((s >= 8).sum())),
        n_substituted=("substituted", "sum"),
        n_censored=("censored", "sum"),
    ).reset_index()

    peak = good.loc[good.groupby(["date", "station"], observed=True)["aqhi"].idxmax(),
                    ["date", "station", "hour_of_day"]].rename(columns={"hour_of_day": "peak_hour"})
    daily = daily.merge(peak, on=["date", "station"], how="left")
    first7 = (good[good["aqhi"] >= 7]
              .groupby(["date", "station"], observed=True)["hour_of_day"].min()
              .rename("first_hour_ge7").reset_index())
    daily = daily.merge(first7, on=["date", "station"], how="left")
    daily = daily.merge(META, on="station", how="left", validate="many_to_one")
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["season"] = daily["month"].map(SEASON)
    daily["daily_max_file"] = dm.set_index(["date", "station"])["aqhi_file_daily_max"].reindex(
        pd.MultiIndex.from_frame(daily[["date", "station"]])
    ).to_numpy()
    daily["daily_max_matches_file"] = np.isclose(
        daily["max"].to_numpy(), daily["daily_max_file"].to_numpy(), equal_nan=True
    )
    daily = daily.sort_values(["station", "date"]).reset_index(drop=True)

    # ---- hourly territory series ------------------------------------------
    hourly_max = good.groupby("timestamp", observed=True)["aqhi"].max().rename("max_any_station")
    hourly_mean = good.groupby("timestamp", observed=True)["aqhi"].mean().rename("mean_station_mean")
    hourly_n = good.groupby("timestamp", observed=True)["aqhi"].size().rename("n_stations")
    hourly_ge7 = (good.assign(ge7=good["aqhi"] >= 7)
                  .groupby("timestamp", observed=True)["ge7"].sum().rename("n_stations_ge7"))
    territory_hour = pd.concat([hourly_mean, hourly_max, hourly_n, hourly_ge7], axis=1).reset_index()
    territory_hour.to_parquet(PROC / "aqhi_territory_hourly.parquet", index=False)

    # ---- daily territory aggregation --------------------------------------
    td = daily.groupby("date", observed=True).agg(
        n_stations_reporting=("n_obs", lambda s: int((s > 0).sum())),
        mean_station_mean=("mean", "mean"),
        median_station_mean=("mean", "median"),
        max_station_max=("max", "max"),
        min_station_min=("min", "min"),
        spatial_sd=("mean", "std"),
        n_stations_ge7=("n_ge7", lambda s: int((s > 0).sum())),
        n_stations_all_hours_ge7=("n_ge7", lambda s: int((s == 24).sum())),
        total_hours_ge7=("n_ge7", "sum"),
        pct_substituted=("n_substituted", lambda s: float(s.sum()) / max(s.sum() * 0 + 24 * len(s), 1)),
    ).reset_index()
    td["spatial_range"] = daily.groupby("date", observed=True)["mean"].agg(lambda s: s.max() - s.min()).to_numpy()
    td = td.merge(
        territory_hour.assign(date=territory_hour["timestamp"].dt.normalize())
        .groupby("date", observed=True)
        .agg(hours_any_ge7=("n_stations_ge7", lambda s: int((s > 0).sum())),
             max_stations_ge7=("n_stations_ge7", "max"))
        .reset_index(),
        on="date", how="left",
    )
    td["year"] = td["date"].dt.year
    td["month"] = td["date"].dt.month
    td["season"] = td["month"].map(SEASON)
    td["dow"] = td["date"].dt.day_of_week
    td["daily_max_band"] = pd.cut(
        td["max_station_max"].clip(lower=1, upper=10),
        bins=[0, 3, 6, 7, 9, 10],
        labels=["Low (1-3)", "Moderate (4-6)", "High (7)", "Very High (8-10)", "Serious (10+)"],
        include_lowest=True,
    )

    # full calendar, gaps kept as NaN (not interpolated)
    calendar = pd.DataFrame({"date": pd.date_range(td["date"].min(), td["date"].max(), freq="D")})
    td = calendar.merge(td, on="date", how="left")
    td["in_file"] = td["n_stations_reporting"].notna()
    td["year"] = td["date"].dt.year
    td["month"] = td["date"].dt.month
    td["season"] = td["month"].map(SEASON)
    td["dow"] = td["date"].dt.day_of_week
    td = td.sort_values("date").reset_index(drop=True)

    # ---- wide hourly matrix ----------------------------------------------
    wide = long.pivot(index="timestamp", columns="station", values="aqhi")
    wide = wide.reindex(columns=stations)
    wide.index.name = "timestamp"

    # ---- data quality by station-year ------------------------------------
    dq_rows = []
    n_calendar_days = (hourly_raw["date"].max() - hourly_raw["date"].min()).days + 1
    for station in stations:
        sub = long[long["station"] == station]
        for year, block in sub.groupby("year", observed=True):
            days_in_file = block["date"].nunique()
            expected_hours = days_in_file * 24
            dq_rows.append({
                "station": station,
                "year": int(year),
                "days_in_file": days_in_file,
                "expected_hours": expected_hours,
                "observed_hours": int(block["aqhi"].notna().sum()),
                "missing_pct": float(block["aqhi"].isna().mean()),
                "substituted_pct": float(block["substituted"].mean()),
                "censored_pct": float(block["censored"].mean()),
            })
    dq = pd.DataFrame(dq_rows)

    # ---- audit / cleaning log --------------------------------------------
    dup = int(raw.duplicated(subset=["Date", "Hour"]).sum())
    dm_match = float(daily["daily_max_matches_file"].mean())
    audit = pd.DataFrame([
        ("raw_rows_including_daily_max", len(raw)),
        ("raw_columns", len(raw.columns)),
        ("station_columns", len(stations)),
        ("hourly_rows", len(hourly_raw)),
        ("daily_max_rows", len(daily_max_raw)),
        ("date_min", str(hourly_raw["date"].min().date())),
        ("date_max", str(hourly_raw["date"].max().date())),
        ("calendar_days_span", n_calendar_days),
        ("days_in_file", int(hourly_raw["date"].nunique())),
        ("calendar_days_missing", int(n_calendar_days - hourly_raw["date"].nunique())),
        ("duplicate_date_hour_rows", dup),
        ("hour_values", "1..24 plus 'Daily Max'"),
        ("cells_in_hourly_long", len(long)),
        ("cells_missing", int(long["aqhi"].isna().sum())),
        ("cells_substituted", int(long["substituted"].sum())),
        ("cells_censored_10plus", int(long["censored"].sum())),
        ("daily_max_equals_hourly_max_share", dm_match),
        ("stations", ", ".join(stations)),
    ], columns=["metric", "value"])
    audit.to_csv(PROC / "t00_file_audit.csv", index=False)
    # keep a copy in the EDA tables folder as well
    save_table(audit, "t00_file_audit.csv")

    missing_days = sorted(set(pd.date_range(hourly_raw["date"].min(), hourly_raw["date"].max(), freq="D"))
                          - set(hourly_raw["date"]))
    cleaning = pd.DataFrame([
        {"rule": "skip EPD preamble", "source": "rows 1-7", "rows_affected": 7,
         "why": "title and four numbered notes, no tabular content"},
        {"rule": "split Daily Max rows out of hourly table", "source": "Hour == 'Daily Max'",
         "rows_affected": int(len(daily_max_raw)),
         "why": "Daily Max is a derived summary; keeping it would weight every day twice"},
        {"rule": "strip '*' substitution marker", "source": "value cells",
         "rows_affected": int(long["substituted"].sum()),
         "why": "asterisk means the value came from the most similar station, not this station"},
        {"rule": "map '10+' to 10 with censored flag", "source": "value cells",
         "rows_affected": int(long["censored"].sum()),
         "why": "AQHI top category is open-ended; 10 is a lower bound"},
        {"rule": "map '-' / '*' to missing", "source": "value cells",
         "rows_affected": int((long["raw_value"].isin(["-", "-*", "*"]) | long["raw_value"].isna()).sum()),
         "why": "no AQHI reported; left missing, never imputed"},
        {"rule": "keep calendar with 38 absent dates", "source": "date index",
         "rows_affected": int(n_calendar_days - hourly_raw["date"].nunique()),
         "why": "2023-08-25..31 and 2024-12-01..31 absent from the export; gaps stay visible"},
        {"rule": "duplicate (date, hour) check", "source": "raw", "rows_affected": dup,
         "why": "0 means the date-hour key is unique before melting"},
        {"rule": "Daily Max equals max of hourly values", "source": "derived check",
         "rows_affected": int(len(daily)), "why": f"share matching = {dm_match:.4f}"},
    ])
    cleaning.to_csv(PROC / "cleaning_log.csv", index=False)

    # ---- write -------------------------------------------------------------
    long.to_parquet(PROC / "aqhi_hourly.parquet", index=False)
    daily.to_parquet(PROC / "aqhi_daily_station.parquet", index=False)
    td.to_parquet(PROC / "aqhi_daily_territory.parquet", index=False)
    wide.reset_index().to_parquet(PROC / "aqhi_wide_hourly.parquet", index=False)
    dq.to_csv(PROC / "data_quality_station_year.csv", index=False)
    META.to_csv(PROC / "station_metadata.csv", index=False)

    print(f"hourly long rows : {len(long):,}")
    print(f"station-days     : {len(daily):,}")
    print(f"territory days   : {len(td):,} ({int(td['in_file'].sum())} with data)")
    print(f"missing cells    : {long['aqhi'].isna().sum():,} ({long['aqhi'].isna().mean():.2%})")
    print(f"substituted cells: {long['substituted'].sum():,} ({long['substituted'].mean():.2%})")
    print(f"censored 10+     : {long['censored'].sum():,}")
    print(f"daily max match  : {dm_match:.6f}")
    print("absent dates     :", ", ".join(str(d.date()) for d in missing_days))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("no arguments; edit constants in the script")
    main()
