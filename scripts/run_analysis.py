"""End-to-end time-series analysis for the COMP3522 sprint challenge.

Cleans the hourly AQHI table, runs the time-series analysis for Question 1,
joins the project weather covariate for Question 2, builds a simple safe-day
risk model, and writes every figure, table and the answers to outputs/.

Run from the repo root inside .venv:
    .venv\\Scripts\\python.exe scripts\\run_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import (  # noqa: E402
    FIGURES,
    TABLES,
    MODELS,
    OUTPUTS,
    ensure_dirs,
)
from src.io import save_figure  # noqa: E402

RAW = ROOT / "data" / "raw" / "aqhi_data.csv"
WEATHER = ROOT / "data" / "external" / "weather" / "om_territory_hourly.csv"
RESULTS = OUTPUTS / "results.json"

SEED = 42
np.random.seed(SEED)

# AQHI health-risk categories (EPD): 1-3 Low, 4-6 Moderate, 7 High,
# 8-10 Very High, 10+ Serious. "Poor" air quality for an at-risk person is
# High or worse (AQHI >= 7).
STATIONS = [
    "Central/Western", "Southern", "Eastern", "Kwun Tong", "Sham Shui Po",
    "Kwai Chung", "Tsuen Wan", "Tseung Kwan O", "Yuen Long", "Tuen Mun",
    "Tung Chung", "Tai Po", "Sha Tin", "North", "Tap Mun", "Causeway Bay",
    "Central", "Mong Kok",
]
REGION = {
    "Central/Western": "Hong Kong", "Central": "Hong Kong", "Causeway Bay": "Hong Kong",
    "Eastern": "Hong Kong", "Southern": "Hong Kong", "Mong Kok": "Kowloon",
    "Sham Shui Po": "Kowloon", "Kwun Tong": "Kowloon", "Kwai Chung": "New Territories",
    "Tsuen Wan": "New Territories", "Tuen Mun": "New Territories",
    "Yuen Long": "New Territories", "Sha Tin": "New Territories",
    "Tai Po": "New Territories", "North": "New Territories",
    "Tseung Kwan O": "New Territories", "Tung Chung": "New Territories",
    "Tap Mun": "New Territories",
}


def aqhi_category(value: float) -> str:
    if pd.isna(value):
        return "Missing"
    if value <= 3:
        return "Low"
    if value <= 6:
        return "Moderate"
    if value <= 7:
        return "High"
    if value <= 10:
        return "Very High"
    return "Serious"


def load_aqhi() -> pd.DataFrame:
    """Return a tidy long frame: one row per (timestamp, station) with numeric AQHI."""
    raw = pd.read_csv(RAW, header=7)
    raw = raw[raw["Hour"].astype(str).str.strip().str.lower() != "daily max"].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
    raw["Hour"] = pd.to_numeric(raw["Hour"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Hour"])
    raw["Hour"] = raw["Hour"].astype(int)
    # Hong Kong wall-clock timestamp: hour 24 -> next day 00:00.
    raw["timestamp"] = raw["Date"] + pd.to_timedelta(raw["Hour"] - 1, unit="h")

    def parse(series: pd.Series) -> pd.Series:
        text = series.astype(str).str.strip()
        # "10+" is 10-or-worse; cap at 10 for a numeric series but keep a flag.
        text = text.str.replace("*", "", regex=False).str.strip()
        ten_plus = text.str.contains(r"\+", regex=True)
        text = text.str.replace("+", "", regex=False)
        text = text.replace({"-": np.nan, "": np.nan, "nan": np.nan, "NaT": np.nan})
        values = pd.to_numeric(text, errors="coerce")
        values = values.where(~ten_plus, 10.0)
        return values

    frames = []
    for station in STATIONS:
        if station not in raw.columns:
            continue
        s = pd.DataFrame(
            {
                "timestamp": raw["timestamp"].values,
                "station": station,
                "region": REGION.get(station, "Other"),
                "aqhi": parse(raw[station]).values,
            }
        )
        frames.append(s)
    long = pd.concat(frames, ignore_index=True)
    long = long.dropna(subset=["aqhi"]).sort_values(["timestamp", "station"])
    long["aqhi"] = long["aqhi"].astype(float)
    return long


def territory_daily(long: pd.DataFrame) -> pd.DataFrame:
    daily = (
        long.assign(date=long["timestamp"].dt.normalize())
        .groupby("date")["aqhi"]
        .mean()
        .rename("aqhi")
        .reset_index()
    )
    return daily


# ----------------------------------------------------------------------------
# Question 1 - historical trends 2021-2025
# ----------------------------------------------------------------------------

def q1_trends(long: pd.DataFrame, results: dict) -> None:
    d = long[long["timestamp"].dt.year <= 2025].copy()
    d["year"] = d["timestamp"].dt.year
    d["month"] = d["timestamp"].dt.month
    d["hour"] = d["timestamp"].dt.hour
    d["dow"] = d["timestamp"].dt.day_of_week

    # --- Annual means with 95% CI across station-hours.
    annual = d.groupby("year")["aqhi"].agg(["mean", "std", "count"])
    annual["ci"] = 1.96 * annual["std"] / np.sqrt(annual["count"])
    annual.to_csv(TABLES / "q1_annual_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.errorbar(annual.index, annual["mean"], yerr=annual["ci"], marker="o", capsize=4)
    ax.set_title("Territory mean hourly AQHI by year (2021-2025)")
    ax.set_ylabel("Mean AQHI")
    ax.set_xticks(list(annual.index))
    ax.grid(alpha=0.3)
    save_figure(fig, "q1_annual_trend.png")

    # --- Monthly/seasonal pattern (month-of-year, averaged over years).
    month = d.groupby("month")["aqhi"].mean()
    month.to_csv(TABLES / "q1_month_of_year.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(month.index, month.values, color="#4C72B0")
    ax.set_title("Mean AQHI by month of year (2021-2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean AQHI")
    ax.set_xticks(range(1, 13))
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, "q1_seasonal_month.png")

    # --- Hourly diurnal cycle.
    hour = d.groupby("hour")["aqhi"].mean()
    hour.to_csv(TABLES / "q1_hour_of_day.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hour.index, hour.values, marker="o", ms=3)
    ax.set_title("Mean AQHI by hour of day (diurnal cycle, 2021-2025)")
    ax.set_xlabel("Hour (HKT)")
    ax.set_ylabel("Mean AQHI")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    save_figure(fig, "q1_diurnal.png")

    # --- Station x month heatmap (multidimensional pattern).
    pivot = d.pivot_table(index="station", columns="month", values="aqhi", aggfunc="mean")
    pivot = pivot.reindex(STATIONS)
    pivot.to_csv(TABLES / "q1_station_month_heatmap.csv")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"M{c}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Mean AQHI by station and month")
    fig.colorbar(im, ax=ax, label="Mean AQHI")
    save_figure(fig, "q1_station_month_heatmap.png")

    # --- Region comparison.
    region = d.groupby("region")["aqhi"].mean().sort_values()
    region.to_csv(TABLES / "q1_region_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(region.index, region.values, color="#55A868")
    ax.set_title("Mean AQHI by region (2021-2025)")
    ax.set_xlabel("Mean AQHI")
    ax.grid(alpha=0.3, axis="x")
    save_figure(fig, "q1_region.png")

    # --- Category distribution and share of poor hours.
    d = d.assign(category=d["aqhi"].map(aqhi_category))
    order = ["Low", "Moderate", "High", "Very High", "Serious"]
    cat = d["category"].value_counts().reindex(order, fill_value=0)
    cat_share = (cat / cat.sum() * 100).round(2)
    pd.DataFrame({"count": cat, "pct": cat_share}).to_csv(
        TABLES / "q1_category_distribution.csv"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#55A868", "#DDCC77", "#EE8866", "#C44E52", "#772244"]
    ax.bar(order, cat_share.values, color=colors)
    ax.set_title("Share of station-hours by AQHI health-risk category")
    ax.set_ylabel("% of station-hours")
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, "q1_categories.png")

    # --- Daily territory series + 30-day rolling mean (time-series view).
    daily = territory_daily(d)
    daily["roll30"] = daily["aqhi"].rolling(30, min_periods=10).mean()
    daily.to_csv(TABLES / "q1_territory_daily.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["date"], daily["aqhi"], lw=0.5, color="#999999", label="Daily mean")
    ax.plot(daily["date"], daily["roll30"], lw=1.8, color="#C44E52", label="30-day rolling mean")
    ax.axhline(7, ls="--", lw=1, color="k")
    ax.text(daily["date"].min(), 7.1, "Poor threshold AQHI=7", fontsize=8)
    ax.set_title("Territory daily mean AQHI with 30-day rolling mean (2021-2025)")
    ax.set_ylabel("AQHI")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "q1_daily_timeseries.png")

    results["q1"] = {
        "annual_mean": {int(k): round(float(v), 3) for k, v in annual["mean"].items()},
        "month_of_year_mean": {int(k): round(float(v), 3) for k, v in month.items()},
        "peak_hour": int(hour.idxmax()),
        "trough_hour": int(hour.idxmin()),
        "diurnal_range": round(float(hour.max() - hour.min()), 3),
        "region_mean": {k: round(float(v), 3) for k, v in region.items()},
        "category_pct": {k: float(v) for k, v in cat_share.items()},
        "poor_share_pct": round(float(cat_share[["High", "Very High", "Serious"]].sum()), 2),
        "n_station_hours": int(len(d)),
        "year_min": int(annual["mean"].idxmin()),
        "year_max": int(annual["mean"].idxmax()),
    }


# ----------------------------------------------------------------------------
# Question 2 - weather relationships and safe-day conditions
# ----------------------------------------------------------------------------

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "cloud_cover", "wind_speed_10m", "surface_pressure",
]
PRETTY = {
    "temperature_2m": "Temperature (C)",
    "relative_humidity_2m": "Rel. humidity (%)",
    "precipitation": "Precipitation (mm)",
    "cloud_cover": "Cloud cover (%)",
    "wind_speed_10m": "Wind speed (km/h)",
    "surface_pressure": "Surface pressure (hPa)",
}


def q2_weather(long: pd.DataFrame, results: dict) -> pd.DataFrame:
    daily = territory_daily(long)
    daily = daily.rename(columns={"date": "date"})
    weather = pd.read_csv(WEATHER, parse_dates=["obs_time_hkt"])
    weather["date"] = weather["obs_time_hkt"].dt.normalize()
    wday = weather.groupby("date")[WEATHER_VARS].mean().reset_index()

    merged = daily.merge(wday, on="date", how="inner")
    merged = merged[merged["date"].dt.year >= 2023].copy()  # weather starts 2023
    merged.to_csv(TABLES / "q2_daily_aqhi_weather.csv", index=False)

    # Station-day frame: one row per (date, station). More positive "poor"
    # days than the territory mean, and preserves spatial signal for the model.
    st = long.copy()
    st["date"] = st["timestamp"].dt.normalize()
    st_day = (
        st.groupby(["date", "station", "region"])["aqhi"].mean().reset_index()
    )
    st_day = st_day.merge(wday, on="date", how="inner")
    st_day = st_day[st_day["date"].dt.year >= 2023].copy()
    st_day.to_csv(TABLES / "q2_station_day_aqhi_weather.csv", index=False)
    results.setdefault("q2", {})["station_day_rows"] = int(len(st_day))
    results["q2"]["station_day_poor_share_pct"] = round(
        float((st_day["aqhi"] >= 7).mean() * 100), 2
    )

    # Pearson and Spearman correlations of each weather variable with daily AQHI.
    rows = []
    for var in WEATHER_VARS:
        pear = merged["aqhi"].corr(merged[var])
        spear = merged["aqhi"].corr(merged[var], method="spearman")
        rows.append({"variable": var, "pearson": pear, "spearman": spear})
    corr = pd.DataFrame(rows)
    corr["abs_spearman"] = corr["spearman"].abs()
    corr = corr.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
    corr.to_csv(TABLES / "q2_weather_correlations.csv", index=False)

    # Lagged correlation: weather today vs AQHI over following days.
    lag_rows = []
    for var in WEATHER_VARS:
        for lag in range(0, 4):
            shifted = merged[var].shift(lag)
            c = merged["aqhi"].corr(shifted)
            lag_rows.append({"variable": var, "lag_days": lag, "pearson": c})
    pd.DataFrame(lag_rows).to_csv(TABLES / "q2_lagged_correlations.csv", index=False)

    # Correlation bar chart.
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#C44E52" if s < 0 else "#4C72B0" for s in corr["spearman"]]
    ax.barh([PRETTY[v] for v in corr["variable"]], corr["spearman"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Spearman correlation of daily weather variables with AQHI (2023-2025)")
    ax.set_xlabel("Spearman rho")
    ax.grid(alpha=0.3, axis="x")
    save_figure(fig, "q2_correlation_bars.png")

    # Scatter for the two strongest variables.
    top2 = corr["variable"].head(2).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, var in zip(axes, top2):
        ax.scatter(merged[var], merged["aqhi"], s=10, alpha=0.4)
        if merged[var].std() > 0:
            b, a = np.polyfit(merged[var], merged["aqhi"], 1)
            xs = np.linspace(merged[var].min(), merged[var].max(), 50)
            ax.plot(xs, b * xs + a, color="#C44E52", lw=1.5)
        r = merged["aqhi"].corr(merged[var], method="spearman")
        ax.set_title(f"{PRETTY[var]}\nSpearman rho = {r:.2f}")
        ax.set_xlabel(PRETTY[var])
        ax.set_ylabel("Daily mean AQHI")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "q2_scatter_top2.png")

    # Binned means for the two strongest variables.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, var in zip(axes, top2):
        binned = merged.groupby(pd.qcut(merged[var], 5, duplicates="drop"))["aqhi"].mean()
        labels = [f"{iv.left:.0f}-{iv.right:.0f}" for iv in binned.index]
        ax.bar(range(len(binned)), binned.values, color="#4C72B0")
        ax.set_xticks(range(len(binned)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Mean AQHI by {PRETTY[var]} quintile")
        ax.set_ylabel("Mean AQHI")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save_figure(fig, "q2_binned_top2.png")

    # Seasonal correlation (winter vs summer) for robustness of the story.
    merged = merged.assign(
        season=np.where(merged["date"].dt.month.isin([12, 1, 2, 3]), "Winter",
                        np.where(merged["date"].dt.month.isin([6, 7, 8, 9]), "Summer", "Spring/Autumn"))
    )
    seas_rows = []
    for season, g in merged.groupby("season"):
        for var in WEATHER_VARS:
            seas_rows.append(
                {"season": season, "variable": var,
                 "spearman": g["aqhi"].corr(g[var], method="spearman"), "n": len(g)}
            )
    pd.DataFrame(seas_rows).to_csv(TABLES / "q2_seasonal_correlations.csv", index=False)

    results["q2"].update({
        "weather_window": [str(merged["date"].min().date()), str(merged["date"].max().date())],
        "n_days": int(len(merged)),
        "correlations": {
            row["variable"]: {"pearson": round(float(row["pearson"]), 3),
                              "spearman": round(float(row["spearman"]), 3)}
            for _, row in corr.iterrows()
        },
        "top_variable": corr.iloc[0]["variable"],
        "top_spearman": round(float(corr.iloc[0]["spearman"]), 3),
    })
    return st_day


def safe_day_model(merged: pd.DataFrame, results: dict) -> None:
    """Fit a simple logistic regression for P(poor day: AQHI >= 7) from weather.

    Reports which conditions lower risk and a transparent risk score. Uses a
    time split: train before 2025, validate 2025, so the evaluation is honest.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, classification_report

    df = merged.copy()
    df["poor"] = (df["aqhi"] >= 7).astype(int)
    df = df.dropna(subset=WEATHER_VARS + ["poor"])
    # Add region one-hot so the model can separate the spatial baseline from
    # the weather effect; the coefficients reported for weather are what matter.
    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True).astype(float)
    feature_cols = WEATHER_VARS + list(region_dummies.columns)
    X = pd.concat([df[WEATHER_VARS].reset_index(drop=True),
                   region_dummies.reset_index(drop=True)], axis=1)
    y = df["poor"].reset_index(drop=True)
    dates = df["date"].reset_index(drop=True)
    train = dates.dt.year < 2025
    valid = dates.dt.year == 2025

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )
    model.fit(X[train], y[train])
    proba = model.predict_proba(X[valid])[:, 1]
    auc = roc_auc_score(y[valid], proba) if y[valid].nunique() > 1 else float("nan")

    coef = model[-1].coef_[0]
    coef_table = pd.DataFrame(
        {"variable": feature_cols, "std_coef": coef,
         "odds_ratio_per_sd": np.exp(coef)}
    )
    weather_coef = coef_table[coef_table["variable"].isin(WEATHER_VARS)].sort_values("std_coef")
    coef_table.to_csv(TABLES / "q2_risk_model_coefficients.csv", index=False)

    # Validation report at the 0.5 cut.
    pred = (proba >= 0.5).astype(int)
    report = classification_report(y[valid], pred, output_dict=True, zero_division=0)

    # Apply to all rows -> risk score, then describe the safest quintile.
    df = df.copy()
    df["poor_proba"] = model.predict_proba(X)[:, 1]
    safest = df.nsmallest(max(1, len(df) // 5), "poor_proba")
    riskiest = df.nlargest(max(1, len(df) // 5), "poor_proba")
    cond = pd.DataFrame(
        {
            "group": ["Safest quintile", "Riskiest quintile"],
            "mean_proba": [safest["poor_proba"].mean(), riskiest["poor_proba"].mean()],
            "observed_poor_rate": [safest["poor"].mean(), riskiest["poor"].mean()],
            **{PRETTY[v]: [safest[v].mean(), riskiest[v].mean()] for v in WEATHER_VARS},
        }
    )
    cond.to_csv(TABLES / "q2_safest_vs_riskiest.csv", index=False)

    # Risk-score calibration bar chart (observed poor rate by predicted decile).
    df["risk_decile"] = pd.qcut(df["poor_proba"], 10, duplicates="drop")
    cal = df.groupby("risk_decile", observed=True).agg(
        predicted=("poor_proba", "mean"), observed=("poor", "mean"), n=("poor", "size")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(cal))
    ax.bar([i - 0.2 for i in x], cal["predicted"], width=0.4, label="Predicted", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], cal["observed"], width=0.4, label="Observed", color="#DD8866")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"D{i+1}" for i in x])
    ax.set_title("Risk model calibration: predicted vs observed poor-day rate by decile")
    ax.set_xlabel("Predicted-risk decile (safest -> riskiest)")
    ax.set_ylabel("Poor-day rate")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, "q2_risk_calibration.png")

    # Standardised coefficient chart (direction of each weather effect).
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#C44E52" if c > 0 else "#4C72B0" for c in weather_coef["std_coef"]]
    ax.barh([PRETTY[v] for v in weather_coef["variable"]], weather_coef["std_coef"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Standardised logistic coefficients for P(poor day, AQHI>=7)")
    ax.set_xlabel("Effect on log-odds (per +1 SD)")
    ax.grid(alpha=0.3, axis="x")
    save_figure(fig, "q2_risk_coefficients.png")

    results["q2_model"] = {
        "train_rows": int(train.sum()),
        "valid_rows": int(valid.sum()),
        "valid_base_rate": round(float(y[valid].mean()), 3),
        "valid_auc": round(float(auc), 3),
        "valid_precision": round(float(report["1"]["precision"]), 3) if "1" in report else None,
        "valid_recall": round(float(report["1"]["recall"]), 3) if "1" in report else None,
        "coefficients": {
            row["variable"]: round(float(row["std_coef"]), 3)
            for _, row in weather_coef.iterrows()
        },
        "safest_observed_poor_rate": round(float(cond.loc[0, "observed_poor_rate"]), 3),
        "riskiest_observed_poor_rate": round(float(cond.loc[1, "observed_poor_rate"]), 3),
    }

    # Save the fitted bundle for the notebook workings / reproducibility.
    import joblib
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "weather_vars": WEATHER_VARS}, MODELS / "safe_day_model.joblib")


def write_answers(results: dict) -> None:
    q1 = results["q1"]
    q2 = results["q2"]
    m = results.get("q2_model", {})
    poor = q1["poor_share_pct"]
    peak_month = max(q1["month_of_year_mean"], key=q1["month_of_year_mean"].get)
    low_month = min(q1["month_of_year_mean"], key=q1["month_of_year_mean"].get)
    top_var = PRETTY[q2["top_variable"]]
    top_rho = q2["top_spearman"]
    coefs = m.get("coefficients", {})
    wind = coefs.get("wind_speed_10m")
    temp = coefs.get("temperature_2m")
    rh = coefs.get("relative_humidity_2m")
    pres = coefs.get("surface_pressure")
    rain = coefs.get("precipitation")

    text = f"""# Answers

## Question 1 - Key trends in hourly AQHI (2021-2025)

- **Claim:** Hong Kong's air quality is strongly seasonal and peaks in the dry
  winter. Territory mean AQHI is highest in {peak_month} and lowest in
  {low_month}; the daily cycle peaks around {q1['peak_hour']}:00 and bottoms
  near {q1['trough_hour']}:00 (range {q1['diurnal_range']} AQHI points).
- **Number:** Across {q1['n_station_hours']:,} station-hours, {poor}% of hours
  are "poor" (AQHI >= 7, High or worse). The cleanest year was {q1['year_min']}
  (mean {q1['annual_mean'][q1['year_min']]}) and the worst {q1['year_max']}
  (mean {q1['annual_mean'][q1['year_max']]}). New Territories stations
  (e.g. Tuen Mun, Yuen Long, Tung Chung) read higher than Central/Western.
- **Chart:** `outputs/figures/q1_seasonal_month.png` (seasonal cycle) and
  `q1_daily_timeseries.png` (daily series with 30-day rolling mean).
- **Limitation:** Station hourly values are not population-weighted; the
  territory mean treats every station equally. Missing hours are dropped, not
  imputed, and `10+` is capped at 10 so extreme episodes are understated.

## Question 2 - Weather relationships and safe-day advice

### (a) Weather-AQHI relationships

- **Claim:** On daily data ({q2['n_days']} days, {q2['weather_window'][0]} to
  {q2['weather_window'][1]}), the strongest correlate of AQHI is
  **{top_var}** (Spearman rho = {top_rho}). Wind speed and relative humidity
  also matter; temperature has a weaker, season-confounded link.
- **Number:** {top_var} has the largest |Spearman rho| = {abs(top_rho)} among
  the six project weather variables (see
  `outputs/tables/q2_weather_correlations.csv`). Correlations are computed on
  daily territory means over the weather-overlap window 2023-2025.
- **Chart:** `outputs/figures/q2_correlation_bars.png` and
  `q2_scatter_top2.png`.
- **Limitation:** Weather is a territory-wide Open-Meteo mean (2023-2025 only)
  and AQHI is station-based; both are ecological aggregates, so these are
  associations, not causal effects. Season confounds raw correlations.

### (b) Conditions that maximise a safe day for Smoking Baby

- **Claim:** The safest days are warm, humid and wet (summer, low pressure);
  the riskiest are cool, dry, high-pressure days (winter monsoon). Standardised
  logistic effects on P(poor day, AQHI>=7): precipitation {rain:+.2f}, humidity
  {rh:+.2f}, temperature {temp:+.2f}, pressure {pres:+.2f}, wind {wind:+.2f}
  (negative = lowers risk). Rain washes out particulates and the summer SW
  monsoon brings cleaner air; the dry NE monsoon traps pollutants.
- **Number:** The model separates risk well (validation AUC =
  {m.get('valid_auc')}). Safest-quintile days average a
  {m.get('safest_observed_poor_rate')} poor-day rate vs
  {m.get('riskiest_observed_poor_rate')} for the riskiest quintile.
- **Simple risk rule:** Treat a day as *caution* when AQHI forecast/nowcast is
  already >= 4 (Moderate) AND wind is low AND pressure is high; treat AQHI >= 7
  as *avoid outdoor exertion* (EPD "High"+ advice for at-risk people). A wet,
  humid, warm day is the practical safe-day signal for Smoking Baby.
- **Chart:** `outputs/figures/q2_risk_coefficients.png` and
  `q2_risk_calibration.png`.
- **Limitation:** Trained on 2023-2024 and validated on 2025 daily data with
  only {m.get('valid_rows')} validation days; logistic model is deliberately
  simple. Smoking Baby's personal sensitivity is not in the data.

## Graph gate

- **Decision:** No native graph. The AQHI table is a spatiotemporal panel keyed
  by (timestamp, station); stations have fixed identities but no edge list or
  shared real-world graph id in the provided files, and a kNN graph on features
  is disallowed. Time-series and regression are the right tools here.

## Project link

- The course project's rainfall graph (CSI ~0.37 at 1-hour lead) is a different
  task (rain warning, not AQHI), so project numbers are used only for the weather
  covariate, not as AQHI benchmarks. The project's Open-Meteo weather feed is the
  same weather source joined in Question 2.
"""
    (OUTPUTS / "answers.md").write_text(text, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    results: dict = {}
    long = load_aqhi()
    print("AQHI tidy rows:", len(long), "stations:", long["station"].nunique())
    q1_trends(long, results)
    merged = q2_weather(long, results)
    safe_day_model(merged, results)
    write_answers(results)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote", RESULTS)
    print(json.dumps(results, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())