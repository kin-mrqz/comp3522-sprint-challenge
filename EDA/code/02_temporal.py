"""02 - temporal structure: trend, season, hour, weekday, persistence.

Figures F07-F20, tables t08-t16.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from eda_lib import (BAND_COLOR, BAND_ORDER, FULL_YEARS, METHOD_CHANGE, PROC, SEASON,
                     SEASON_ORDER, TAB, save_fig, save_table, stamp, write_findings)

h = pd.read_parquet(PROC / "aqhi_hourly.parquet")
td = pd.read_parquet(PROC / "aqhi_daily_territory.parquet")
th = pd.read_parquet(PROC / "aqhi_territory_hourly.parquet")
# pandas 3 keeps nullable dtypes; plotting and numpy need plain float64
h["aqhi"] = h["aqhi"].astype("float64")
for _col in ["mean_station_mean", "max_station_max", "min_station_min", "spatial_sd", "spatial_range"]:
    td[_col] = td[_col].astype("float64")
th["mean_station_mean"] = th["mean_station_mean"].astype("float64")
th["max_any_station"] = th["max_any_station"].astype("float64")
td["date"] = pd.to_datetime(td["date"])
th["timestamp"] = pd.to_datetime(th["timestamp"])
daily = td.set_index("date")["mean_station_mean"]
hourly = th.set_index("timestamp")["mean_station_mean"]
findings = []

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def gap_runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    runs, start = [], None
    for stamp_, flag in mask.items():
        if flag and start is None:
            start = stamp_
        elif not flag and start is not None:
            runs.append((start, stamp_ - pd.Timedelta(days=1)))
            start = None
    if start is not None:
        runs.append((start, mask.index[-1]))
    return runs


# ------------------------------------------------------- F07 territory series
fig, ax = plt.subplots(figsize=(13, 5.0))
ax.plot(daily.index, daily.to_numpy(), color="0.65", lw=0.5, label="daily mean of 18 stations")
ax.plot(daily.index, daily.rolling(30, min_periods=20).mean(), color="#c44e52", lw=1.8,
        label="30-day rolling mean")
for start, end in gap_runs(~td.set_index("date")["in_file"]):
    ax.axvspan(start, end, color="0.85", alpha=0.7, zorder=0)
annual = daily.groupby(daily.index.year).mean()
ax.set_ylim(0.8, 5.1)
ax.set_ylabel("territory mean AQHI")
ax.set_title(f"F07 Territory hourly AQHI, 2021-01 to 2026-06: annual means {annual.loc[FULL_YEARS].min():.2f}-{annual.loc[FULL_YEARS].max():.2f}")
ax.legend(loc="lower right", frameon=False, ncol=2)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
stamp(ax, "grey bands = dates absent from the export\nannual means  " + "  ".join(f"{y}:{annual[y]:.2f}" for y in FULL_YEARS), "upper left")
save_fig(fig, "F07_territory_daily_series.png")
save_table(daily.rename("mean_station_mean").reset_index(), "t08_territory_daily_series.csv")

findings.append({
    "section": "temporal", "finding": "Annual territory mean is flat, not trending",
    "value": "; ".join(f"{y}: {annual[y]:.2f}" for y in FULL_YEARS),
    "unit": "mean AQHI", "figure": "F07",
    "reading": "Year-to-year spread is small relative to the seasonal swing visible in the same panel.",
})

# ---------------------------------------------------- F08 monthly lines by year
monthly = td.dropna(subset=["mean_station_mean"]).groupby(["year", "month"])["mean_station_mean"].agg(["mean", "count"]).reset_index()
piv = monthly.pivot(index="month", columns="year", values="mean").reindex(range(1, 13))
fig, ax = plt.subplots(figsize=(9.5, 5.0))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(FULL_YEARS) + 1))
for i, year in enumerate(FULL_YEARS + [2026]):
    if year not in piv.columns:
        continue
    ax.plot(piv.index, piv[year], marker="o", ms=3, lw=1.6, color=colors[i],
            label=f"{year}" + (" (H1 only)" if year == 2026 else ""))
ax.set_xticks(range(1, 13), MONTH_ABBR)
ax.set_xlabel("month")
ax.set_ylabel("mean territory AQHI")
ax.set_title(f"F08 Monthly mean AQHI by year: {piv.loc[1].min():.2f} in Jan to {piv.loc[7].max():.2f} in Jul across years")
ax.legend(frameon=False, ncol=3, fontsize=8.5)
stamp(ax, "seasonal shape repeats every year;\nlevel differences are small", "upper left")
save_fig(fig, "F08_monthly_mean_by_year.png")
save_table(piv.reset_index().rename(columns={"index": "month"}), "t09_monthly_mean_by_year.csv")

seasonal = piv[FULL_YEARS].mean(axis=1)
findings.append({
    "section": "temporal", "finding": "Seasonal cycle is the largest signal",
    "value": f"winter Dec-Feb {seasonal[[12,1,2]].mean():.2f} vs summer Jun-Aug {seasonal[[6,7,8]].mean():.2f}",
    "unit": "mean AQHI", "figure": "F08",
    "reading": "Summer is the clean season in this index; the annual mean hides a swing about twice the between-year spread.",
})

# --------------------------------------------------- F09 year x month heatmap
year_month = td.dropna(subset=["mean_station_mean"]).groupby(["year", "month"])["mean_station_mean"].mean().unstack("month")
fig, ax = plt.subplots(figsize=(10, 4.4))
im = ax.imshow(year_month.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=2.6, vmax=4.3)
ax.set_xticks(range(12), MONTH_ABBR)
ax.set_yticks(range(len(year_month.index)), year_month.index)
for i in range(year_month.shape[0]):
    for j in range(year_month.shape[1]):
        value = year_month.iloc[i, j]
        if pd.notna(value):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if value > 3.8 else "0.15")
ax.set_title("F09 Mean territory AQHI by year and month (2026 is Jan-Jun only)")
ax.set_xlabel("month")
fig.colorbar(im, ax=ax, label="mean AQHI")
save_fig(fig, "F09_year_month_heatmap.png")
save_table(year_month.reset_index(), "t10_year_month_mean.csv")

# --------------------------------------------------- F10 hour x month heatmap
hour_month = (h.dropna(subset=["aqhi"]).groupby(["month", "hour_of_day"], observed=True)["aqhi"]
              .mean().unstack("hour_of_day"))
fig, ax = plt.subplots(figsize=(12, 5.0))
im = ax.imshow(hour_month.to_numpy(), aspect="auto", cmap="YlOrRd",
               vmin=2.5, vmax=float(np.nanmax(hour_month.to_numpy())))
ax.set_yticks(range(12), MONTH_ABBR)
ax.set_xticks(range(0, 24, 2), [f"{x:02d}" for x in range(0, 24, 2)])
ax.set_xlabel("hour of day (hour beginning)")
ax.set_title(f"F10 Mean AQHI by month and hour: peak {hour_month.to_numpy().max():.2f} at {hour_month.stack().idxmax()[1]:02d}:00 in {MONTH_ABBR[hour_month.stack().idxmax()[0]-1]}")
fig.colorbar(im, ax=ax, label="mean AQHI")
save_fig(fig, "F10_hour_month_heatmap.png")
save_table(hour_month.reset_index(), "t11_hour_month_mean.csv")

# ----------------------------------------------------- F11 diurnal by season
diurnal = (h.dropna(subset=["aqhi"]).groupby(["season", "hour_of_day"], observed=True)["aqhi"]
           .mean().unstack("hour_of_day").reindex(SEASON_ORDER))
fig, ax = plt.subplots(figsize=(9.5, 4.8))
for i, season in enumerate(SEASON_ORDER):
    ax.plot(range(24), diurnal.loc[season].to_numpy(), marker="o", ms=3, lw=1.8,
            color=plt.cm.twilight_shifted(0.15 + 0.2 * i), label=season)
ax.set_xticks(range(0, 24, 2), [f"{x:02d}" for x in range(0, 24, 2)])
ax.set_xlabel("hour of day (hour beginning)")
ax.set_ylabel("mean AQHI")
ax.set_title(f"F11 Diurnal AQHI by season: night-to-afternoon rise is {diurnal.mean(axis=1).max() - diurnal.mean(axis=1).min():.2f} AQHI")
ax.legend(frameon=False, ncol=4, fontsize=8.5)
stamp(ax, f"winter peak hour {diurnal.loc['Winter'].idxmax():02d}:00\nsummer peak hour {diurnal.loc['Summer'].idxmax():02d}:00", "upper left")
save_fig(fig, "F11_diurnal_by_season.png")
save_table(diurnal.reset_index(), "t12_diurnal_by_season.csv")

# --------------------------------------------------------------- F12 weekday
dow = td.dropna(subset=["mean_station_mean"]).copy()
dow_profile = dow.groupby("dow")["mean_station_mean"].agg(["mean", "std", "count"])
dow_profile["ci"] = 1.96 * dow_profile["std"] / np.sqrt(dow_profile["count"])
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.bar(range(7), dow_profile["mean"], yerr=dow_profile["ci"], capsize=3,
       color=["#4c72b0"] * 5 + ["#c44e52"] * 2)
ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_ylim(dow_profile["mean"].min() - 0.3, dow_profile["mean"].max() + 0.3)
ax.set_ylabel("mean territory AQHI")
ax.set_title(f"F12 Weekday vs weekend: {dow_profile.loc[:4,'mean'].mean():.3f} weekdays vs {dow_profile.loc[5:,'mean'].mean():.3f} weekend")
stamp(ax, f"largest gap {dow_profile['mean'].max()-dow_profile['mean'].min():.3f} AQHI\nbars = 95% CI of daily means", "upper right")
save_fig(fig, "F12_day_of_week.png")
save_table(dow_profile.reset_index(), "t13_day_of_week.csv")

weekday_mean = dow_profile.loc[:4, "mean"].mean()
weekend_mean = dow_profile.loc[5:, "mean"].mean()
findings.append({
    "section": "temporal", "finding": "Weekday/weekend difference is negligible",
    "value": f"{weekday_mean:.3f} vs {weekend_mean:.3f} ({weekend_mean-weekday_mean:+.3f})",
    "unit": "mean AQHI", "figure": "F12",
    "reading": "Day-of-week is not a useful lever for the scenario; hour and season are.",
})

# ------------------------------------------------------ F13 annual 3-panel
full = td[td["year"].isin(FULL_YEARS)].dropna(subset=["max_station_max"])
annual_stats = full.groupby("year").agg(
    mean=("mean_station_mean", "mean"),
    days=("date", "nunique"),
    days_ge7=("max_station_max", lambda s: int((s >= 7).sum())),
    days_ge8=("max_station_max", lambda s: int((s >= 8).sum())),
    max=("max_station_max", "max"),
)
hours_by_year = h.dropna(subset=["aqhi"]).groupby("year").agg(
    share_ge7=("aqhi", lambda s: float((s >= 7).mean())),
    share_ge8=("aqhi", lambda s: float((s >= 8).mean())),
)
annual = annual_stats.join(hours_by_year)
annual["mean_ci"] = full.groupby("year")["mean_station_mean"].apply(lambda s: 1.96 * s.std() / np.sqrt(len(s)))
annual = annual.loc[FULL_YEARS]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
axes[0].errorbar(annual.index, annual["mean"], yerr=annual["mean_ci"], marker="o", color="#4c72b0", capsize=3)
axes[0].set_title("annual mean AQHI")
axes[0].set_xticks(FULL_YEARS)
axes[0].set_ylim(2.9, 3.75)
axes[1].bar(annual.index, annual["days_ge7"], color="#ff7f0e")
axes[1].set_title("days with any station >= 7")
axes[1].set_xticks(FULL_YEARS)
axes[2].bar(annual.index, annual["share_ge7"] * 100, color="#d62728")
axes[2].set_title("share of station-hours >= 7 (%)")
axes[2].set_xticks(FULL_YEARS)
fig.suptitle(f"F13 Annual summary 2021-2025: mean range {annual['mean'].min():.2f}-{annual['mean'].max():.2f}, high-hour share {annual['share_ge7'].min()*100:.1f}-{annual['share_ge7'].max()*100:.1f}%",
             fontweight="bold", y=1.02)
save_fig(fig, "F13_annual_summary.png")
save_table(annual.reset_index(), "t14_annual_summary.csv")

# ------------------------------------------------------- F14 band share bars
band_year = (h.dropna(subset=["aqhi"]).groupby(["year", "band"], observed=True).size()
             .unstack("band").reindex(columns=BAND_ORDER).fillna(0))
band_share = band_year.div(band_year.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
bottom = np.zeros(len(band_share))
for band_name in BAND_ORDER:
    values = band_share[band_name].to_numpy() * 100
    ax.bar(band_share.index.astype(str), values, bottom=bottom, color=BAND_COLOR[band_name],
           label=band_name, width=0.65)
    bottom += values
ax.set_ylabel("% of station-hours")
ax.set_title(f"F14 AQHI band mix by year: low band {band_share['Low (1-3)'].min()*100:.1f}-{band_share['Low (1-3)'].max()*100:.1f}% every year")
ax.legend(frameon=False, ncol=5, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.10))
ax.set_ylim(0, 100)
save_fig(fig, "F14_band_share_by_year.png")
save_table(band_share.reset_index(), "t15_band_share_by_year.csv")

# ---------------------------------------------------- F15 daily max boxplots
fig, ax = plt.subplots(figsize=(9.0, 4.6))
data = [full.loc[full["year"] == y, "max_station_max"].dropna().to_numpy() for y in FULL_YEARS]
parts = ax.boxplot(data, tick_labels=FULL_YEARS, showfliers=True, patch_artist=True,
                   flierprops=dict(marker=".", ms=3, alpha=0.5))
for patch in parts["boxes"]:
    patch.set_facecolor("#9ecae1")
for i, y in enumerate(FULL_YEARS):
    share = float((full.loc[full["year"] == y, "max_station_max"] >= 7).mean())
    ax.text(i + 1, 10.35, f"{share*100:.0f}%", ha="center", fontsize=8)
ax.set_ylim(0.5, 10.8)
ax.set_ylabel("max AQHI across 18 stations that day")
ax.set_title("F15 Daily territory maximum by year (label = % of days reaching >= 7)")
save_fig(fig, "F15_daily_max_by_year.png")
save_table(pd.DataFrame({"year": FULL_YEARS,
                         "p50": [np.median(x) for x in data],
                         "p90": [np.quantile(x, 0.9) for x in data],
                         "p99": [np.quantile(x, 0.99) for x in data],
                         "max": [x.max() for x in data]}), "t16_daily_max_quantiles.csv")

# ----------------------------------------------------------------- F16 ACF
hour_grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
hour_series = hourly.reindex(hour_grid)
day_grid = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
day_series = daily.reindex(day_grid)


def acf(series: pd.Series, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    values = series.to_numpy(dtype="float64")
    out, counts = [], []
    for lag in range(max_lag + 1):
        a = values[: len(values) - lag] if lag else values
        b = values[lag:] if lag else values
        mask = np.isfinite(a) & np.isfinite(b)
        out.append(float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 30 else np.nan)
        counts.append(int(mask.sum()))
    return np.asarray(out), np.asarray(counts)


acf_h, n_h = acf(hour_series, 96)
acf_d, n_d = acf(day_series, 60)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
axes[0].bar(range(97), acf_h, color="#4c72b0", width=1.0)
axes[0].axhline(0, color="0.3", lw=0.8)
axes[0].axhline(1.96 / np.sqrt(n_h[1]), color="0.5", ls="--", lw=0.7)
axes[0].set_xlabel("lag (hours)")
axes[0].set_ylabel("autocorrelation")
axes[0].set_title(f"hourly: lag1 {acf_h[1]:.3f}, lag24 {acf_h[24]:.3f}")
axes[1].bar(range(61), acf_d, color="#c44e52", width=1.0)
axes[1].axhline(0, color="0.3", lw=0.8)
axes[1].axhline(1.96 / np.sqrt(n_d[1]), color="0.5", ls="--", lw=0.7)
axes[1].set_xlabel("lag (days)")
axes[1].set_title(f"daily: lag1 {acf_d[1]:.3f}, lag7 {acf_d[7]:.3f}")
fig.suptitle("F16 Autocorrelation of the territory mean: strong hour-to-hour memory, weak week-to-week memory",
             fontweight="bold", y=1.02)
save_fig(fig, "F16_acf.png")
save_table(pd.DataFrame({"lag_hours": range(97), "acf_hourly": acf_h, "n_pairs_hourly": n_h}), "t17_acf_hourly.csv")
save_table(pd.DataFrame({"lag_days": range(61), "acf_daily": acf_d, "n_pairs_daily": n_d}), "t18_acf_daily.csv")

findings.append({
    "section": "temporal", "finding": "Persistence decays slowly at hourly scale",
    "value": f"hourly lag1 {acf_h[1]:.3f}, lag24 {acf_h[24]:.3f}; daily lag1 {acf_d[1]:.3f}, lag7 {acf_d[7]:.3f}",
    "unit": "autocorrelation", "figure": "F16",
    "reading": "Today is a good predictor of the next few hours; a week ahead is nearly independent at the daily level.",
})

# ------------------------------------------------------- F17 band transition
s = h.sort_values(["station", "timestamp"]).copy()
s["next_band"] = s.groupby("station", observed=True)["band"].shift(-1)
s["next_ts"] = s.groupby("station", observed=True)["timestamp"].shift(-1)
consecutive = (s["next_ts"] - s["timestamp"]).eq(pd.Timedelta(hours=1)) & s["band"].notna() & s["next_band"].notna()
trans_counts = pd.crosstab(s.loc[consecutive, "band"], s.loc[consecutive, "next_band"]).reindex(
    index=BAND_ORDER, columns=BAND_ORDER).fillna(0)
trans = trans_counts.div(trans_counts.sum(axis=1), axis=0)
high_rows = BAND_ORDER[2:]
stay_high = float(trans_counts.loc[high_rows, high_rows].to_numpy().sum() /
                  trans_counts.loc[high_rows].to_numpy().sum())
fig, ax = plt.subplots(figsize=(7.6, 5.4))
im = ax.imshow(trans.to_numpy() * 100, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(5), BAND_ORDER, rotation=25, ha="right")
ax.set_yticks(range(5), BAND_ORDER)
for i in range(5):
    for j in range(5):
        value = trans.iloc[i, j] * 100
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8.5,
                color="white" if value > 55 else "0.15")
ax.set_xlabel("next hour band")
ax.set_ylabel("current hour band")
ax.set_title(f"F17 Hour-to-hour band transition: P(stay >=7 | currently >=7) = {stay_high*100:.1f}%")
fig.colorbar(im, ax=ax, label="row %")
save_fig(fig, "F17_band_transition.png")
save_table(trans.reset_index(), "t19_band_transition.csv")

# -------------------------------------------------------- F18 regime break
monthly_ts = td.dropna(subset=["mean_station_mean"]).set_index("date")["mean_station_mean"].resample("MS").mean()
window = monthly_ts.loc["2024-01-01":]
fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.plot(window.index, window.to_numpy(), marker="o", ms=4, color="#333333", lw=1.5)
ax.axvline(METHOD_CHANGE, color="#d62728", ls="--", lw=1.4)
ax.text(METHOD_CHANGE + pd.Timedelta(days=12), 2.05, "AQHI compilation\nupdated 2025-03-22",
        color="#d62728", fontsize=8, va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d62728", alpha=0.9))
matched = (td[td["month"].between(4, 12) & td["year"].between(2021, 2025)]
           .dropna(subset=["mean_station_mean"])
           .groupby(["year", "month"])["mean_station_mean"].mean().unstack("month"))
diff_by_month = matched.loc[2025] - matched.loc[2021:2024].mean()
diff = float(diff_by_month.mean())
ci = float(1.96 * diff_by_month.std(ddof=1) / np.sqrt(len(diff_by_month)))
ax.set_ylabel("monthly mean AQHI")
ax.set_title(f"F18 Step at the 2025-03 AQHI method change: Apr-Dec 2025 is {diff:+.3f} AQHI vs Apr-Dec 2021-2024 (95% CI {diff-ci:+.3f} to {diff+ci:+.3f})")
save_fig(fig, "F18_method_change.png")
save_table(pd.DataFrame({"month": diff_by_month.index, "diff_2025_minus_2021_2024": diff_by_month.to_numpy()}),
           "t20_method_change_by_month.csv")

findings.append({
    "section": "temporal", "finding": "A construction change sits inside the series",
    "value": f"Apr-Dec 2025 vs Apr-Dec 2021-2024: {diff:+.3f} AQHI (95% CI {diff-ci:+.3f} to {diff+ci:+.3f})",
    "unit": "mean AQHI", "figure": "F18",
    "reading": "EPD updated the ozone risk factor on 2025-03-22. Any 2025-2026 level must be compared with that caveat, not read as an air-quality trend.",
})

# -------------------------------------------------- F19 season x year heatmap
season_year = (td.dropna(subset=["mean_station_mean"]).groupby(["season", "year"])["mean_station_mean"]
               .mean().unstack("year").reindex(SEASON_ORDER))
fig, ax = plt.subplots(figsize=(8.4, 4.4))
im = ax.imshow(season_year.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=2.4, vmax=4.4)
ax.set_yticks(range(4), SEASON_ORDER)
ax.set_xticks(range(len(season_year.columns)), season_year.columns)
for i in range(season_year.shape[0]):
    for j in range(season_year.shape[1]):
        value = season_year.iloc[i, j]
        if pd.notna(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 3.9 else "0.15")
ax.set_title("F19 Mean AQHI by season and year")
fig.colorbar(im, ax=ax, label="mean AQHI")
save_fig(fig, "F19_season_year.png")
save_table(season_year.reset_index(), "t21_season_year.csv")

# --------------------------------------------- F20 share >=7 by year x month
exceed = (h.dropna(subset=["aqhi"]).assign(ge7=lambda d: d["aqhi"] >= 7)
          .groupby(["year", "month"], observed=True)["ge7"].mean().unstack("month"))
fig, ax = plt.subplots(figsize=(10, 4.4))
im = ax.imshow(exceed.to_numpy() * 100, aspect="auto", cmap="Reds", vmin=0,
               vmax=max(1.0, float(np.nanmax(exceed.to_numpy())) * 100))
ax.set_xticks(range(12), MONTH_ABBR)
ax.set_yticks(range(len(exceed.index)), exceed.index)
for i in range(exceed.shape[0]):
    for j in range(exceed.shape[1]):
        value = exceed.iloc[i, j] * 100
        if pd.notna(value):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if value > 0.6 * np.nanmax(exceed.to_numpy()) * 100 else "0.15")
ax.set_title("F20 Share of station-hours at AQHI >= 7 (%), by year and month")
fig.colorbar(im, ax=ax, label="% of hours")
save_fig(fig, "F20_exceedance_year_month.png")
save_table(exceed.reset_index(), "t22_exceedance_year_month.csv")

# ------------------------------------------------------------ trend checks
annual_mean = annual["mean"]
slope5, intercept5, r5, p5, se5 = stats.linregress(annual_mean.index.astype(float), annual_mean.to_numpy())
pre = annual_mean.loc[2021:2024]
slope4, intercept4, r4, p4, se4 = stats.linregress(pre.index.astype(float), pre.to_numpy())
theil = stats.theilslopes(annual_mean.to_numpy(), annual_mean.index.astype(float))
save_table(pd.DataFrame([
    {"window": "2021-2025", "slope_per_year": slope5, "ci_low": slope5 - 2.776 * se5,
     "ci_high": slope5 + 2.776 * se5, "p_value": p5, "n_years": 5},
    {"window": "2021-2024", "slope_per_year": slope4, "ci_low": slope4 - 2.776 * se4,
     "ci_high": slope4 + 2.776 * se4, "p_value": p4, "n_years": 4},
    {"window": "2021-2025 theil-sen", "slope_per_year": theil[0], "ci_low": theil[2],
     "ci_high": theil[3], "p_value": np.nan, "n_years": 5},
]), "t23_trend_tests.csv")
findings.append({
    "section": "temporal", "finding": "No defensible monotonic trend in five annual means",
    "value": f"OLS 2021-2025 {slope5:+.3f}/yr (p={p5:.2f}); 2021-2024 {slope4:+.3f}/yr (p={p4:.2f})",
    "unit": "AQHI per year", "figure": "F13",
    "reading": "With five points and a mid-series method change, the trend estimate is not separable from year-to-year weather variability.",
})

write_findings(findings, "findings_02.csv")
print("02 temporal done")
print(f"  annual means: {annual['mean'].round(3).to_dict()}")
print(f"  seasonal winter {seasonal[[12,1,2]].mean():.3f} summer {seasonal[[6,7,8]].mean():.3f}")
print(f"  ACF hourly lag1 {acf_h[1]:.3f} lag24 {acf_h[24]:.3f}; daily lag1 {acf_d[1]:.3f} lag7 {acf_d[7]:.3f}")
print(f"  method-change Apr-Dec diff {diff:+.4f}  CI {diff-ci:+.4f}..{diff+ci:+.4f}")
print(f"  trend 2021-2025 {slope5:+.4f}/yr p={p5:.3f}; 2021-2024 {slope4:+.4f}/yr p={p4:.3f}")
print(f"  hour-month peak {hour_month.stack().idxmax()}")
