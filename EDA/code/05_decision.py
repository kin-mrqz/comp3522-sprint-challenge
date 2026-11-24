"""05 - decision layer for the scenario: when is outside safe, descriptively.

Figures F41-F46, tables t45-t50.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_lib import (BAND_COLOR, BAND_ORDER, FULL_YEARS, PROC, SEASON, SEASON_ORDER,
                     save_fig, save_table, stamp, write_findings)

h = pd.read_parquet(PROC / "aqhi_hourly.parquet")
td = pd.read_parquet(PROC / "aqhi_daily_territory.parquet")
th = pd.read_parquet(PROC / "aqhi_territory_hourly.parquet")
h["aqhi"] = h["aqhi"].astype("float64")
td["date"] = pd.to_datetime(td["date"])
td["max_station_max"] = td["max_station_max"].astype("float64")
th["timestamp"] = pd.to_datetime(th["timestamp"])
th["mean_station_mean"] = th["mean_station_mean"].astype("float64")
findings = []
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

H = h.dropna(subset=["aqhi"]).copy()
H["ge7"] = H["aqhi"] >= 7
H["ge8"] = H["aqhi"] >= 8

# ------------------------------------------- F41 risk by month and hour
risk = H.groupby(["month", "hour_of_day"], observed=True)["ge7"].agg(["mean", "size"]).reset_index()
risk_piv = risk.pivot(index="month", columns="hour_of_day", values="mean").reindex(range(1, 13))
fig, ax = plt.subplots(figsize=(12.5, 5.2))
im = ax.imshow(risk_piv.to_numpy() * 100, aspect="auto", cmap="Reds", vmin=0,
               vmax=float(np.nanmax(risk_piv.to_numpy())) * 100)
ax.set_yticks(range(12), MONTH_ABBR)
ax.set_xticks(range(0, 24, 2), [f"{x:02d}" for x in range(0, 24, 2)])
ax.set_xlabel("hour of day (hour beginning)")
daytime_risk = risk_piv.loc[:, 6:19]
ax.set_title(f"F41 P(AQHI >= 7) by month and hour: safest daytime cell {daytime_risk.stack().idxmin()[1]:02d}:00 in {MONTH_ABBR[daytime_risk.stack().idxmin()[0]-1]}, worst {daytime_risk.stack().idxmax()[1]:02d}:00 in {MONTH_ABBR[daytime_risk.stack().idxmax()[0]-1]}")
fig.colorbar(im, ax=ax, label="% of station-hours >= 7")
save_fig(fig, "F41_risk_month_hour.png")
save_table(risk_piv.reset_index(), "t45_prob_ge7_month_hour.csv")

flat = risk_piv.stack().rename("p_ge7").reset_index()
flat.columns = ["month", "hour", "p_ge7"]
flat["month_name"] = flat["month"].map(lambda m: MONTH_ABBR[m - 1])
flat = flat[flat["hour"].between(6, 19)].sort_values("p_ge7")
save_table(pd.concat([flat.head(8).assign(rank_type="safest"),
                      flat.tail(8).sort_values("p_ge7", ascending=False).assign(rank_type="worst")]),
           "t46_safest_worst_month_hour.csv")

findings.append({
    "section": "decision", "finding": "The safest daytime cells are early morning, and the spread across cells is large",
    "value": f"min {flat['p_ge7'].min()*100:.2f}% ({flat.iloc[0]['month_name']} {int(flat.iloc[0]['hour']):02d}:00); max {flat['p_ge7'].max()*100:.1f}% ({flat.iloc[-1]['month_name']} {int(flat.iloc[-1]['hour']):02d}:00)",
    "unit": "% of hours >= 7", "figure": "F41",
    "reading": "Timing and month matter more than the annual average; the worst cell is more than 20 times the best.",
})

# --------------------------------------------- F42 risk by season and hour
season_risk = H.groupby(["season", "hour_of_day"], observed=True)["ge7"].mean().unstack("hour_of_day").reindex(SEASON_ORDER)
season_risk8 = H.groupby(["season", "hour_of_day"], observed=True)["ge8"].mean().unstack("hour_of_day").reindex(SEASON_ORDER)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
for i, season in enumerate(SEASON_ORDER):
    color = plt.cm.twilight_shifted(0.15 + 0.2 * i)
    axes[0].plot(range(24), season_risk.loc[season] * 100, marker="o", ms=3, lw=1.8, color=color, label=season)
    axes[1].plot(range(24), season_risk8.loc[season] * 100, marker="o", ms=3, lw=1.8, color=color, label=season)
axes[0].set_title("P(AQHI >= 7), %")
axes[1].set_title("P(AQHI >= 8), %")
for ax in axes:
    ax.set_xticks(range(0, 24, 3), [f"{x:02d}" for x in range(0, 24, 3)])
    ax.set_xlabel("hour of day")
    ax.legend(frameon=False, fontsize=8)
fig.suptitle(f"F42 Season-hour risk: summer is lowest at every hour; winter peak {season_risk.loc['Winter'].max()*100:.1f}% at {season_risk.loc['Winter'].idxmax():02d}:00",
             fontweight="bold", y=1.02)
save_fig(fig, "F42_season_hour_risk.png")
save_table(season_risk.reset_index(), "t47_prob_ge7_season_hour.csv")

# Month-level risk is not stable year to year: check the two months that look worst.
month_year_risk = H.groupby(["month", "year"], observed=True)["ge7"].mean().unstack("year")
apr = month_year_risk.loc[4]
sep = month_year_risk.loc[9]
findings.append({
    "section": "decision", "finding": "Month-level risk is driven by a few years, not a stable monthly law",
    "value": (f"April: {apr.min()*100:.1f}% in {int(apr.idxmin())} vs {apr.max()*100:.1f}% in {int(apr.idxmax())}; "
              f"September: {sep.min()*100:.1f}% in {int(sep.idxmin())} vs {sep.max()*100:.1f}% in {int(sep.idxmax())}"),
    "unit": "% of hours >= 7", "figure": "F42",
    "reading": "April 2025 and September 2022 create the month-level tail; do not promise a quiet September on the strength of the pooled average.",
})

# ---------------------------------------------- F43 safe days by month
full = td[td["year"].isin(FULL_YEARS)].copy()
full["band"] = pd.cut(full["max_station_max"].clip(lower=1, upper=10),
                      bins=[0, 3, 6, 7, 9, 10], labels=BAND_ORDER, include_lowest=True)
counts = pd.crosstab(full["month"], full["band"]).reindex(columns=BAND_ORDER).fillna(0)
shares = counts.div(counts.sum(axis=1), axis=0)
high_share = counts[["High (7)", "Very High (8-10)", "Serious (10+)"]].sum(axis=1) / counts.sum(axis=1)
fig, ax = plt.subplots(figsize=(10, 4.8))
bottom = np.zeros(12)
for band_name in BAND_ORDER:
    values = shares[band_name].to_numpy() * 100
    ax.bar(range(1, 13), values, bottom=bottom, color=BAND_COLOR[band_name], label=band_name)
    bottom += values
for month in range(1, 13):
    ax.text(month, 101.5, f"{high_share.loc[month]*100:.0f}%", ha="center", fontsize=8, color="0.2")
ax.set_xticks(range(1, 13), MONTH_ABBR)
ax.set_ylim(0, 108)
ax.set_ylabel("% of days in the month")
ax.set_title(f"F43 Daily maximum band by month (2021-2025), label = % of days reaching >= 7: {high_share.min()*100:.0f}% in {MONTH_ABBR[int(high_share.idxmin())-1]} to {high_share.max()*100:.0f}% in {MONTH_ABBR[int(high_share.idxmax())-1]}")
ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.22))
save_fig(fig, "F43_safe_days_by_month.png")
save_table(counts.reset_index(), "t48_safe_days_by_month_counts.csv")
save_table(shares.reset_index(), "t49_safe_days_by_month_shares.csv")

findings.append({
    "section": "decision", "finding": "Summer has the fewest high days, spring and autumn the most",
    "value": f"June {high_share.loc[6]*100:.0f}%, August {high_share.loc[8]*100:.0f}%, January {high_share.loc[1]*100:.0f}%, April {high_share.loc[4]*100:.0f}%, September {high_share.loc[9]*100:.0f}%",
    "unit": "% of days reaching >= 7", "figure": "F43",
    "reading": "June and August are the safest months; January, April and September carry the highest share of high days.",
})

# ------------------------------------------ F44 when the daily peak happens
peak_idx = th.groupby(th["timestamp"].dt.normalize())["mean_station_mean"].idxmax()
peaks = th.loc[peak_idx, ["timestamp", "mean_station_mean"]].copy()
peaks["date"] = peaks["timestamp"].dt.normalize()
peaks["hour"] = peaks["timestamp"].dt.hour
peaks["month"] = peaks["timestamp"].dt.month
peaks["season"] = peaks["month"].map(SEASON)
peak_dist = pd.crosstab(peaks["season"], peaks["hour"]).reindex(SEASON_ORDER).reindex(columns=range(24)).fillna(0)
peak_share = peak_dist.div(peak_dist.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
im = ax.imshow(peak_share.to_numpy() * 100, aspect="auto", cmap="Blues", vmin=0, vmax=30)
ax.set_yticks(range(4), SEASON_ORDER)
ax.set_xticks(range(0, 24, 2), [f"{x:02d}" for x in range(0, 24, 2)])
ax.set_xlabel("hour of the territory-mean daily peak")
ax.set_title(f"F44 When the day peaks: median hour {int(peaks['hour'].median()):02d}:00, {peaks['hour'].between(15, 17).mean()*100:.0f}% between 15:00 and 17:59")
fig.colorbar(im, ax=ax, label="% of days")
save_fig(fig, "F44_peak_hour.png")
save_table(peak_share.reset_index(), "t50_peak_hour_by_season.csv")

# ------------------------------------------- F45 best 6-hour window
H["window_start"] = H["hour_of_day"].where(H["hour_of_day"] <= 18)
window_rows = []
for month, block in H.groupby("month", observed=True):
    for start in range(6, 13):
        hours = [(start + k) % 24 for k in range(6)]
        cell = block[block["hour_of_day"].isin(hours)]
        per_day = cell.groupby(["date", "station"], observed=True)["ge7"].max()
        window_rows.append({"month": int(month), "start_hour": start,
                            "mean_aqhi": float(cell["aqhi"].mean()),
                            "p_ge7": float(per_day.mean()),
                            "n_station_days": int(len(per_day))})
windows = pd.DataFrame(window_rows)
mean_grid = windows.pivot(index="month", columns="start_hour", values="mean_aqhi")
risk_grid = windows.pivot(index="month", columns="start_hour", values="p_ge7")
best = windows.loc[windows.groupby("month")["p_ge7"].idxmin()].sort_values("month")
best["window"] = best["start_hour"].map(lambda h_: f"{h_:02d}:00-{(h_+5)%24:02d}:59")
save_table(best, "t51_best_six_hour_window.csv")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
im0 = axes[0].imshow(mean_grid.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=2.2, vmax=4.4)
axes[0].set_title("mean AQHI in the 6-hour window")
im1 = axes[1].imshow(risk_grid.to_numpy() * 100, aspect="auto", cmap="Reds", vmin=0,
                     vmax=float(risk_grid.to_numpy().max()) * 100)
axes[1].set_title("share of station-days with any hour >= 7, %")
for ax in axes:
    ax.set_yticks(range(12), MONTH_ABBR)
    ax.set_xticks(range(7), [f"{x:02d}" for x in range(6, 13)])
    ax.set_xlabel("window start hour (6-hour window)")
    ax.scatter(best["start_hour"] - 6, best["month"] - 1, marker="*", s=130, color="#1f77b4",
               edgecolor="white", zorder=5)
fig.colorbar(im0, ax=axes[0], label="mean AQHI")
fig.colorbar(im1, ax=axes[1], label="% of station-days")
delay_ratio = float(risk_grid.mean(axis=0).loc[12] / max(risk_grid.mean(axis=0).loc[6], 1e-9))
fig.suptitle(f"F45 Every month's lowest-risk 6-hour window starts at 06:00; moving the start to 12:00 raises the risk {delay_ratio:.1f}x",
             fontweight="bold", y=1.02)
save_fig(fig, "F45_best_window.png")
save_table(windows, "t52_window_grid.csv")

findings.append({
    "section": "decision", "finding": "The best daytime six-hour window is early morning in every month",
    "value": "; ".join(f"{MONTH_ABBR[int(r.month)-1]}: {r.window} (P>=7 {r.p_ge7*100:.1f}%)" for r in best.head(4).itertuples()),
    "unit": "window", "figure": "F45",
    "reading": "06:00-11:59 is the lowest-risk block; the afternoon build-up is the thing to avoid.",
})

# --------------------------------------- F46 today band -> tomorrow band
dd = td[td["in_file"]].copy()
dd["today_band"] = pd.cut(dd["max_station_max"].clip(lower=1, upper=10), bins=[0, 3, 6, 7, 9, 10],
                          labels=BAND_ORDER, include_lowest=True)
dd["tomorrow_band"] = dd["today_band"].shift(-1)
dd["next_date"] = dd["date"].shift(-1)
dd = dd[dd["next_date"].sub(dd["date"]).dt.days.eq(1) & dd["tomorrow_band"].notna()]
day_trans = pd.crosstab(dd["today_band"], dd["tomorrow_band"], normalize="index").reindex(
    index=BAND_ORDER, columns=BAND_ORDER).fillna(0)
fig, ax = plt.subplots(figsize=(7.6, 5.4))
im = ax.imshow(day_trans.to_numpy() * 100, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(5), BAND_ORDER, rotation=25, ha="right")
ax.set_yticks(range(5), BAND_ORDER)
for i in range(5):
    for j in range(5):
        value = day_trans.iloc[i, j] * 100
        ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=9,
                color="white" if value > 55 else "0.15")
ax.set_xlabel("tomorrow's daily maximum band")
ax.set_ylabel("today's daily maximum band")
ax.set_title(f"F46 Day-to-day band transition: P(tomorrow low | today low) = {day_trans.loc['Low (1-3)','Low (1-3)']*100:.0f}%")
fig.colorbar(im, ax=ax, label="row %")
save_fig(fig, "F46_day_band_transition.png")
save_table(day_trans.reset_index(), "t53_day_band_transition.csv")

# ------------------------------------------------ practical summary table
summary_rows = []
for month in range(1, 13):
    block = H[(H["month"] == month) & (H["year"].isin(FULL_YEARS))]
    safe = best[best["month"] == month].iloc[0]
    summary_rows.append({
        "month": MONTH_ABBR[month - 1],
        "mean_aqhi": float(block["aqhi"].mean()),
        "p_ge7": float(block["ge7"].mean()),
        "best_window": safe["window"],
        "best_window_mean": float(safe["mean_aqhi"]),
        "best_window_risk_station_days": float(safe["p_ge7"]),
        "share_days_max_le6": float(shares.loc[month, ["Low (1-3)", "Moderate (4-6)"]].sum()),
    })
practical = pd.DataFrame(summary_rows)
save_table(practical, "t54_practical_month_guide.csv")

write_findings(findings, "findings_05.csv")
print("05 decision done")
print(practical.to_string(index=False))
print(f"  safest cell: {flat.iloc[0]['month_name']} {int(flat.iloc[0]['hour']):02d}:00 p={flat.iloc[0]['p_ge7']*100:.3f}%")
print(f"  worst cell:  {flat.iloc[-1]['month_name']} {int(flat.iloc[-1]['hour']):02d}:00 p={flat.iloc[-1]['p_ge7']*100:.3f}%")
print(f"  day persistence low->low {day_trans.loc['Low (1-3)','Low (1-3)']:.3f}")
