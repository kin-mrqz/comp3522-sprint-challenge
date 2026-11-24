"""03 - pollution episodes and persistence.

Figures F21-F27, tables t24-t28.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_lib import (FULL_YEARS, PROC, SEASON, SEASON_ORDER, save_fig, save_table, stamp,
                     write_findings)

td = pd.read_parquet(PROC / "aqhi_daily_territory.parquet")
th = pd.read_parquet(PROC / "aqhi_territory_hourly.parquet")
td["date"] = pd.to_datetime(td["date"])
th["timestamp"] = pd.to_datetime(th["timestamp"])
td["max_station_max"] = td["max_station_max"].astype("float64")
td["mean_station_mean"] = td["mean_station_mean"].astype("float64")
th["max_any_station"] = th["max_any_station"].astype("float64")
th["mean_station_mean"] = th["mean_station_mean"].astype("float64")
findings = []

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------- daily episodes
d = td[["date", "max_station_max", "n_stations_ge7", "season", "year", "month", "in_file"]].copy()
d["high"] = d["max_station_max"].ge(7)
d["prev_high"] = d["high"].shift(1, fill_value=False)
d["gap_days"] = d["date"].diff().dt.days
d["start"] = d["high"] & (~d["prev_high"] | d["gap_days"].ne(1))
d["episode"] = d["start"].cumsum().where(d["high"])

rows = []
for episode_id, block in d.dropna(subset=["episode"]).groupby("episode", observed=True):
    peak_row = block.loc[block["max_station_max"].idxmax()]
    rows.append({
        "episode_id": int(episode_id),
        "start": block["date"].min(),
        "end": block["date"].max(),
        "days": int(len(block)),
        "year": int(block["date"].min().year),
        "month": int(block["date"].min().month),
        "season": SEASON[int(block["date"].min().month)],
        "peak_max": float(block["max_station_max"].max()),
        "peak_date": peak_row["date"],
        "mean_daily_max": float(block["max_station_max"].mean()),
        "max_stations_ge7": int(block["n_stations_ge7"].max()),
        "days_all_stations_ge7": int((block["n_stations_ge7"] >= 18).sum()),
    })
episodes = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
save_table(episodes, "t24_episodes_daily.csv")

# yearly episode summary
summary = []
for year in FULL_YEARS + [2026]:
    block = d[d["year"] == year]
    eps = episodes[episodes["year"] == year]
    summary.append({
        "year": year,
        "days_in_file": int(block["in_file"].sum()),
        "high_days_any_station": int(block["high"].sum()),
        "share_days_high": float(block["high"].sum() / block["in_file"].sum()),
        "high_days_all_stations": int((block["n_stations_ge7"] >= 18).sum()),
        "high_days_majority": int((block["n_stations_ge7"] >= 9).sum()),
        "n_episodes": int(len(eps)),
        "longest_episode_days": int(eps["days"].max()) if len(eps) else 0,
        "mean_episode_days": float(eps["days"].mean()) if len(eps) else np.nan,
        "max_daily_max": float(block["max_station_max"].max()),
    })
summary = pd.DataFrame(summary)
save_table(summary, "t25_episode_summary_year.csv")

# ------------------------------------------------- F21 high days year x month
monthly_counts = (d.groupby(["year", "month"], observed=True)
                  .agg(high=("high", "sum"), n=("in_file", "sum")).reset_index())
monthly_counts["rate"] = monthly_counts["high"] / monthly_counts["n"]
high_month = monthly_counts.pivot(index="year", columns="month", values="rate").reindex(columns=range(1, 13))
high_count = monthly_counts.pivot(index="year", columns="month", values="high").reindex(columns=range(1, 13)).fillna(0).astype(int)
fig, ax = plt.subplots(figsize=(10, 4.4))
im = ax.imshow(high_month.to_numpy() * 100, aspect="auto", cmap="Oranges", vmin=0,
               vmax=max(5.0, float(np.nanmax(high_month.to_numpy())) * 100))
ax.set_xticks(range(12), MONTH_ABBR)
ax.set_yticks(range(len(high_month.index)), high_month.index)
for i in range(high_month.shape[0]):
    for j in range(high_month.shape[1]):
        value = high_month.iloc[i, j]
        if pd.notna(value):
            ax.text(j, i, f"{value*100:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if value > 0.6 * np.nanmax(high_month.to_numpy()) else "0.15")
ax.set_title(f"F21 Share of days with AQHI >= 7 somewhere (%): {int(d['high'].sum())} of {int(d['in_file'].sum())} days ({d['high'].sum()/d['in_file'].sum()*100:.1f}%)")
fig.colorbar(im, ax=ax, label="% of days in the month")
save_fig(fig, "F21_high_days_year_month.png")
save_table(high_count.reset_index(), "t26_high_days_year_month.csv")
save_table(high_month.reset_index(), "t26b_high_day_rate_year_month.csv")

season_high = d[d["high"]].groupby("season", observed=True).size().reindex(SEASON_ORDER).fillna(0).astype(int)
season_share = season_high / season_high.sum()
findings.append({
    "section": "episodes", "finding": "High days cluster in spring and autumn, not winter alone",
    "value": ", ".join(f"{s} {season_share[s]*100:.1f}%" for s in SEASON_ORDER) + f" of {int(season_high.sum())} high days",
    "unit": "share of high days", "figure": "F21",
    "reading": "Summer is clearly the clean season, but the worst months include April and September, so 'winter is bad' is the wrong rule.",
})

# ------------------------------------------------------ F22 episode duration
fig, ax = plt.subplots(figsize=(8.6, 4.4))
bins = np.arange(0.5, episodes["days"].max() + 1.5, 1)
ax.hist(episodes["days"], bins=bins, color="#ff7f0e", edgecolor="white")
median_days = float(episodes["days"].median())
share_single = float((episodes["days"] == 1).mean())
ax.set_xlabel("episode length (consecutive days with max >= 7)")
ax.set_ylabel("episodes")
ax.set_title(f"F22 {len(episodes)} daily episodes: median {median_days:.0f} day, {share_single*100:.0f}% last one day, longest {int(episodes['days'].max())} days")
stamp(ax, f"mean {episodes['days'].mean():.1f} days\npeak AQHI up to {episodes['peak_max'].max():.0f}", "upper right")
save_fig(fig, "F22_episode_duration.png")

other_high = summary.loc[(summary["year"].isin(FULL_YEARS)) & (summary["year"] != 2023), "high_days_any_station"]
findings.append({
    "section": "episodes", "finding": "One spring 2023 episode dominates the record",
    "value": f"{episodes.sort_values('days', ascending=False).iloc[0]['start'].date()} to {episodes.sort_values('days', ascending=False).iloc[0]['end'].date()}, {int(episodes['days'].max())} days, peak {episodes['peak_max'].max():.0f}; 2023 had {int(summary.loc[summary['year']==2023,'high_days_any_station'].iloc[0])} high days vs {int(other_high.min())}-{int(other_high.max())} in the other full years",
    "unit": "days", "figure": "F24",
    "reading": "The 2023 spring episode is the single largest pattern in the record; a five-year average hides it.",
})

findings.append({
    "section": "episodes", "finding": "Most episodes are short",
    "value": f"{share_single*100:.0f}% one day, median {median_days:.0f}, longest {int(episodes['days'].max())}",
    "unit": "days", "figure": "F22",
    "reading": "A one-day-ahead warning covers most episodes; multi-day episodes are rare but carry the highest peaks.",
})

# --------------------------------------------------- F23 peak vs duration
fig, ax = plt.subplots(figsize=(7.8, 5.0))
scatter = ax.scatter(episodes["days"], episodes["peak_max"], s=18 + 6 * episodes["max_stations_ge7"],
                     c=episodes["year"], cmap="viridis", alpha=0.85, edgecolor="white", lw=0.4)
ax.set_xlabel("episode length (days)")
ax.set_ylabel("peak daily maximum AQHI")
ax.set_title(f"F23 Peak vs length: correlation {episodes['days'].corr(episodes['peak_max']):.2f}; biggest peak {episodes['peak_max'].max():.0f}")
fig.colorbar(scatter, ax=ax, label="year")
stamp(ax, "dot size = max stations >= 7 in the episode", "lower right")
save_fig(fig, "F23_peak_vs_duration.png")
save_table(episodes[["episode_id", "start", "end", "days", "peak_max", "max_stations_ge7"]], "t27_peak_vs_duration.csv")

# ------------------------------------------------- F24 top episodes hourly
top = episodes.sort_values(["peak_max", "days"], ascending=False).head(5).sort_values("start")
fig, axes = plt.subplots(len(top), 1, figsize=(11, 2.1 * len(top)), sharex=False)
if len(top) == 1:
    axes = [axes]
hourly = th.set_index("timestamp")
for ax, row in zip(axes, top.itertuples()):
    lo = row.start - pd.Timedelta(days=1)
    hi = row.end + pd.Timedelta(days=1)
    block = hourly.loc[lo:hi]
    ax.plot(block.index, block["max_any_station"], color="#c44e52", lw=1.2, label="max across stations")
    ax.plot(block.index, block["mean_station_mean"], color="#4c72b0", lw=1.0, label="station mean")
    ax.axhline(7, color="0.4", ls="--", lw=0.8)
    ax.axvspan(row.start, row.end + pd.Timedelta(hours=23), color="#ff7f0e", alpha=0.12)
    ax.set_ylim(0.5, 10.8)
    ax.set_ylabel("AQHI")
    ax.set_title(f"{row.start.date()} to {row.end.date()}: {row.days} d, peak {row.peak_max:.0f}, up to {row.max_stations_ge7}/18 stations >=7", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.suptitle("F24 Five highest-peaking episodes, hourly (one day of context on each side)", fontweight="bold", y=1.0)
fig.tight_layout()
save_fig(fig, "F24_top_episodes.png")

# ----------------------------------------------------- F25 persistence
d2 = d[d["in_file"]].copy()
d2["next_high"] = d2["high"].shift(-1)
d2["next_date"] = d2["date"].shift(-1)
consecutive = d2["next_date"].sub(d2["date"]).dt.days.eq(1) & d2["next_high"].notna()
d2 = d2[consecutive]
base = float(d2["high"].mean())
p_next_given_high = float(d2.loc[d2["high"], "next_high"].mean())
p_next_given_low = float(d2.loc[~d2["high"], "next_high"].mean())
d2["today_band"] = pd.cut(d2["max_station_max"].clip(lower=1, upper=10),
                          bins=[0, 3, 6, 7, 9, 10],
                          labels=["1-3", "4-6", "7", "8-9", "10+"])
by_band = d2.groupby("today_band", observed=True)["next_high"].agg(["mean", "count"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
axes[0].bar(["base rate\n(tomorrow high)", "today high", "today not high"],
            [base * 100, p_next_given_high * 100, p_next_given_low * 100],
            color=["#7f7f7f", "#d62728", "#2ca02c"])
for i, value in enumerate([base, p_next_given_high, p_next_given_low]):
    axes[0].text(i, value * 100 + 1, f"{value*100:.1f}%", ha="center", fontsize=9)
axes[0].set_ylabel("P(tomorrow is a high day), %")
axes[0].set_title("one-day persistence")
axes[0].set_ylim(0, max(p_next_given_high, base) * 130 + 2)
axes[1].bar(range(len(by_band)), by_band["mean"] * 100, color="#d62728")
axes[1].set_xticks(range(len(by_band)), by_band.index.astype(str))
axes[1].set_xlabel("today's daily maximum AQHI")
axes[1].set_ylabel("P(tomorrow >= 7), %")
axes[1].set_title("escalation by today's peak")
for i, (value, count) in enumerate(zip(by_band["mean"], by_band["count"])):
    axes[1].text(i, value * 100 + 0.8, f"{value*100:.0f}%\nn={int(count)}", ha="center", fontsize=8)
fig.suptitle(f"F25 Persistence of high days: {p_next_given_high*100:.1f}% after a high day vs {base*100:.1f}% base rate", fontweight="bold", y=1.02)
save_fig(fig, "F25_daily_persistence.png")
save_table(pd.DataFrame([
    {"condition": "base_rate", "p_tomorrow_high": base, "n": len(d2)},
    {"condition": "today_high", "p_tomorrow_high": p_next_given_high, "n": int(d2["high"].sum())},
    {"condition": "today_not_high", "p_tomorrow_high": p_next_given_low, "n": int((~d2["high"]).sum())},
]), "t28_daily_persistence.csv")
save_table(by_band.reset_index(), "t29_persistence_by_today_band.csv")

findings.append({
    "section": "episodes", "finding": "A high day nearly triples tomorrow's odds",
    "value": f"{p_next_given_high*100:.1f}% vs base {base*100:.1f}% (today not high {p_next_given_low*100:.1f}%)",
    "unit": "P(tomorrow high)", "figure": "F25",
    "reading": "Persistence is real but not decisive; the day after a high day is the only day with materially elevated risk.",
})

# ------------------------------------------------ F26 stations >=7 per day
fig, ax = plt.subplots(figsize=(9.0, 4.4))
vc = d.loc[d["high"], "n_stations_ge7"].value_counts().reindex(range(1, 19)).fillna(0).astype(int)
ax.bar(vc.index, vc.to_numpy(), color="#ff7f0e")
ax.set_xticks(range(1, 19))
ax.set_xlabel("stations reporting AQHI >= 7 on the day (of 18)")
ax.set_ylabel("days")
ax.set_title(f"F26 High days are usually local: {vc.loc[1:3].sum()/vc.sum()*100:.0f}% touch at most 3 stations, {vc.loc[18]/vc.sum()*100:.1f}% reach all 18")
save_fig(fig, "F26_stations_ge7_per_day.png")
save_table(vc.rename_axis("n_stations_ge7").reset_index(name="days"), "t30_stations_ge7_per_day.csv")

# --------------------------------------------------- hourly episodes
hh = th.sort_values("timestamp").copy()
hh["high"] = hh["max_any_station"].ge(7)
hh["gap_h"] = hh["timestamp"].diff().dt.total_seconds().div(3600)
hh["start"] = hh["high"] & (~hh["high"].shift(1, fill_value=False) | hh["gap_h"].ne(1))
hh["episode"] = hh["start"].cumsum().where(hh["high"])
hrows = []
for episode_id, block in hh.dropna(subset=["episode"]).groupby("episode", observed=True):
    peak_row = block.loc[block["max_any_station"].idxmax()]
    hrows.append({
        "episode_id": int(episode_id),
        "start": block["timestamp"].min(),
        "end": block["timestamp"].max(),
        "hours": int(len(block)),
        "peak": float(block["max_any_station"].max()),
        "peak_time": peak_row["timestamp"],
        "mean_n_stations_ge7": float(block["n_stations_ge7"].mean()),
    })
hep = pd.DataFrame(hrows).sort_values("start").reset_index(drop=True)
save_table(hep, "t31_episodes_hourly.csv")

fig, ax = plt.subplots(figsize=(8.6, 4.4))
bins = np.arange(0.5, hep["hours"].max() + 1.5, 1)
ax.hist(hep["hours"], bins=bins, color="#d62728", edgecolor="white")
ax.set_xlabel("episode length (consecutive hours with any station >= 7)")
ax.set_ylabel("episodes")
ax.set_title(f"F27 {len(hep)} hourly episodes: median {hep['hours'].median():.0f} h, 90th pct {hep['hours'].quantile(0.9):.0f} h, longest {int(hep['hours'].max())} h")
stamp(ax, f"share <= 6 h: {(hep['hours']<=6).mean()*100:.0f}%\nshare >= 24 h: {(hep['hours']>=24).mean()*100:.0f}%", "upper right")
save_fig(fig, "F27_hourly_episode_duration.png")

findings.append({
    "section": "episodes", "finding": "Even the hourly episodes are short",
    "value": f"median {hep['hours'].median():.0f} h, {(hep['hours']<=6).mean()*100:.0f}% <= 6 h, longest {int(hep['hours'].max())} h",
    "unit": "hours", "figure": "F27",
    "reading": "A bad air day is a burst of hours, not a constant condition; timing the outdoor window matters.",
})

write_findings(findings, "findings_03.csv")
print("03 episodes done")
print(summary.to_string(index=False))
print(f"  daily episodes {len(episodes)}, median {episodes['days'].median():.1f}, longest {episodes['days'].max()}")
print(f"  persistence base {base:.3f}, after high {p_next_given_high:.3f}, after not high {p_next_given_low:.3f}")
print(f"  hourly episodes {len(hep)}, median {hep['hours'].median():.0f}, longest {hep['hours'].max()}")
print(f"  high-day station spread: <=3 stations {vc.loc[1:3].sum()}/{vc.sum()}")
