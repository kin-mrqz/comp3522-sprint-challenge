"""01 - data trust: missingness, substitution, censoring, coverage.

Figures F01-F05, tables t01-t04.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_lib import (FIG, META, PROC, TAB, TYPE_COLOR, save_fig, save_table, stamp,
                     write_findings)

h = pd.read_parquet(PROC / "aqhi_hourly.parquet")
dq = pd.read_csv(PROC / "data_quality_station_year.csv")
order = META["station"].tolist()
findings = []

# ---------------------------------------------------------------- F01 missing
piv = dq.pivot(index="station", columns="year", values="missing_pct").reindex(order)
fig, ax = plt.subplots(figsize=(9.5, 6.4))
im = ax.imshow(piv.to_numpy() * 100, aspect="auto", cmap="Reds", vmin=0,
               vmax=max(5.0, float(np.nanmax(piv.to_numpy())) * 100))
ax.set_xticks(range(len(piv.columns)), piv.columns)
ax.set_yticks(range(len(piv.index)), piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        value = piv.iloc[i, j] * 100
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7.5,
                color="white" if value > 0.55 * max(5.0, np.nanmax(piv.to_numpy()) * 100) else "0.2")
ax.set_title(f"F01 Missing hourly AQHI by station-year (overall {h['aqhi'].isna().mean()*100:.2f}% of cells)")
ax.set_xlabel("year")
fig.colorbar(im, ax=ax, label="% of expected hours missing")
save_fig(fig, "F01_missingness_station_year.png")
save_table(piv.reset_index(), "t01_missingness_station_year.csv")

worst = dq.sort_values("missing_pct", ascending=False).head(5)
findings.append({
    "section": "data_trust", "finding": "Worst station-year missingness",
    "value": f"{worst.iloc[0]['station']} {int(worst.iloc[0]['year'])}: {worst.iloc[0]['missing_pct']*100:.1f}%",
    "unit": "% of expected hours", "figure": "F01",
    "reading": "Tap Mun and 2023 carry the missingness; every other station-year is near 1%.",
})

# ------------------------------------------------------------ F02 substitution
sub = dq.pivot(index="station", columns="year", values="substituted_pct").reindex(order)
fig, ax = plt.subplots(figsize=(9.5, 6.4))
im = ax.imshow(sub.to_numpy() * 100, aspect="auto", cmap="Purples", vmin=0,
               vmax=max(1.0, float(np.nanmax(sub.to_numpy())) * 100))
ax.set_xticks(range(len(sub.columns)), sub.columns)
ax.set_yticks(range(len(sub.index)), sub.index)
for i in range(sub.shape[0]):
    for j in range(sub.shape[1]):
        value = sub.iloc[i, j] * 100
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7.5,
                color="white" if value > 0.6 * np.nanmax(sub.to_numpy()) * 100 else "0.2")
ax.set_title(f"F02 Share of hours flagged '*' (value borrowed from a similar station): {h['substituted'].mean()*100:.2f}% overall")
ax.set_xlabel("year")
fig.colorbar(im, ax=ax, label="% of hours substituted")
save_fig(fig, "F02_substitution_station_year.png")
save_table(sub.reset_index(), "t02_substitution_station_year.csv")

top_sub = dq.sort_values("substituted_pct", ascending=False).head(5)
findings.append({
    "section": "data_trust", "finding": "Station-years relying most on substituted values",
    "value": "; ".join(f"{r.station} {int(r.year)} {r.substituted_pct*100:.1f}%" for r in top_sub.itertuples()),
    "unit": "% of hours", "figure": "F02",
    "reading": "Substitution is not uniform: a few station-years are mostly borrowed data, so a station ranking must be read with that in mind.",
})

# --------------------------------------------------------- F03 value histogram
counts = h["aqhi"].value_counts().reindex([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).fillna(0).astype(int)
censored = int(h["censored"].sum())
own = int(counts.sum() - censored)
low_share = float((h["aqhi"] <= 3).fillna(False).mean())
mod_share = float(((h["aqhi"] >= 4) & (h["aqhi"] <= 6)).fillna(False).mean())
high_share = float((h["aqhi"] >= 7).fillna(False).mean())
fig, ax = plt.subplots(figsize=(9.5, 4.8))
bars = ax.bar(range(1, 11), counts.to_numpy(), color=["#2ca02c"] * 3 + ["#ffbf00"] * 3 + ["#ff7f0e"] + ["#d62728"] * 3)
ax.set_xticks(range(1, 11))
ax.set_xlabel("hourly AQHI value (10 includes the 304 '10+' cells)")
ax.set_ylabel("hours")
ax.set_title(f"F03 Hourly AQHI distribution: 1-3 low {low_share*100:.1f}%, 4-6 moderate {mod_share*100:.1f}%, >=7 high {high_share*100:.2f}%")
total = len(h)  # denominator includes missing cells, so bars sum to 98.75%
for x, v in zip(range(1, 11), counts.to_numpy()):
    ax.text(x, v + total * 0.004, f"{v/total*100:.1f}%", ha="center", fontsize=8)
stamp(ax, f"n = {total:,} station-hours\n'+10' cells = {censored:,} ({censored/total*100:.2f}%)", "upper right")
save_fig(fig, "F03_value_distribution.png")
dist = pd.DataFrame({"aqhi": counts.index, "hours": counts.to_numpy()})
dist["share"] = dist["hours"] / total
dist.loc[dist["aqhi"] == 10, "note"] = f"includes {censored} censored 10+ cells"
save_table(dist, "t03_value_distribution.csv")

findings.append({
    "section": "data_trust", "finding": "AQHI is dominated by the low band",
    "value": f"low 1-3 = {low_share*100:.1f}%, high >=7 = {high_share*100:.2f}%",
    "unit": "% of all station-hours", "figure": "F03",
    "reading": "The index is discrete 1-10; any mean near 3 is a low-air-pollution baseline, and 7+ is the health-relevant tail.",
})

# ------------------------------------------------------ F04 availability raster
h["_obs"] = h["aqhi"].notna().astype(int)
avail = (h.groupby(["date", "station"], observed=True)["_obs"].sum()
         .unstack("station").reindex(columns=order) / 24.0)
fig, ax = plt.subplots(figsize=(13, 5.2))
im = ax.imshow(avail.T.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1,
               interpolation="nearest")
years = pd.date_range(avail.index.min(), avail.index.max(), freq="YS")
ax.set_xticks([avail.index.get_indexer([y], method="nearest")[0] for y in years],
              [y.year for y in years])
ax.set_yticks(range(len(order)), order)
ax.set_title("F04 Daily data capture by station: dark blocks are gaps, not low AQHI")
ax.set_xlabel("date")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("share of 24 hours reported")
for day, label, ypos in [("2023-08-25", "Aug 2023 gap", -1.5),
                         ("2023-10-01", "Oct 2023", -2.6),
                         ("2024-12-01", "Dec 2024 gap", -1.5)]:
    x = avail.index.get_indexer([pd.Timestamp(day)], method="nearest")[0]
    ax.axvline(x, color="white", lw=0.9, ls="--", alpha=0.9)
    ax.annotate(label, xy=(x, 17.6), xytext=(x, ypos), textcoords=("data", "data"),
                ha="center", fontsize=7.5, color="0.15", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))
save_fig(fig, "F04_daily_capture_raster.png")
save_table(avail.reset_index(), "t04_daily_capture.csv")

# ------------------------------------------------- F05 missingness by hour
miss_hour = (h.assign(m=h["aqhi"].isna())
             .groupby(["station", "hour_of_day"], observed=True)["m"].mean()
             .unstack("hour_of_day").reindex(order))
fig, ax = plt.subplots(figsize=(12, 5.2))
im = ax.imshow(miss_hour.to_numpy() * 100, aspect="auto", cmap="Reds", vmin=0,
               vmax=max(1.0, float(np.nanmax(miss_hour.to_numpy())) * 100))
ax.set_yticks(range(len(order)), order)
ax.set_xticks(range(0, 24, 2), [f"{x:02d}" for x in range(0, 24, 2)])
ax.set_xlabel("hour of day (hour beginning)")
ax.set_title("F05 Missingness by station and hour of day: gaps are whole-day outages, not hourly drops")
fig.colorbar(im, ax=ax, label="% missing")
save_fig(fig, "F05_missingness_by_hour.png")
save_table(miss_hour.reset_index(), "t05_missingness_by_hour.csv")

# ------------------------------------------------------------- gap structure
def longest_run(flags: pd.Series) -> int:
    best = cur = 0
    for flag in flags:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best

gap_rows = []
for station in order:
    daily_obs = h[h["station"] == station].groupby("date", observed=True)["aqhi"].count()
    full = pd.date_range(daily_obs.index.min(), daily_obs.index.max(), freq="D")
    present = daily_obs.reindex(full).fillna(0).to_numpy()
    gap_rows.append({
        "station": station,
        "days_with_no_value": int((present == 0).sum()),
        "days_partial": int(((present > 0) & (present < 24)).sum()),
        "days_complete": int((present == 24).sum()),
        "longest_run_days_no_value": longest_run(present == 0),
        "total_missing_hours": int(24 * len(present) - present.sum()),
    })
gaps = pd.DataFrame(gap_rows)
save_table(gaps, "t06_gap_structure.csv")
worst_gap = gaps.sort_values("longest_run_days_no_value", ascending=False).iloc[0]
findings.append({
    "section": "data_trust", "finding": "Longest continuous outage",
    "value": f"{worst_gap.station}: {int(worst_gap.longest_run_days_no_value)} days with no value",
    "unit": "days", "figure": "F04",
    "reading": "The export has three outage shapes: 38 absent dates, 17 all-station blank days in 2023, and a Tap Mun outage in Sep-Oct 2023.",
})

# --------------------------------------------------- substitution vs own value
own_mean = h.loc[~h["substituted"], "aqhi"].mean()
sub_mean = h.loc[h["substituted"], "aqhi"].mean()
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.bar(["own measurement", "substituted '*'" ], [own_mean, sub_mean],
       color=["#4c72b0", "#9467bd"])
for i, v in enumerate([own_mean, sub_mean]):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, max(own_mean, sub_mean) * 1.25)
ax.set_ylabel("mean AQHI")
ax.set_title(f"F06 Substituted hours read {sub_mean - own_mean:+.3f} AQHI vs own measurements")
stamp(ax, f"own n={int((~h['substituted']).sum()):,}\nsub n={int(h['substituted'].sum()):,}", "upper right")
save_fig(fig, "F06_substitution_bias.png")
findings.append({
    "section": "data_trust", "finding": "Substitution is not neutral for the level",
    "value": f"own {own_mean:.3f} vs substituted {sub_mean:.3f} ({sub_mean-own_mean:+.3f})",
    "unit": "mean AQHI", "figure": "F06",
    "reading": "Borrowed values sit slightly away from own measurements, so station means mix two data-generating processes; report a sensitivity without them.",
})

# -------------------------------------------------------------- token counts
tokens = h["raw_value"].value_counts(dropna=False).rename_axis("raw_token").reset_index(name="n_cells")
tokens["share"] = tokens["n_cells"] / len(h)
save_table(tokens, "t07_raw_token_counts.csv")

write_findings(findings, "findings_01.csv")
print("01 profile done")
print(f"  missing overall      {h['aqhi'].isna().mean():.4%}")
print(f"  substituted overall  {h['substituted'].mean():.4%}")
print(f"  censored 10+ cells   {censored:,}")
print(f"  own mean {own_mean:.4f}  substituted mean {sub_mean:.4f}")
print(gaps.sort_values("longest_run_days_no_value", ascending=False).head(4).to_string(index=False))
