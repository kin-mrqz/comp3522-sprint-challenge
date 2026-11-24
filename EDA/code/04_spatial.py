"""04 - spatial structure: station levels, co-movement, spread, region and type.

Figures F28-F36, tables t32-t39.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from eda_lib import (FULL_YEARS, META, PROC, REGION_COLOR, REGION_ORDER, SEASON_ORDER,
                     TYPE_COLOR, save_fig, save_table, stamp, write_findings)

h = pd.read_parquet(PROC / "aqhi_hourly.parquet")
ds = pd.read_parquet(PROC / "aqhi_daily_station.parquet")
td = pd.read_parquet(PROC / "aqhi_daily_territory.parquet")
wide = pd.read_parquet(PROC / "aqhi_wide_hourly.parquet").set_index("timestamp")
h["aqhi"] = h["aqhi"].astype("float64")
ds["mean"] = ds["mean"].astype("float64")
ds["max"] = ds["max"].astype("float64")
td["spatial_range"] = td["spatial_range"].astype("float64")
td["mean_station_mean"] = td["mean_station_mean"].astype("float64")
wide = wide.astype("float64")
order = META["station"].tolist()
findings = []

# ------------------------------------------------------------- station table
rows = []
for station in order:
    block = ds[ds["station"] == station]
    daily_mean = block.set_index("date")["mean"]
    hourly_block = h[h["station"] == station]
    rows.append({
        "station": station,
        "region": META.loc[META["station"] == station, "region"].iloc[0],
        "type": META.loc[META["station"] == station, "station_type"].iloc[0],
        "days": int(daily_mean.notna().sum()),
        "mean": float(daily_mean.mean()),
        "median": float(daily_mean.median()),
        "sd_daily": float(daily_mean.std()),
        "se_mean": float(daily_mean.std() / np.sqrt(daily_mean.notna().sum())),
        "p90_daily": float(daily_mean.quantile(0.90)),
        "max": float(hourly_block["aqhi"].max()),
        "share_ge7": float((hourly_block["aqhi"] >= 7).mean()),
        "share_ge8": float((hourly_block["aqhi"] >= 8).mean()),
        "share_substituted": float(hourly_block["substituted"].mean()),
    })
station = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
save_table(station, "t32_station_summary.csv")

fig, ax = plt.subplots(figsize=(10.5, 5.6))
y = np.arange(len(station))[::-1]
ax.barh(y, station["mean"], xerr=1.96 * station["se_mean"], capsize=2,
        color=[TYPE_COLOR[t] for t in station["type"]])
ax.set_yticks(y, station["station"])
ax.set_xlim(2.6, 4.1)
ax.set_xlabel("mean AQHI (daily means, 2021-2026)")
ax.set_title(f"F28 Station ranking: {station.iloc[0]['station']} {station.iloc[0]['mean']:.2f} to {station.iloc[-1]['station']} {station.iloc[-1]['mean']:.2f}")
handles = [plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR[t]) for t in ["general", "roadside"]]
ax.legend(handles, ["general station", "roadside station"], frameon=False, loc="lower right")
ax.axvline(station["mean"].mean(), color="0.4", ls="--", lw=0.8)
stamp(ax, f"territory mean {station['mean'].mean():.3f}\nroadside premium {station.loc[station['type']=='roadside','mean'].mean()-station.loc[station['type']=='general','mean'].mean():+.3f}", "upper right")
save_fig(fig, "F28_station_ranking.png")

findings.append({
    "section": "spatial", "finding": "Roadside stations sit above general stations",
    "value": f"{station.loc[station['type']=='roadside','mean'].mean():.2f} vs {station.loc[station['type']=='general','mean'].mean():.2f}",
    "unit": "mean AQHI", "figure": "F28",
    "reading": "The 18-station average mixes two exposure regimes; the scenario should name the station or district, not 'Hong Kong'.",
})

# ------------------------------------------------ F29 station x month heatmap
sm = (h.dropna(subset=["aqhi"]).groupby(["station", "month"], observed=True)["aqhi"].mean()
      .unstack("month").reindex(order))
sm = sm.loc[station["station"].tolist()]
fig, ax = plt.subplots(figsize=(10.5, 6.0))
im = ax.imshow(sm.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=2.2, vmax=4.4)
ax.set_yticks(range(len(sm.index)), sm.index)
ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
for i in range(sm.shape[0]):
    for j in range(sm.shape[1]):
        value = sm.iloc[i, j]
        if pd.notna(value):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if value > 3.9 else "0.15")
ax.set_title("F29 Mean AQHI by station and month (rows ordered by station mean)")
fig.colorbar(im, ax=ax, label="mean AQHI")
save_fig(fig, "F29_station_month_heatmap.png")
save_table(sm.reset_index(), "t33_station_month_mean.csv")

# ------------------------------------------- F30/F31 correlation + clustering
daily_wide = ds.pivot(index="date", columns="station", values="mean").reindex(columns=order)
month_mean = daily_wide.groupby(daily_wide.index.month).transform("mean")
anom = daily_wide - month_mean
corr = anom.corr(min_periods=500)
level_corr = wide.corr(min_periods=5000)
save_table(corr.reset_index(), "t34_station_correlation_anomaly.csv")
save_table(level_corr.reset_index(), "t35_station_correlation_level.csv")

fig, ax = plt.subplots(figsize=(9.6, 8.0))
im = ax.imshow(corr.to_numpy(), cmap="RdYlBu_r", vmin=0.7, vmax=1.0)
ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8.5)
ax.set_yticks(range(len(order)), order, fontsize=8.5)
ax.set_title(f"F30 Daily-anomaly correlation: mean off-diagonal {corr.where(~np.eye(len(order), dtype=bool)).stack().mean():.2f}", pad=12)
fig.colorbar(im, ax=ax, label="Pearson r (daily station means)")
save_fig(fig, "F30_station_correlation.png")

distance = 1 - corr.to_numpy()
np.fill_diagonal(distance, 0.0)
condensed = squareform(distance, checks=False)
linkage = hierarchy.linkage(condensed, method="average")
fig, ax = plt.subplots(figsize=(8.6, 5.6))
hierarchy.dendrogram(linkage, labels=order, orientation="right", ax=ax, leaf_font_size=8.5,
                     color_threshold=0.5 * max(linkage[:, 2]))
ax.set_xlabel("1 - correlation (daily anomaly)")
ax.set_title("F31 Station clustering by day-to-day co-movement")
save_fig(fig, "F31_station_dendrogram.png")
clusters = hierarchy.fcluster(linkage, t=0.10, criterion="distance")
cluster_table = pd.DataFrame({"station": order, "cluster": clusters})
cluster_table["cluster_size"] = cluster_table.groupby("cluster")["station"].transform("size")
save_table(cluster_table, "t36_station_clusters.csv")
cluster_groups = {int(k): list(v) for k, v in cluster_table.groupby("cluster")["station"].apply(list).items()}

findings.append({
    "section": "spatial", "finding": "Stations are not one homogeneous network",
    "value": f"anomaly correlation off-diagonal mean {corr.where(~np.eye(len(order), dtype=bool)).stack().mean():.2f}",
    "unit": "Pearson r", "figure": "F30",
    "reading": "Co-movement is high but not uniform; a territory mean smooths real district differences.",
})

# --------------------------------------------------------- F32 spatial range
monthly_range = td.set_index("date")["spatial_range"].resample("MS").mean()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [2, 1]})
axes[0].plot(monthly_range.index, monthly_range.to_numpy(), color="#6a3d9a", lw=1.6)
axes[0].set_ylabel("mean daily max-min across stations")
axes[0].set_title("monthly mean spatial range")
axes[1].boxplot([td.loc[td["season"] == s, "spatial_range"].dropna().to_numpy() for s in SEASON_ORDER],
                tick_labels=SEASON_ORDER, patch_artist=True,
                boxprops=dict(facecolor="#cab2d6"))
axes[1].set_ylabel("daily range (AQHI)")
axes[1].set_title("by season")
fig.suptitle(f"F32 Between-station spread: median {td['spatial_range'].median():.2f} AQHI, 90th pct {td['spatial_range'].quantile(0.9):.2f}, max {td['spatial_range'].max():.1f}",
             fontweight="bold", y=1.02)
save_fig(fig, "F32_spatial_range.png")
save_table(td.groupby("season")["spatial_range"].agg(["median", "mean", "max"]).reindex(SEASON_ORDER).reset_index(),
           "t37_spatial_range_by_season.csv")

corr_range = td[["spatial_range", "mean_station_mean"]].dropna().corr().iloc[0, 1]
findings.append({
    "section": "spatial", "finding": "Spatial spread widens on dirty days",
    "value": f"daily range median {td['spatial_range'].median():.2f}, max {td['spatial_range'].max():.1f}; corr(range, territory mean) {corr_range:.2f}",
    "unit": "AQHI", "figure": "F32",
    "reading": "A territory mean hides that one district can be several bands worse than another on the same day.",
})

# ---------------------------------------------------- F33 region time series
ds["region"] = ds["station"].map(META.set_index("station")["region"])
region_daily = ds.pivot_table(index="date", columns="region", values="mean", aggfunc="mean")
region_month = region_daily.resample("MS").mean().reindex(columns=REGION_ORDER)
fig, ax = plt.subplots(figsize=(12.5, 4.8))
for region in REGION_ORDER:
    ax.plot(region_month.index, region_month[region], lw=1.5, color=REGION_COLOR[region], label=region)
ax.set_ylabel("mean AQHI")
ax.set_title(f"F33 Regional monthly means: NT general stations sit {region_month[REGION_ORDER[2]].mean():.2f} vs HK Island {region_month[REGION_ORDER[0]].mean():.2f}")
ax.legend(frameon=False, ncol=3)
save_fig(fig, "F33_region_series.png")
save_table(region_month.reset_index(), "t38_region_monthly.csv")
save_table(region_daily.describe().T.reset_index(), "t39_region_daily_summary.csv")

# ------------------------------------------------------- F34 rank by year
rank_rows = []
for year in FULL_YEARS:
    block = ds[ds["year"] == year].groupby("station")["mean"].mean()
    rank_rows.append(block.rank(ascending=False).rename(year))
ranks = pd.concat(rank_rows, axis=1).loc[order]
fig, ax = plt.subplots(figsize=(8.2, 6.2))
im = ax.imshow(ranks.to_numpy(), aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=18)
ax.set_xticks(range(len(FULL_YEARS)), FULL_YEARS)
ax.set_yticks(range(len(order)), order)
for i in range(ranks.shape[0]):
    for j in range(ranks.shape[1]):
        ax.text(j, i, f"{int(ranks.iloc[i, j])}", ha="center", va="center", fontsize=7.5,
                color="white" if ranks.iloc[i, j] <= 3 or ranks.iloc[i, j] >= 16 else "0.15")
ax.set_title(f"F34 Station rank by annual mean (1 = dirtiest): Spearman 2021 vs 2025 = {ranks[2021].corr(ranks[2025], method='spearman'):.2f}")
fig.colorbar(im, ax=ax, label="rank")
save_fig(fig, "F34_station_rank_by_year.png")
save_table(ranks.reset_index(), "t40_station_rank_by_year.csv")

# ---------------------------------------------- F35 station mean vs variability
fig, ax = plt.subplots(figsize=(8.6, 5.4))
for station_type in ["general", "roadside"]:
    block = station[station["type"] == station_type]
    ax.scatter(block["mean"], block["sd_daily"], s=70, color=TYPE_COLOR[station_type],
               label=station_type, edgecolor="white", lw=0.6)
for row in station.itertuples():
    ax.annotate(row.station, (row.mean, row.sd_daily), fontsize=7.2, xytext=(3, 3),
                textcoords="offset points", color="0.25")
ax.set_xlabel("mean daily AQHI")
ax.set_ylabel("SD of daily means")
ax.set_title(f"F35 Level vs variability: correlation {station['mean'].corr(station['sd_daily']):.2f}")
ax.legend(frameon=False)
save_fig(fig, "F35_station_mean_vs_sd.png")

# -------------------------------------------------- F36 diurnal by type
type_hour = (h.dropna(subset=["aqhi"]).groupby(["station_type", "hour_of_day"], observed=True)["aqhi"]
             .mean().unstack("hour_of_day"))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
colors = {"general": TYPE_COLOR["general"], "roadside": TYPE_COLOR["roadside"]}
for station_type in ["general", "roadside"]:
    axes[0].plot(range(24), type_hour.loc[station_type], marker="o", ms=3, lw=1.8,
                 color=colors[station_type], label=station_type)
axes[0].set_title("mean AQHI by hour")
axes[0].set_xlabel("hour of day")
axes[0].legend(frameon=False)
prem = type_hour.loc["roadside"] - type_hour.loc["general"]
axes[1].bar(range(24), prem.to_numpy(), color="#c44e52")
axes[1].set_title(f"roadside minus general: {prem.min():+.2f} to {prem.max():+.2f}")
axes[1].set_xlabel("hour of day")
for ax in axes:
    ax.set_xticks(range(0, 24, 3), [f"{x:02d}" for x in range(0, 24, 3)])
fig.suptitle(f"F36 Roadside premium peaks at {prem.idxmax():02d}:00 ({prem.max():+.2f} AQHI)", fontweight="bold", y=1.02)
save_fig(fig, "F36_type_diurnal.png")
save_table(type_hour.reset_index(), "t41_type_diurnal.csv")

findings.append({
    "section": "spatial", "finding": "The roadside gap is largest in the evening",
    "value": f"premium {prem.min():+.2f} to {prem.max():+.2f}, peak hour {prem.idxmax():02d}:00",
    "unit": "mean AQHI", "figure": "F36",
    "reading": "A general-station reading understates roadside exposure exactly when people commute.",
})

# ------------------------------------------------- F37 exceedance by station
fig, ax = plt.subplots(figsize=(9.6, 5.4))
plot = station.sort_values("share_ge7")
ax.barh(plot["station"], plot["share_ge7"] * 100, color=[TYPE_COLOR[t] for t in plot["type"]])
ax.set_xlabel("% of hours at AQHI >= 7")
ax.set_title(f"F37 Exceedance share by station: {plot['share_ge7'].min()*100:.1f}% to {plot['share_ge7'].max()*100:.1f}%")
stamp(ax, f"network {h['aqhi'].ge(7).mean()*100:.1f}% of station-hours", "lower right")
save_fig(fig, "F37_station_exceedance.png")

# ------------------------------------- F38/F39 station x hour heatmaps
station_hour_mean = (h.dropna(subset=["aqhi"]).groupby(["station", "hour_of_day"], observed=True)["aqhi"]
                     .mean().unstack("hour_of_day").loc[station["station"].tolist()])
station_hour_high = (h.dropna(subset=["aqhi"]).assign(ge7=lambda d: d["aqhi"] >= 7)
                     .groupby(["station", "hour_of_day"], observed=True)["ge7"].mean()
                     .unstack("hour_of_day").loc[station["station"].tolist()])

fig, ax = plt.subplots(figsize=(12, 6.0))
im = ax.imshow(station_hour_mean.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=2.4, vmax=4.4)
ax.set_yticks(range(len(station_hour_mean.index)), station_hour_mean.index)
ax.set_xticks(range(0, 24, 2), [f"{x:02d}" for x in range(0, 24, 2)])
ax.set_xlabel("hour of day (hour beginning)")
ax.set_title("F38 Mean AQHI by station and hour: every station peaks at 16:00-17:00, the amplitude differs")
fig.colorbar(im, ax=ax, label="mean AQHI")
save_fig(fig, "F38_station_hour_mean.png")
save_table(station_hour_mean.reset_index(), "t42_station_hour_mean.csv")

fig, ax = plt.subplots(figsize=(12, 6.0))
im = ax.imshow(station_hour_high.to_numpy() * 100, aspect="auto", cmap="Reds", vmin=0, vmax=10)
ax.set_yticks(range(len(station_hour_high.index)), station_hour_high.index)
ax.set_xticks(range(0, 24, 2), [f"{x:02d}" for x in range(0, 24, 2)])
ax.set_xlabel("hour of day (hour beginning)")
ax.set_title(f"F39 Share of hours at AQHI >= 7 by station and hour: 16:00 max {station_hour_high.max().max()*100:.1f}% at {station_hour_high.max(axis=1).idxmax()}")
fig.colorbar(im, ax=ax, label="% of hours >= 7")
save_fig(fig, "F39_station_hour_exceedance.png")
save_table(station_hour_high.reset_index(), "t43_station_hour_exceedance.csv")

amplitude = (station_hour_mean.max(axis=1) - station_hour_mean.min(axis=1)).rename("diurnal_amplitude")
amp = pd.concat([amplitude, station.set_index("station")["share_ge7"], station.set_index("station")["mean"],
                 station.set_index("station")["type"]], axis=1).reset_index(names="station")
fig, ax = plt.subplots(figsize=(8.2, 5.0))
for station_type in ["general", "roadside"]:
    block = amp[amp["type"] == station_type]
    ax.scatter(block["diurnal_amplitude"], block["share_ge7"] * 100, s=70,
               color=TYPE_COLOR[station_type], label=station_type, edgecolor="white", lw=0.6)
for row in amp.itertuples():
    ax.annotate(row.station, (row.diurnal_amplitude, row.share_ge7 * 100), fontsize=7.2,
                xytext=(3, 3), textcoords="offset points", color="0.25")
ax.set_xlabel("diurnal amplitude (max hour mean - min hour mean, AQHI)")
ax.set_ylabel("% of hours at AQHI >= 7")
ax.set_title(f"F40 Larger afternoon swing goes with a heavier tail (r = {amp['diurnal_amplitude'].corr(amp['share_ge7']):.2f})")
ax.legend(frameon=False)
save_fig(fig, "F40_amplitude_vs_exceedance.png")
save_table(amp, "t44_amplitude_vs_exceedance.csv")

findings.append({
    "section": "spatial", "finding": "The heaviest afternoon tail is not the highest-mean station",
    "value": f"Tuen Mun 2.95% of hours >=7, Tap Mun 2.86%, Causeway Bay 1.35%; amplitude-tail r = {amp['diurnal_amplitude'].corr(amp['share_ge7']):.2f}",
    "unit": "% hours >= 7", "figure": "F40",
    "reading": "Rural and western New Territories stations swing harder in the afternoon and cross the high band more often, while roadside stations have a higher average but a thinner tail.",
})

write_findings(findings, "findings_04.csv")
print("04 spatial done")
print(station[["station", "type", "mean", "share_ge7"]].to_string(index=False))
print(f"  level corr mean {level_corr.where(~np.eye(len(order), dtype=bool)).stack().mean():.3f}")
print(f"  anomaly corr mean {corr.where(~np.eye(len(order), dtype=bool)).stack().mean():.3f}")
print(f"  spatial range median {td['spatial_range'].median():.3f}, corr with mean {corr_range:.3f}")
print(f"  clusters: {cluster_groups}")
