"""06 - assemble the EDA findings sheet, figure inventory, and completeness check.

Reads findings_*.csv, the processed tables, and every PNG in EDA/results/figures.
Writes:
  EDA/results/tables/t55_all_findings.csv
  EDA/results/figure_inventory.csv
  EDA/results/table_inventory.csv
  EDA/results/EDA_FINDINGS.md
  EDA/results/headline_numbers.json
"""
from __future__ import annotations

import json

import pandas as pd

from eda_lib import FIG, PROC, RES, SEASON_ORDER, TAB

SECTIONS = [
    ("data_trust", "1. Data trust", "01_profile.py", "F01-F06"),
    ("temporal", "2. Temporal structure", "02_temporal.py", "F07-F20"),
    ("episodes", "3. Episodes and persistence", "03_episodes.py", "F21-F27"),
    ("spatial", "4. Spatial structure", "04_spatial.py", "F28-F40"),
    ("decision", "5. Decision layer", "05_decision.py", "F41-F46"),
]

frames = []
for idx, (key, label, script, figs) in enumerate(SECTIONS, start=1):
    path = TAB / f"findings_{idx:02d}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run {script} first")
    frame = pd.read_csv(path)
    frame["section"] = key
    frame["section_label"] = label
    frame["source_script"] = script
    frames.append(frame)
all_findings = pd.concat(frames, ignore_index=True)
all_findings.to_csv(TAB / "t55_all_findings.csv", index=False)

# ------------------------------------------------------------- inventories
figures = pd.DataFrame([
    {"figure": path.stem.split("_")[0], "file": path.name, "bytes": path.stat().st_size,
     "path": str(path.relative_to(RES.parent))}
    for path in sorted(FIG.glob("*.png"))
])
figures.to_csv(RES / "figure_inventory.csv", index=False)
tables = pd.DataFrame([
    {"table": path.name, "bytes": path.stat().st_size, "path": str(path.relative_to(RES.parent))}
    for path in sorted(TAB.glob("*.csv"))
])
tables.to_csv(RES / "table_inventory.csv", index=False)

# ---------------------------------------------------------- headline numbers
audit = pd.read_csv(PROC / "t00_file_audit.csv").set_index("metric")["value"]
annual = pd.read_csv(TAB / "t14_annual_summary.csv").set_index("year")
diurnal = pd.read_csv(TAB / "t12_diurnal_by_season.csv").set_index("season")
monthly = pd.read_csv(TAB / "t09_monthly_mean_by_year.csv").set_index("month")
monthly_fy = monthly[[str(y) for y in (2021, 2022, 2023, 2024, 2025)]].mean(axis=1)
winter_mean = float(monthly_fy.loc[[12, 1, 2]].mean())
summer_mean = float(monthly_fy.loc[[6, 7, 8]].mean())
epi = pd.read_csv(TAB / "t25_episode_summary_year.csv").set_index("year")
hep = pd.read_csv(TAB / "t31_episodes_hourly.csv")
pers = pd.read_csv(TAB / "t28_daily_persistence.csv").set_index("condition")
station = pd.read_csv(TAB / "t32_station_summary.csv")
spread = pd.read_csv(TAB / "t37_spatial_range_by_season.csv").set_index("season")
risk = pd.read_csv(TAB / "t45_prob_ge7_month_hour.csv", index_col=0)
practical = pd.read_csv(TAB / "t54_practical_month_guide.csv")
trend = pd.read_csv(TAB / "t23_trend_tests.csv").set_index("window")
td = pd.read_parquet(PROC / "aqhi_daily_territory.parquet")
td["high"] = td["max_station_max"] >= 7
season_counts = td.loc[td["high"]].groupby("season").size().reindex(SEASON_ORDER).fillna(0)
high_share = pd.read_csv(TAB / "t49_safe_days_by_month_shares.csv", index_col=0)
high_month_share = (1 - high_share[["Low (1-3)", "Moderate (4-6)"]].sum(axis=1))

headlines = {
    "data": {
        "raw_rows": int(audit["raw_rows_including_daily_max"]),
        "stations": int(audit["station_columns"]),
        "hourly_station_rows": int(audit["cells_in_hourly_long"]),
        "date_min": str(audit["date_min"]),
        "date_max": str(audit["date_max"]),
        "missing_dates": int(audit["calendar_days_missing"]),
        "missing_share": float(audit["cells_missing"]) / float(audit["cells_in_hourly_long"]),
        "substituted_share": float(audit["cells_substituted"]) / float(audit["cells_in_hourly_long"]),
        "censored_cells": int(audit["cells_censored_10plus"]),
    },
    "level": {
        "annual_means": {str(y): round(float(v), 3) for y, v in annual["mean"].items()},
        "winter_mean": round(winter_mean, 3),
        "summer_mean": round(summer_mean, 3),
        "trend_2021_2025_per_year": round(float(trend.loc["2021-2025", "slope_per_year"]), 3),
        "trend_p": round(float(trend.loc["2021-2025", "p_value"]), 2),
    },
    "episodes": {
        "high_days": int(td["high"].sum()),
        "high_day_share": round(float(td["high"].sum() / td["in_file"].sum()), 3),
        "high_days_by_season_share": {s: round(float(season_counts[s] / season_counts.sum()), 3) for s in SEASON_ORDER},
        "high_day_share_by_month": {str(int(m)): round(float(v), 3) for m, v in high_month_share.items()},
        "longest_episode_days": int(epi["longest_episode_days"].max()),
        "median_hourly_episode_hours": int(hep["hours"].median()),
    },
    "persistence": {
        "base": round(float(pers.loc["base_rate", "p_tomorrow_high"]), 3),
        "after_high": round(float(pers.loc["today_high", "p_tomorrow_high"]), 3),
        "after_not_high": round(float(pers.loc["today_not_high", "p_tomorrow_high"]), 3),
    },
    "spatial": {
        "roadside_mean": round(float(station.loc[station["type"] == "roadside", "mean"].mean()), 2),
        "general_mean": round(float(station.loc[station["type"] == "general", "mean"].mean()), 2),
        "spatial_range_median": round(float(spread["median"].median()), 2),
    },
    "decision": {
        "worst_month_hour_p_ge7": round(float(risk.max().max()), 4),
        "best_window": str(practical["best_window"].mode().iloc[0]),
        "june_high_day_share": round(float(high_month_share.loc[6]), 3),
        "august_high_day_share": round(float(high_month_share.loc[8]), 3),
        "january_high_day_share": round(float(high_month_share.loc[1]), 3),
        "april_high_day_share": round(float(high_month_share.loc[4]), 3),
        "september_high_day_share": round(float(high_month_share.loc[9]), 3),
    },
}
(RES / "headline_numbers.json").write_text(json.dumps(headlines, indent=2), encoding="utf-8")

# ------------------------------------------------------------- findings md
lines = ["# AQHI EDA - findings sheet", "",
         "Generated by `EDA/code/06_report.py` from `EDA/results/tables/findings_*.csv`.", ""]
for key, label, script, figs in SECTIONS:
    block = all_findings[all_findings["section"] == key]
    lines += [f"## {label}", "", f"Script `EDA/code/{script}`; figures {figs}.", "",
              "| Finding | Value | Unit | Figure | Reading |", "|---|---|---|---|---|"]
    for row in block.itertuples():
        lines.append(f"| {row.finding} | {row.value} | {row.unit} | {row.figure} | {row.reading} |")
    lines.append("")

lines += ["## Figure inventory", "", "| Figure | File | Size (kB) |", "|---|---|---|"]
for row in figures.itertuples():
    lines.append(f"| {row.figure} | {row.file} | {row.bytes/1024:.0f} |")
lines += ["", "## Table inventory", "", "| Table | Size (kB) |", "|---|---|"]
for row in tables.itertuples():
    if row.table == "t55_all_findings.csv":
        continue
    lines.append(f"| {row.table} | {row.bytes/1024:.1f} |")
(RES / "EDA_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# --------------------------------------------------------- completeness check
expected = {f"F{i:02d}" for i in range(1, 47)}
present = set(figures["figure"])
print(f"figures: {len(figures)} (expected 46); missing: {sorted(expected - present)}; extra: {sorted(present - expected)}")
report_path = RES / "EDA_REPORT.md"
if report_path.exists():
    text = report_path.read_text(encoding="utf-8")
    missing_in_report = sorted(code for code in expected if code not in text)
    print(f"figures missing from EDA_REPORT.md: {missing_in_report}")
print(f"tables: {len(tables)}")
print(f"findings: {len(all_findings)}")
print(all_findings.groupby("section").size().to_string())
