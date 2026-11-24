"""00 - build the daily AQHI series and audit the modelling window.

Figures F01-F03, tables t00-t03, facts series.json.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsa_lib import (DATA, DOW_ABBR, FULL_END, MONTH_ABBR, TEST_END, TEST_START, TRAIN_END,
                     TRAIN_START, VAL_END, VAL_START, save_facts, save_fig, save_table,
                     load_daily, splits)

series = load_daily()
parts = splits(series)
train, val, test, full = parts["train"], parts["val"], parts["test"], parts["full"]

# ------------------------------------------------------------------ audit
audit = pd.DataFrame([
    ("first_date", str(series.index.min().date())),
    ("last_date", str(series.index.max().date())),
    ("calendar_days", len(series)),
    ("days_with_value", int(series.notna().sum())),
    ("missing_days", int(series.isna().sum())),
    ("train_start", TRAIN_START), ("train_end", TRAIN_END),
    ("train_days", len(train)), ("train_observed", int(train.notna().sum())),
    ("val_start", VAL_START), ("val_end", VAL_END),
    ("val_days", len(val)), ("val_observed", int(val.notna().sum())),
    ("test_start", TEST_START), ("test_end", TEST_END),
    ("test_days", len(test)), ("test_observed", int(test.notna().sum())),
    ("full_fit_end", FULL_END), ("full_fit_observed", int(full.notna().sum())),
    ("train_mean", round(float(train.mean()), 4)),
    ("train_sd", round(float(train.std()), 4)),
    ("train_min", round(float(train.min()), 4)),
    ("train_max", round(float(train.max()), 4)),
    ("val_mean", round(float(val.mean()), 4)),
    ("test_mean", round(float(test.mean()), 4)),
], columns=["metric", "value"])
save_table(audit, "t00_series_audit.csv")
audit.to_csv(DATA / "series_audit.csv", index=False)

# daily series with splits
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(train.index, train.to_numpy(), color="#4c72b0", lw=0.7, label="train 2021-2023")
ax.plot(val.index, val.to_numpy(), color="#ff7f0e", lw=0.7, label="validation 2024 (Jan-Nov)")
ax.plot(test.index, test.to_numpy(), color="#2ca02c", lw=0.7, label="test 2025")
ax.axvline(pd.Timestamp(TRAIN_END), color="0.3", ls="--", lw=0.8)
ax.axvline(pd.Timestamp(TEST_START), color="0.3", ls="--", lw=0.8)
for start, end in [("2023-01-07", "2023-01-07"), ("2023-06-27", "2023-06-27"),
                   ("2023-08-25", "2023-08-31"), ("2023-10-01", "2023-10-15"),
                   ("2024-12-01", "2024-12-31")]:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), color="0.85", alpha=0.6, zorder=0)
ax.set_ylabel("daily territory mean AQHI")
ax.set_title(f"F01 Daily AQHI series and the frozen splits: train {len(train)} d, validation {len(val)} d, test {len(test)} d")
ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper right")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.text(pd.Timestamp("2023-08-20"), 7.2, "gaps", fontsize=7.5, color="0.35")
save_fig(fig, "F01_series_and_splits.png")

# monthly and annual means
monthly = series.resample("MS").mean()
annual = series.resample("YS").mean()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [2.4, 1]})
axes[0].plot(monthly.index, monthly.to_numpy(), marker="o", ms=3, color="#333333", lw=1.2)
axes[0].set_title(f"monthly mean, {int(monthly.notna().sum())} months")
axes[0].set_ylabel("AQHI")
axes[0].xaxis.set_major_locator(mdates.YearLocator())
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].bar(annual.index.year, annual.to_numpy(), color="#4c72b0", width=0.6)
for i, value in enumerate(annual.to_numpy()):
    if np.isfinite(value):
        axes[1].text(annual.index.year[i], value + 0.03, f"{value:.2f}", ha="center", fontsize=8)
axes[1].set_ylim(3.0, 3.55)
axes[1].set_title("annual mean")
axes[1].set_xticks(annual.index.year)
fig.suptitle("F02 Monthly and annual means: a strong annual cycle, no visible level drift", fontweight="bold", y=1.02)
save_fig(fig, "F02_monthly_annual.png")
save_table(monthly.rename("monthly_mean").reset_index(), "t01_monthly_mean.csv")
save_table(annual.rename("annual_mean").reset_index(), "t02_annual_mean.csv")

# seasonal profiles
month_profile = train.groupby(train.index.month).agg(["mean", "std", "count"]).reindex(range(1, 13))
dow_profile = train.groupby(train.index.dayofweek).agg(["mean", "std", "count"]).reindex(range(7))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
axes[0].bar(range(1, 13), month_profile["mean"], color="#4c72b0")
axes[0].set_xticks(range(1, 13), MONTH_ABBR)
axes[0].set_ylim(2.2, 4.1)
axes[0].set_title(f"month profile: {month_profile['mean'].min():.2f} in {MONTH_ABBR[month_profile['mean'].idxmin()-1]} to {month_profile['mean'].max():.2f} in {MONTH_ABBR[month_profile['mean'].idxmax()-1]}")
axes[0].set_ylabel("mean AQHI")
axes[1].bar(range(7), dow_profile["mean"], color="#c44e52")
axes[1].set_xticks(range(7), DOW_ABBR)
axes[1].set_ylim(3.0, 3.5)
axes[1].set_title(f"day-of-week profile: range {dow_profile['mean'].max()-dow_profile['mean'].min():.3f}")
fig.suptitle("F03 Deterministic seasonal profiles in the training window", fontweight="bold", y=1.02)
save_fig(fig, "F03_seasonal_profiles.png")
save_table(month_profile.reset_index().rename(columns={"index": "month"}), "t03_month_profile.csv")
save_table(dow_profile.reset_index().rename(columns={"index": "dow"}), "t04_dow_profile.csv")

series.rename("aqhi").to_csv(DATA / "daily_series.csv")
save_facts({
    "audit": dict(zip(audit["metric"], audit["value"])),
    "train_mean": float(train.mean()),
    "val_mean": float(val.mean()),
    "test_mean": float(test.mean()),
    "train_sd": float(train.std()),
    "month_profile": {MONTH_ABBR[m - 1]: float(month_profile.loc[m, "mean"]) for m in range(1, 13)},
    "dow_profile": {DOW_ABBR[d]: float(dow_profile.loc[d, "mean"]) for d in range(7)},
}, "series.json")

print("00 series built")
print(audit.to_string(index=False))
print(f"train {len(train)} (obs {train.notna().sum()}), val {len(val)}, test {len(test)}")
print(f"month profile range {month_profile['mean'].min():.3f}-{month_profile['mean'].max():.3f}")
print(f"dow range {dow_profile['mean'].min():.3f}-{dow_profile['mean'].max():.3f}")
