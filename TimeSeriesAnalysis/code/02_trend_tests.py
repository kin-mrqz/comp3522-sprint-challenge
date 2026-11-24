"""02 - is there a trend? Naive, seasonal, monthly, annual and bootstrap tests.

Figures F09-F13, tables t09-t12, facts trend.json.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from tsa_lib import (DOW_ABBR, MONTH_ABBR, ORIGIN, exog_frame, load_daily, save_facts, save_fig,
                     save_table, splits)

series = load_daily()
train = splits(series)["train"]
y = train.dropna()
idx = y.index
t = (y.index - ORIGIN).days.to_numpy(dtype=float)  # calendar time, gaps preserved
t_years = t / 365.25


def ols_trend(yvalues, tvalues, extra: pd.DataFrame | None = None, maxlags: int = 30):
    X = pd.DataFrame({"const": 1.0, "t": tvalues}, index=yvalues.index)
    if extra is not None:
        X = pd.concat([X, extra.reindex(yvalues.index)], axis=1)
    fit = sm.OLS(yvalues, X).fit()
    hac = sm.OLS(yvalues, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return fit, hac


def slope_row(label, fit, hac, scale, n, extra=""):
    i = list(fit.model.exog_names).index("t")
    return {
        "specification": label,
        "n": int(n),
        "slope_per_year": float(fit.params.iloc[i] * scale),
        "se_iid": float(fit.bse.iloc[i] * scale),
        "p_iid": float(fit.pvalues.iloc[i]),
        "se_hac": float(hac.bse.iloc[i] * scale),
        "p_hac": float(hac.pvalues.iloc[i]),
        "r2": float(fit.rsquared),
        "extra_regressors": extra,
    }


rows = []
# 1 naive daily trend
fit1, hac1 = ols_trend(y, t)
rows.append(slope_row("daily: linear trend only", fit1, hac1, 365.25, len(y)))
# 2 month dummies
month_dummies = pd.DataFrame({f"m{m}": (idx.month == m).astype(float) for m in range(2, 13)}, index=idx)
fit2, hac2 = ols_trend(y, t, month_dummies)
rows.append(slope_row("daily: trend + 11 month dummies", fit2, hac2, 365.25, len(y), "month dummies"))
# 3 Fourier + DOW
fourier_dow = exog_frame(idx, "fourier_dow", k=2)
fit3, hac3 = ols_trend(y, t, fourier_dow)
rows.append(slope_row("daily: trend + Fourier K=2 + DOW", fit3, hac3, 365.25, len(y), "Fourier K=2, DOW"))
# 4 Fourier only
fourier = exog_frame(idx, "fourier", k=2)
fit4, hac4 = ols_trend(y, t, fourier)
rows.append(slope_row("daily: trend + Fourier K=2", fit4, hac4, 365.25, len(y), "Fourier K=2"))

# monthly
monthly = series.resample("MS").mean().loc["2021-01-01":"2024-11-01"].dropna()
mt = np.arange(len(monthly), dtype=float)
Xm = pd.DataFrame({"const": 1.0, "t": mt}, index=monthly.index)
fitm = sm.OLS(monthly, Xm).fit()
hacm = sm.OLS(monthly, Xm).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
rows.append(slope_row("monthly: linear trend", fitm, hacm, 12, len(monthly)))
# annual
annual = series.resample("YS").mean().loc["2021-01-01":"2024-01-01"].dropna()
at = np.arange(len(annual), dtype=float)
Xa = pd.DataFrame({"const": 1.0, "t": at}, index=annual.index)
fita = sm.OLS(annual, Xa).fit()
haca = sm.OLS(annual, Xa).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
rows.append(slope_row("annual: linear trend (2024 is Jan-Nov)", fita, haca, 1, len(annual)))

# the same daily regression on the full 2021-2024 fit window
full = splits(series)["full"].dropna()
tfull = (full.index - ORIGIN).days.to_numpy(dtype=float)
fitfull, hacfull = ols_trend(full, tfull)
rows.append(slope_row("daily: 2021-2024 window", fitfull, hacfull, 365.25, len(full)))

trends = pd.DataFrame(rows)
save_table(trends, "t09_trend_regressions.csv")

# HAC at two bandwidths for the naive model
hac30 = rows[0]["p_hac"]
fit60 = sm.OLS(y.to_numpy(), sm.add_constant(t)).fit(cov_type="HAC", cov_kwds={"maxlags": 60})
p60 = float(fit60.pvalues[1])
se60 = float(fit60.bse[1] * 365.25)
print("naive HAC30 p", hac30, "HAC60 p", p60, "se60/year", se60)

# ------------------------------------------------------------- MK and Sen
def mann_kendall(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    s = 0
    for i in range(n - 1):
        s += np.sign(x[i + 1:] - x[i]).sum()
    var = n * (n - 1) * (2 * n + 5) / 18
    if s == 0:
        z = 0.0
    else:
        z = (s - np.sign(s)) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(s), float(z), float(p)


mk_s, mk_z, mk_p = mann_kendall(monthly.to_numpy())
sen_slope, sen_intercept, sen_low, sen_high = stats.theilslopes(monthly.to_numpy(), mt)
sen_year = float(sen_slope * 12)
save_table(pd.DataFrame([
    {"test": "Mann-Kendall on monthly means", "stat": mk_s, "z": mk_z, "p_value": mk_p,
     "slope_per_year": np.nan, "note": "supplementary"},
    {"test": "Sen slope on monthly means", "stat": sen_slope, "z": np.nan, "p_value": np.nan,
     "slope_per_year": sen_year, "note": "supplementary"},
    {"test": "Sen 95% interval (per month)", "stat": np.nan, "z": np.nan, "p_value": np.nan,
     "slope_per_year": np.nan, "note": f"{sen_low*12:.4f} to {sen_high*12:.4f} per year"},
]), "t10_monthly_trend_tests.csv")

# ------------------------------------------------------ block bootstrap
rng = np.random.default_rng(42)
naive_slope = float(np.polyfit(t, y.to_numpy(), 1)[0])
fitted = np.polyval(np.polyfit(t, y.to_numpy(), 1), t)
resid = y.to_numpy() - fitted
block = 30
reps = 2000
boot = []
n = len(y)
for _ in range(reps):
    starts = rng.integers(0, n - block, size=int(np.ceil(n / block)))
    index = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    sample = fitted + resid[index]
    boot.append(np.polyfit(t, sample, 1)[0] * 365.25)
boot = np.asarray(boot)
boot_ci = np.quantile(boot, [0.025, 0.975])
save_table(pd.DataFrame({"bootstrap_slope_per_year": boot}), "t11_block_bootstrap_slopes.csv")

# ------------------------------------------------------------- figures
annual_fit_line = np.polyval(np.polyfit(t, y.to_numpy(), 1), t)
seasonal_slope = float(fit3.params.iloc[1])
seasonal_line = float(fit3.params.iloc[0]) + seasonal_slope * t

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, yy, tt, label, row_i in [(axes[0], y, t, "2021-2023 (training)", 0),
                                 (axes[1], full, tfull, "2021-2024 (fit window)", 6)]:
    ax.plot(yy.index, yy.to_numpy(), color="0.75", lw=0.5, label="daily AQHI")
    roll = yy.rolling(365, min_periods=300).mean()
    ax.plot(roll.index, roll.to_numpy(), color="#1f77b4", lw=1.8, label="365-day rolling mean")
    line = np.polyval(np.polyfit(tt, yy.to_numpy(), 1), tt)
    ax.plot(yy.index, line, color="#d62728", lw=1.6, label="naive OLS trend")
    ax.set_title(f"{label}: {trends.iloc[row_i]['slope_per_year']:+.3f}/yr (HAC p = {trends.iloc[row_i]['p_hac']:.2f})")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0].set_ylabel("AQHI")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("F09 The trend depends on the endpoint: flat through 2023, downward only when the low 2024 is added",
             fontweight="bold", y=1.02)
save_fig(fig, "F09_trend_fits.png")

# forest plot
fig, ax = plt.subplots(figsize=(9.6, 4.6))
plot_trends = trends[~trends["specification"].str.startswith("annual")].reset_index(drop=True)
labels = plot_trends["specification"].tolist()
ypos = np.arange(len(labels))[::-1]
for i, row in enumerate(plot_trends.itertuples()):
    ax.plot([row.slope_per_year - 1.96 * row.se_iid, row.slope_per_year + 1.96 * row.se_iid],
            [ypos[i] + 0.12, ypos[i] + 0.12], color="#1f77b4", lw=1.4)
    ax.plot([row.slope_per_year - 1.96 * row.se_hac, row.slope_per_year + 1.96 * row.se_hac],
            [ypos[i] - 0.12, ypos[i] - 0.12], color="#d62728", lw=1.8)
    ax.plot(row.slope_per_year, ypos[i], "o", color="black", ms=4)
ax.axvline(0, color="0.3", lw=0.8)
ax.set_yticks(ypos, labels, fontsize=8.5)
ax.set_xlabel("trend in AQHI per year (95% interval)")
ax.set_title("F10 Trend slope under six specifications: the naive interval (blue) is too narrow; HAC (red) and seasonal controls centre on zero")
ax.plot([], [], color="#1f77b4", lw=1.4, label="iid 95% CI")
ax.plot([], [], color="#d62728", lw=1.8, label="HAC 95% CI")
ax.legend(frameon=False, loc="lower right")
save_fig(fig, "F10_trend_forest.png")

# monthly means
fig, ax = plt.subplots(figsize=(10.5, 4.4))
ax.plot(monthly.index, monthly.to_numpy(), marker="o", ms=3.5, color="#333333", lw=1.0, label="monthly mean")
ax.plot(monthly.index, np.polyval([float(fitm.params.iloc[1]), float(fitm.params.iloc[0])], mt),
        color="#d62728", lw=1.5, label=f"OLS {trends.iloc[4]['slope_per_year']:+.3f}/yr")
ax.plot(monthly.index, sen_intercept + sen_slope * mt, color="#2ca02c", lw=1.5, ls="--",
        label=f"Sen {sen_year:+.3f}/yr")
ax.set_ylabel("AQHI")
ax.set_title(f"F11 Monthly means: OLS {trends.iloc[4]['slope_per_year']:+.3f}/yr (p = {trends.iloc[4]['p_hac']:.2f}), Sen {sen_year:+.3f}/yr, Mann-Kendall p = {mk_p:.2f}")
ax.legend(frameon=False, fontsize=8.5)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
save_fig(fig, "F11_monthly_trend.png")

# bootstrap
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.hist(boot, bins=50, color="#6a3d9a", alpha=0.8)
ax.axvline(naive_slope * 365.25, color="black", lw=1.2, label="point estimate")
ax.axvline(0, color="#d62728", lw=1.2, ls="--", label="zero")
ax.axvspan(boot_ci[0], boot_ci[1], color="#d62728", alpha=0.08)
ax.set_xlabel("block-bootstrap trend (AQHI per year)")
ax.set_ylabel("replicates")
ax.set_title(f"F12 Residual block bootstrap: 95% interval {boot_ci[0]:+.3f} to {boot_ci[1]:+.3f} per year, straddles zero")
ax.legend(frameon=False)
save_fig(fig, "F12_block_bootstrap.png")

# rolling mean
rolling = y.rolling(365, min_periods=300).mean()
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(rolling.index, rolling.to_numpy(), color="#1f77b4", lw=1.8)
imin, imax = rolling.idxmin(), rolling.idxmax()
ax.scatter([imin, imax], [rolling.min(), rolling.max()], color="#d62728", zorder=5)
ax.annotate(f"min {rolling.min():.2f}\n{imin.date()}", (imin, rolling.min()), xytext=(8, -18),
            textcoords="offset points", fontsize=8)
ax.annotate(f"max {rolling.max():.2f}\n{imax.date()}", (imax, rolling.max()), xytext=(8, 8),
            textcoords="offset points", fontsize=8)
ax.set_ylabel("365-day rolling mean")
ax.set_title(f"F13 Rolling annual mean: range {rolling.max()-rolling.min():.2f} AQHI, no monotonic path")
save_fig(fig, "F13_rolling_mean.png")

rolling_summary = pd.DataFrame([
    {"stat": "rolling_min", "value": float(rolling.min()), "date": str(imin.date())},
    {"stat": "rolling_max", "value": float(rolling.max()), "date": str(imax.date())},
    {"stat": "rolling_range", "value": float(rolling.max() - rolling.min()), "date": ""},
    {"stat": "first_value", "value": float(rolling.dropna().iloc[0]), "date": str(rolling.dropna().index[0].date())},
    {"stat": "last_value", "value": float(rolling.dropna().iloc[-1]), "date": str(rolling.dropna().index[-1].date())},
])
save_table(rolling_summary, "t12_rolling_mean_summary.csv")

save_facts({
    "naive_slope_year": float(rows[0]["slope_per_year"]),
    "naive_p_iid": float(rows[0]["p_iid"]),
    "naive_p_hac30": float(rows[0]["p_hac"]),
    "naive_p_hac60": p60,
    "naive_se_hac60_year": se60,
    "seasonal_slope_year": float(rows[2]["slope_per_year"]),
    "seasonal_p_hac": float(rows[2]["p_hac"]),
    "month_dummy_slope_year": float(rows[1]["slope_per_year"]),
    "month_dummy_p_hac": float(rows[1]["p_hac"]),
    "monthly_slope_year": float(rows[4]["slope_per_year"]),
    "monthly_p_hac": float(rows[4]["p_hac"]),
    "annual_slope_year": float(rows[5]["slope_per_year"]),
    "annual_p": float(rows[5]["p_iid"]),
    "full_window_slope_year": float(rows[6]["slope_per_year"]),
    "full_window_p_iid": float(rows[6]["p_iid"]),
    "full_window_p_hac": float(rows[6]["p_hac"]),
    "mann_kendall_p": mk_p,
    "sen_slope_year": sen_year,
    "bootstrap_ci": [float(boot_ci[0]), float(boot_ci[1])],
    "rolling_min": float(rolling.min()), "rolling_min_date": str(imin.date()),
    "rolling_max": float(rolling.max()), "rolling_max_date": str(imax.date()),
    "rolling_range": float(rolling.max() - rolling.min()),
    "annual_means": {str(k.year): float(v) for k, v in annual.items()},
}, "trend.json")

print("02 trend done")
print(trends.to_string(index=False))
print(f"MK p {mk_p:.3f}, Sen {sen_year:+.4f}/yr")
print(f"bootstrap CI {boot_ci[0]:+.4f} to {boot_ci[1]:+.4f}")
print(f"rolling range {rolling.max()-rolling.min():.3f}")
