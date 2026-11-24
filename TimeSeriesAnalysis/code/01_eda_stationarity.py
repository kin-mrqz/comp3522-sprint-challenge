"""01 - stationarity, transformation and identification evidence.

Figures F04-F08, tables t05-t08, facts stationarity.json.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.unitroot import PhillipsPerron
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from tsa_lib import (MONTH_ABBR, PERIOD, save_facts, save_fig, save_table, load_daily, splits)

series = load_daily()
train = splits(series)["train"]
y = train.dropna()
logy = np.log(y)

# ------------------------------------------------------------- ADF grid
rows = []
for regression in ("n", "c", "ct"):
    for lag in range(1, 15):
        try:
            res = adfuller(y, maxlag=lag, regression=regression, autolag=None)
            stat, pvalue, usedlag, nobs = res[:4]
        except Exception as exc:  # noqa: BLE001
            rows.append({"series": "level", "regression": regression, "lag": lag,
                         "stat": np.nan, "p_value": np.nan, "nobs": np.nan, "error": str(exc)})
            continue
        rows.append({"series": "level", "regression": regression, "lag": lag,
                     "stat": float(stat), "p_value": float(pvalue), "nobs": int(nobs)})
for name, values in [("log_level", logy), ("diff1", y.diff().dropna()), ("diff7", y.diff(7).dropna()),
                     ("diff1_diff7", y.diff().diff(7).dropna())]:
    for regression in ("c", "ct"):
        for lag in (1, 4, 8, 12):
            try:
                res = adfuller(values, maxlag=lag, regression=regression, autolag=None)
                stat, pvalue, usedlag, nobs = res[:4]
                rows.append({"series": name, "regression": regression, "lag": lag,
                             "stat": float(stat), "p_value": float(pvalue), "nobs": int(nobs)})
            except Exception:  # noqa: BLE001
                pass
adf = pd.DataFrame(rows)
save_table(adf, "t05_adf_grid.csv")

piv = adf[adf["series"] == "level"].pivot(index="regression", columns="lag", values="p_value")
core = piv.loc[["c", "ct"]]
adf_max_p = float(core.to_numpy().max())
adf_all_reject = bool((core.to_numpy() < 0.05).all())
fig, ax = plt.subplots(figsize=(11, 2.8))
im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.2)
ax.set_yticks(range(len(piv.index)), ["no constant (mean is not 0)", "constant", "constant + trend"])
ax.set_xticks(range(len(piv.columns)), piv.columns)
ax.set_xlabel("lag length k")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        value = piv.iloc[i, j]
        ax.text(j, i, f"{value:.3f}" if value >= 0.001 else "<.001", ha="center", va="center", fontsize=7)
ax.set_title(f"F04 ADF p-values on the daily level: with a constant, every lag rejects (max p = {adf_max_p:.1e}); the no-constant row is misspecified")
fig.colorbar(im, ax=ax, label="p-value")
save_fig(fig, "F04_adf_grid.png")

# ------------------------------------------------------- KPSS and PP
stationarity_rows = []
for name, values in [("level", y), ("log_level", logy), ("diff1", y.diff().dropna()),
                     ("diff7", y.diff(7).dropna())]:
    for regression in ("c", "ct"):
        try:
            kp = kpss(values, regression=regression, nlags="auto")
            stationarity_rows.append({"series": name, "test": "KPSS", "regression": regression,
                                      "stat": float(kp[0]), "p_value": float(kp[1]), "note": "supplementary"})
        except Exception:  # noqa: BLE001
            pass
    try:
        pp = PhillipsPerron(values, trend="c", lags=14)
        stationarity_rows.append({"series": name, "test": "Phillips-Perron", "regression": "c",
                                  "stat": float(pp.stat), "p_value": float(pp.pvalue), "note": "supplementary"})
    except Exception:  # noqa: BLE001
        pass
stationarity = pd.DataFrame(stationarity_rows)
save_table(stationarity, "t06_kpss_pp.csv")

# ------------------------------------------------------------ ACF/PACF
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
plot_acf(y, lags=60, ax=axes[0], alpha=0.05)
axes[0].set_title(f"level ACF: r1 = {y.autocorr(1):.3f}, r7 = {y.autocorr(7):.3f}, r365 = {y.autocorr(365):.3f}")
plot_pacf(y, lags=60, ax=axes[1], alpha=0.05, method="ywm")
axes[1].set_title("level PACF")
fig.suptitle("F05 Sample ACF and PACF of the daily level (training window)", fontweight="bold", y=1.02)
save_fig(fig, "F05_acf_pacf_level.png")

month_mean = y.groupby(y.index.month).transform("mean")
adjusted = y - month_mean
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
plot_acf(adjusted, lags=60, ax=axes[0], alpha=0.05)
axes[0].set_title(f"month-adjusted ACF: r1 = {adjusted.autocorr(1):.3f}, r7 = {adjusted.autocorr(7):.3f}")
plot_pacf(adjusted, lags=60, ax=axes[1], alpha=0.05, method="ywm")
axes[1].set_title("month-adjusted PACF")
fig.suptitle("F06 After removing the calendar-month mean, the annual cycle is gone; short memory remains",
             fontweight="bold", y=1.02)
save_fig(fig, "F06_acf_pacf_adjusted.png")

acf_table = pd.DataFrame({
    "lag": [1, 2, 3, 7, 14, 21, 30, 60, 365],
    "acf_level": [y.autocorr(k) for k in [1, 2, 3, 7, 14, 21, 30, 60, 365]],
    "acf_month_adjusted": [adjusted.autocorr(k) for k in [1, 2, 3, 7, 14, 21, 30, 60, 365]],
})
save_table(acf_table, "t07_acf_selected_lags.csv")

# ------------------------------------------------------------ STL
stl_input = train.interpolate(limit_direction="both")
stl = STL(stl_input, period=365, robust=True).fit()
fig, axes = plt.subplots(4, 1, figsize=(12, 7.6), sharex=True)
axes[0].plot(stl_input.index, stl_input.to_numpy(), color="#333333", lw=0.7)
axes[0].set_ylabel("observed")
axes[1].plot(stl.trend.index, stl.trend.to_numpy(), color="#c44e52", lw=1.2)
axes[1].set_ylabel("trend")
axes[2].plot(stl.seasonal.index, stl.seasonal.to_numpy(), color="#4c72b0", lw=0.7)
axes[2].set_ylabel("annual")
axes[3].plot(stl.resid.index, stl.resid.to_numpy(), color="0.4", lw=0.5)
axes[3].set_ylabel("remainder")
fig.suptitle(f"F07 STL with a 365-day period: annual component range {stl.seasonal.min():.2f} to {stl.seasonal.max():.2f} AQHI",
             fontweight="bold", y=1.0)
fig.tight_layout()
save_fig(fig, "F07_stl_decomposition.png")

strength_seasonal = max(0.0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))
strength_trend = max(0.0, 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))

# ------------------------------------------ ACF comparison + variance by month
fig, ax = plt.subplots(figsize=(9.5, 4.4))
lags = np.arange(1, 31)
ax.plot(lags, [y.autocorr(int(k)) for k in lags], marker="o", ms=3, label="level", color="#333333")
ax.plot(lags, [adjusted.autocorr(int(k)) for k in lags], marker="s", ms=3, label="month-adjusted", color="#4c72b0")
ax.plot(lags, [y.diff().dropna().autocorr(int(k)) for k in lags], marker="^", ms=3, label="first difference", color="#c44e52")
ax.axhline(0, color="0.3", lw=0.8)
ax.axhline(1.96 / np.sqrt(len(y)), color="0.6", ls="--", lw=0.7)
ax.set_xlabel("lag (days)")
ax.set_ylabel("autocorrelation")
ax.set_title(f"F08 ACF under three treatments: the level's slow decay is the annual cycle, not a unit root")
ax.legend(frameon=False)
save_fig(fig, "F08_acf_comparison.png")

var_month = train.groupby(train.index.month).agg(["mean", "std", "count"]).reindex(range(1, 13))
save_table(var_month.reset_index().rename(columns={"index": "month"}), "t08_month_level_variance.csv")

# ------------------------------------------------------------- facts
r1 = float(y.autocorr(1))
neff = len(y) * (1 - r1) / (1 + r1)
save_facts({
    "adf_level_max_p": adf_max_p,
    "adf_level_min_p": float(core.to_numpy().min()),
    "adf_all_reject_5pct": adf_all_reject,
    "adf_no_constant_max_p": float(piv.loc["n"].max()),
    "kpss_level_p": float(stationarity.query("series == 'level' and test == 'KPSS' and regression == 'c'")["p_value"].iloc[0]),
    "n_obs": int(len(y)),
    "r1": r1,
    "r7": float(y.autocorr(7)),
    "r365": float(y.autocorr(365)),
    "r1_month_adjusted": float(adjusted.autocorr(1)),
    "effective_n": float(neff),
    "stl_seasonal_strength": float(strength_seasonal),
    "stl_trend_strength": float(strength_trend),
    "stl_annual_range": [float(stl.seasonal.min()), float(stl.seasonal.max())],
}, "stationarity.json")

print("01 stationarity done")
print(f"ADF level p range: {piv.to_numpy().min():.2e} to {piv.to_numpy().max():.2e}; all reject at 5%: {bool((piv.to_numpy() < 0.05).all())}")
kpss_level_p = float(stationarity.query("series == 'level' and test == 'KPSS' and regression == 'c'")["p_value"].iloc[0])
print(f"KPSS level p: {kpss_level_p:.3f}")
print(f"acf1 {r1:.3f}  acf7 {y.autocorr(7):.3f}  acf365 {y.autocorr(365):.3f}  effective n {neff:.1f}")
print(f"month-adjusted acf1 {adjusted.autocorr(1):.3f}")
print(f"STL seasonal strength {strength_seasonal:.3f}, trend strength {strength_trend:.3f}")
