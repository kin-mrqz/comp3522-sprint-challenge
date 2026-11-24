"""Advanced time-series analysis of the AQHI series.

Builds on run_analysis.py (same loading). Produces STL decomposition, ACF/PACF,
stationarity tests (ADF/KPSS), Mann-Kendall trend + Sen's slope, PELT change-point
detection, spectral analysis, and a seasonal forecasting backtest.

Run from the repo root inside .venv:
    .venv\\Scripts\\python.exe scripts\\advanced_timeseries.py
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

from src.paths import FIGURES, TABLES, OUTPUTS, ensure_dirs  # noqa: E402
from src.io import save_figure  # noqa: E402
from scripts.run_analysis import load_aqhi, territory_daily, RAW  # noqa: E402

RESULTS_PATH = OUTPUTS / "results.json"


def monthly_series(long: pd.DataFrame) -> pd.Series:
    d = long[long["timestamp"].dt.year <= 2025]
    s = (
        d.assign(month=d["timestamp"].dt.to_period("M"))
        .groupby("month")["aqhi"].mean()
    )
    s.index = s.index.to_timestamp()
    return s.sort_index()


def daily_series(long: pd.DataFrame) -> pd.Series:
    daily = territory_daily(long)
    return daily.set_index("date")["aqhi"].sort_index()


def mann_kendall(x: np.ndarray) -> tuple[float, float, float]:
    """Mann-Kendall trend test + Sen's slope. Returns (tau, two-sided p, slope/period)."""
    n = len(x)
    s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(x[i + 1:] - x[i]))
    # variance with tie correction
    unique, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5)) / 18.0
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var -= tie / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var)
    elif s < 0:
        z = (s + 1) / np.sqrt(var)
    else:
        z = 0.0
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    # Sen's slope
    slopes = [
        (x[j] - x[i]) / (j - i)
        for i in range(n - 1)
        for j in range(i + 1, n)
    ]
    slope = float(np.median(slopes)) if slopes else float("nan")
    tau = s / (0.5 * n * (n - 1))
    return float(tau), float(p), slope


def stl_decompose(monthly: pd.Series, results: dict) -> None:
    from statsmodels.tsa.seasonal import STL

    stl = STL(monthly, period=12, robust=True).fit()
    comp = pd.DataFrame(
        {
            "trend": stl.trend,
            "seasonal": stl.seasonal,
            "residual": stl.resid,
        }
    )
    comp.index.name = "month"
    comp.to_csv(TABLES / "ats_stl_components.csv")

    fig, axes = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
    axes[0].plot(monthly.index, monthly.values, color="#0B2545")
    axes[0].set_ylabel("Observed")
    axes[1].plot(comp.index, comp["trend"], color="#EE6C4D")
    axes[1].set_ylabel("Trend")
    axes[2].plot(comp.index, comp["seasonal"], color="#1B998B")
    axes[2].set_ylabel("Seasonal")
    axes[3].plot(comp.index, comp["residual"], color="#888888")
    axes[3].set_ylabel("Residual")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("STL decomposition of monthly territory AQHI (period=12)")
    fig.tight_layout()
    save_figure(fig, "ats_stl_monthly.png")

    results["stl"] = {
        "resid_std": round(float(np.nanstd(stl.resid)), 3),
        "seasonal_amplitude": round(float(stl.seasonal.max() - stl.seasonal.min()), 3),
        "trend_start": round(float(np.nanmean(stl.trend[:6])), 3),
        "trend_end": round(float(np.nanmean(stl.trend[-6:])), 3),
    }


def stationarity_tests(series: pd.Series, name: str, results: dict) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss

    x = series.dropna().values
    adf = adfuller(x, autolag="AIC", result_object=False)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kp = kpss(x, regression="c", nlags="auto")
    out = {
        "adf_stat": round(float(adf[0]), 3),
        "adf_p": round(float(adf[1]), 4),
        "adf_stationary": bool(adf[1] < 0.05),
        "kpss_stat": round(float(kp[0]), 3),
        "kpss_p": round(float(kp[1]), 4),
        "kpss_stationary": bool(kp[1] >= 0.05),
    }
    results.setdefault("stationarity", {})[name] = out
    return out


def acf_pacf(series: pd.Series, name: str, nlags: int) -> None:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    x = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    plot_acf(x, lags=nlags, ax=axes[0])
    plot_pacf(x, lags=nlags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF ({name})")
    axes[1].set_title(f"PACF ({name})")
    fig.tight_layout()
    save_figure(fig, f"ats_acf_{name}.png")


def seasonal_subseries(long: pd.DataFrame) -> None:
    d = long[long["timestamp"].dt.year <= 2025].copy()
    d["month"] = d["timestamp"].dt.month
    d["year"] = d["timestamp"].dt.year
    piv = d.pivot_table(index="year", columns="month", values="aqhi", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for year in piv.index:
        ax.plot(piv.columns, piv.loc[year], marker="o", ms=4, label=str(year))
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean AQHI")
    ax.set_title("Seasonal subseries: mean AQHI by month and year")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.3)
    save_figure(fig, "ats_seasonal_subseries.png")


def trend_tests(monthly: pd.Series, daily: pd.Series, results: dict) -> None:
    mk_m = mann_kendall(monthly.values)
    mk_d = mann_kendall(daily.dropna().values)
    # OLS trend on monthly for comparison
    from scipy import stats
    xs = np.arange(len(monthly))
    ols = stats.linregress(xs, monthly.values)
    pd.DataFrame(
        [
            {"series": "monthly", "mk_tau": mk_m[0], "mk_p": mk_m[1],
             "sens_slope_per_month": mk_m[2], "n": len(monthly)},
            {"series": "daily", "mk_tau": mk_d[0], "mk_p": mk_d[1],
             "sens_slope_per_day": mk_d[2], "n": int(daily.notna().sum())},
        ]
    ).to_csv(TABLES / "ats_mk_trend.csv", index=False)
    results["mann_kendall"] = {
        "monthly": {"tau": round(mk_m[0], 3), "p": round(mk_m[1], 4),
                    "sens_slope_per_month": round(mk_m[2], 5),
                    "ols_slope_per_month": round(float(ols.slope), 5),
                    "ols_p": round(float(ols.pvalue), 4)},
        "daily": {"tau": round(mk_d[0], 3), "p": round(mk_d[1], 4),
                  "sens_slope_per_day": round(mk_d[2], 6)},
        "trend_significant_5pct": bool(min(mk_m[1], mk_d[1]) < 0.05),
    }


def changepoints(monthly: pd.Series, results: dict) -> None:
    import ruptures as rpt
    from statsmodels.tsa.seasonal import STL

    # Run change-point detection on the STL trend (deseasonalised) so seasonal
    # swings are not mistaken for structural breaks.
    trend = STL(monthly, period=12, robust=True).fit().trend.dropna()
    signal = trend.values.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf", min_size=6).fit(signal)
    bkps = algo.predict(pen=1.0)
    # convert back to timestamps (last bkp is the end)
    idx = trend.index
    times = [idx[min(b - 1, len(idx) - 1)] for b in bkps[:-1]]
    pd.DataFrame({"break_month": [str(t.date()) for t in times]}).to_csv(
        TABLES / "ats_changepoints.csv", index=False
    )
    results["changepoints"] = {
        "n_breaks": len(times),
        "break_months": [str(t.date()) for t in times],
    }

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.index, monthly.values, color="#bbbbbb", label="Monthly mean")
    ax.plot(idx, trend.values, color="#0B2545", label="STL trend")
    for t in times:
        ax.axvline(pd.Timestamp(t), color="#EE6C4D", ls="--", lw=1.2)
    ax.set_title("PELT change-point detection on STL trend (pen=1)")
    ax.set_ylabel("Mean AQHI")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_figure(fig, "ats_changepoint.png")


def periodogram(daily: pd.Series, results: dict) -> None:
    x = daily.dropna()
    x = x - x.mean()
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power = np.abs(np.fft.rfft(x)) ** 2
    # dominant non-zero frequency (in cycles/day); annual ~ 1/365.25
    mask = freqs > 0
    dom = freqs[mask][np.argmax(power[mask])]
    period = 1.0 / dom if dom > 0 else np.inf
    results["periodogram"] = {
        "dominant_period_days": round(float(period), 1),
        "dominant_freq_cycles_per_day": round(float(dom), 6),
        "annual_freq_cycles_per_day": round(1 / 365.25, 6),
    }
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(freqs[mask], power[mask], color="#1B998B", lw=0.8)
    ax.axvline(1 / 365.25, color="#EE6C4D", ls="--", label="1/year")
    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel("Power")
    ax.set_title("Periodogram of daily AQHI")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "ats_periodogram.png")


def forecast(monthly: pd.Series, results: dict) -> None:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    train = monthly[monthly.index.year <= 2023]
    test = monthly[monthly.index.year >= 2024]

    # Seasonal naive baseline: repeat value from 12 months earlier.
    hist = monthly.copy()
    naive_pred = []
    for t in test.index:
        prev = t - pd.DateOffset(months=12)
        naive_pred.append(hist.get(prev, np.nan))
    naive_pred = pd.Series(naive_pred, index=test.index)

    # ETS (Holt-Winter) with additive yearly seasonality.
    ets = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=12
    ).fit()
    ets_pred = ets.forecast(len(test))

    def metrics(y, p):
        mask = ~np.isnan(p)
        y, p = y[mask], p[mask]
        return {
            "mae": round(float(mean_absolute_error(y, p)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 3),
            "n": int(len(y)),
        }

    res = {
        "train_range": [str(train.index.min().date()), str(train.index.max().date())],
        "test_range": [str(test.index.min().date()), str(test.index.max().date())],
        "seasonal_naive": metrics(test.values, naive_pred.values),
        "ets_holt_winters": metrics(test.values, ets_pred.values),
        "test_mean": round(float(test.mean()), 3),
        "test_std": round(float(test.std()), 3),
    }
    pd.DataFrame(
        [
            {"model": "seasonal_naive", **res["seasonal_naive"]},
            {"model": "ets_holt_winters", **res["ets_holt_winters"]},
        ]
    ).to_csv(TABLES / "ats_forecast_metrics.csv", index=False)
    results["forecast"] = res

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(monthly.index, monthly.values, color="#0B2545", label="Observed")
    ax.plot(ets_pred.index, ets_pred.values, color="#EE6C4D", label="ETS forecast")
    ax.plot(naive_pred.index, naive_pred.values, color="#1B998B", ls="--",
            label="Seasonal naive")
    ax.axvspan(train.index.max(), test.index.max(), color="#cccccc", alpha=0.3)
    ax.set_title("Monthly AQHI forecast backtest (train<=2023, test 2024-2025)")
    ax.set_ylabel("Mean AQHI")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_figure(fig, "ats_forecast.png")


def main() -> int:
    ensure_dirs()
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    adv: dict = {}

    long = load_aqhi()
    monthly = monthly_series(long)
    daily = daily_series(long)

    print("monthly points:", len(monthly), "daily points:", len(daily))
    stl_decompose(monthly, adv)
    stationarity_tests(daily, "daily_level", adv)
    stationarity_tests(monthly, "monthly_level", adv)
    acf_pacf(monthly, "monthly", nlags=24)
    acf_pacf(daily, "daily", nlags=60)
    seasonal_subseries(long)
    trend_tests(monthly, daily, adv)
    changepoints(monthly, adv)
    periodogram(daily, adv)
    forecast(monthly, adv)

    results["advanced_ts"] = adv
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(adv, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())