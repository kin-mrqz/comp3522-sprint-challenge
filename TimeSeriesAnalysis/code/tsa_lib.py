"""Shared helpers for the AQHI time-series analysis.

Run from the repository root with the STAT4601 venv:
    ~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/01_eda_stationarity.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*No frequency information.*")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")
warnings.filterwarnings("ignore", message=".*Non-invertible starting MA parameters.*")
warnings.filterwarnings("ignore", message=".*Non-stationary starting autoregressive parameters.*")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TSA = ROOT / "TimeSeriesAnalysis"
RES = TSA / "results"
FIG = RES / "figures"
TAB = RES / "tables"
FACTS = RES / "facts"
DATA = RES / "data"
REPORT = TSA / "report"
for _p in (FIG, TAB, FACTS, DATA, REPORT):
    _p.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2021-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-11-30"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"
FULL_END = "2024-11-30"
ORIGIN = pd.Timestamp("2021-01-01")
PERIOD = 365.25
SEASON_WEEK = 7

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_fig(fig, name: str) -> Path:
    path = FIG / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    path = TAB / name
    df.to_csv(path, index=index, float_format="%.6g")
    return path


def save_facts(payload: dict, name: str) -> Path:
    path = FACTS / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_daily() -> pd.Series:
    """Daily territory mean AQHI, one value per calendar day (NaN on gaps)."""
    frame = pd.read_parquet(PROCESSED / "aqhi_daily_territory.parquet")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    series = frame["mean_station_mean"].astype("float64")
    series = series.asfreq("D")
    series.name = "aqhi"
    return series


def splits(series: pd.Series) -> dict[str, pd.Series]:
    return {
        "train": series.loc[TRAIN_START:TRAIN_END],
        "val": series.loc[VAL_START:VAL_END],
        "test": series.loc[TEST_START:TEST_END],
        "full": series.loc[TRAIN_START:FULL_END],
    }


def exog_frame(index: pd.DatetimeIndex, kind: str = "fourier_dow", k: int = 2) -> pd.DataFrame:
    """Deterministic regressors. Same formula for train, validation and test.

    kind: 'fourier', 'dow', 'fourier_dow', 'month_dow', 'fourier_dow_trend'.
    """
    index = pd.DatetimeIndex(index)
    t = (index - ORIGIN).days.to_numpy(dtype=float)
    columns: dict[str, np.ndarray] = {}
    if kind.startswith("fourier") or kind in {"fourier_dow", "fourier_dow_trend"}:
        for kk in range(1, k + 1):
            columns[f"sin{kk}"] = np.sin(2 * np.pi * kk * t / PERIOD)
            columns[f"cos{kk}"] = np.cos(2 * np.pi * kk * t / PERIOD)
    if kind in {"month_dow", "month"}:
        for month in range(2, 13):
            columns[f"month{month}"] = (index.month == month).astype(float)
    if kind in {"dow", "fourier_dow", "month_dow", "fourier_dow_trend"}:
        for day in range(1, 7):
            columns[f"dow{day}"] = (index.dayofweek == day).astype(float)
    if kind == "fourier_dow_trend":
        columns["trend"] = t / 365.25
    return pd.DataFrame(columns, index=index)


def param_count(order, seasonal_order, n_exog: int = 0, trend: str = "c") -> int:
    p, d, q = order
    P, D, Q, s = seasonal_order
    return int(p + q + P + Q + n_exog + (1 if trend not in ("n", None) else 0) + 1)


def ljung_box_adjusted(resid: pd.Series, lags=(7, 14, 21, 30), m: int = 0) -> dict:
    """Ljung-Box p-values with the course convention df = K - m."""
    clean = pd.Series(resid).dropna()
    if len(clean) < max(lags) + 5:
        return {f"lb_p_{k}": np.nan for k in lags}
    table = acorr_ljungbox(clean, lags=list(lags), return_df=True)
    out = {}
    for k in lags:
        stat = float(table.loc[k, "lb_stat"])
        df = max(int(k - m), 1)
        out[f"lb_p_{k}"] = float(stats.chi2.sf(stat, df))
        out[f"lb_stat_{k}"] = stat
    return out


def near_unit(res, threshold: float = 0.98) -> tuple[bool, float, float]:
    """Flag AR/MA parameters pinned near the boundary and report the smallest root."""
    estimates = []
    for attr in ("arparams", "maparams", "seasonalarparams", "seasonalmaparams"):
        values = getattr(res, attr, np.array([]))
        estimates.extend(np.abs(np.asarray(values, dtype=float)).ravel().tolist())
    max_abs = max(estimates) if estimates else 0.0
    roots = []
    for attr in ("arroots", "maroots", "seasonalarroots", "seasonalmaroots"):
        values = getattr(res, attr, np.array([]))
        values = np.asarray(values, dtype=complex).ravel()
        if len(values):
            roots.append(float(np.min(np.abs(values))))
    min_root = min(roots) if roots else np.nan
    flag = bool(max_abs > threshold or (np.isfinite(min_root) and min_root < 1.02))
    return flag, max_abs, min_root


def fit_model(
    y: pd.Series,
    order=(1, 0, 0),
    seasonal_order=(0, 0, 0, 0),
    exog: pd.DataFrame | None = None,
    trend: str = "c",
    name: str = "model",
    family: str = "ARIMA",
    exog_kind: str = "",
    k_fourier: int = 0,
    maxiter: int = 300,
) -> dict:
    """Fit one SARIMAX model and return a record plus the results object."""
    y = pd.Series(y).astype("float64").asfreq("D")
    record = {
        "name": name,
        "family": family,
        "order": str(tuple(order)),
        "seasonal_order": str(tuple(seasonal_order)),
        "exog_kind": exog_kind,
        "k_fourier": k_fourier,
        "trend": trend,
        "nobs": int(y.notna().sum()),
        "status": "ok",
    }
    try:
        model = SARIMAX(
            y,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        result = model.fit(disp=False, maxiter=maxiter)
    except Exception as exc:  # noqa: BLE001 - the grid records failures rather than stopping
        record.update({"status": f"failed: {type(exc).__name__}: {exc}"})
        return {"record": record, "result": None}

    resid = pd.Series(result.resid, index=y.index).astype("float64")
    p, d, q = order
    P, D, Q, s = seasonal_order
    m = p + q + P + Q
    k_params = len(result.params)
    n = int(y.notna().sum())
    aicc = result.aic + (2 * k_params * (k_params + 1) / (n - k_params - 1)) if n > k_params + 1 else np.nan
    flag, max_abs, min_root = near_unit(result)
    arch_p = np.nan
    try:
        clean = resid.dropna()
        if len(clean) > 30:
            arch_p = float(het_arch(clean, nlags=10)[1])
    except Exception:  # noqa: BLE001
        pass
    jb_p = np.nan
    try:
        jb_p = float(stats.jarque_bera(resid.dropna())[1])
    except Exception:  # noqa: BLE001
        pass

    record.update({
        "aic": float(result.aic),
        "bic": float(result.bic),
        "aicc": float(aicc),
        "sigma2": float(result.params.get("sigma2", np.nan)),
        "n_params": int(k_params),
        "loglik": float(result.llf),
        "near_unit": flag,
        "max_abs_arma": max_abs,
        "min_root": min_root,
        "arch_lm_p_10": arch_p,
        "jb_p": jb_p,
        "converged": bool(getattr(result.mle_retvals, "get", lambda *_: True)("converged", True)),
    })
    record.update(ljung_box_adjusted(resid, m=m))
    return {"record": record, "result": result, "resid": resid}


def forecast_result(result, steps: int, exog: pd.DataFrame | None = None, alpha: float = 0.05):
    fc = result.get_forecast(steps=steps, exog=exog)
    mean = pd.Series(fc.predicted_mean).astype("float64")
    ci = pd.DataFrame(fc.conf_int(alpha=alpha))
    ci.columns = ["lower", "upper"]
    return mean, ci


def score(y_true: pd.Series, y_pred: pd.Series, lower=None, upper=None, train=None) -> dict:
    frame = pd.concat([pd.Series(y_true).rename("y"), pd.Series(y_pred).rename("f")], axis=1).dropna()
    if frame.empty:
        return {}
    err = frame["y"] - frame["f"]
    out = {
        "n": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(err.mean()),
        "max_abs": float(np.abs(err).max()),
    }
    if train is not None and len(train.dropna()) > SEASON_WEEK:
        denominator = float(np.mean(np.abs(pd.Series(train).dropna().diff(SEASON_WEEK).dropna())))
        out["mase"] = float(out["mae"] / denominator) if denominator > 0 else np.nan
    if lower is not None and upper is not None:
        lo = pd.Series(lower).reindex(frame.index)
        hi = pd.Series(upper).reindex(frame.index)
        inside = ((frame["y"] >= lo) & (frame["y"] <= hi)).dropna()
        out["coverage_95"] = float(inside.mean()) if len(inside) else np.nan
        out["width_95"] = float((hi - lo).dropna().mean())
        # pinball loss for the central 95% interval, averaged over the two quantiles
        yv = frame["y"].to_numpy(dtype=float)
        lov = lo.to_numpy(dtype=float)
        hiv = hi.to_numpy(dtype=float)
        loss_lo = np.where(yv >= lov, 0.025 * (yv - lov), 0.975 * (lov - yv))
        loss_hi = np.where(yv >= hiv, 0.975 * (yv - hiv), 0.025 * (hiv - yv))
        out["pinball_95"] = float(0.5 * (np.mean(loss_lo) + np.mean(loss_hi)))
    return out


def dm_test(err_a: pd.Series, err_b: pd.Series, maxlags: int = 7) -> dict:
    """Diebold-Mariano on squared-error loss, HAC variance."""
    frame = pd.concat([pd.Series(err_a).rename("a"), pd.Series(err_b).rename("b")], axis=1).dropna()
    if len(frame) < 20:
        return {"dm_stat": np.nan, "dm_p": np.nan, "n": int(len(frame))}
    d = frame["a"] ** 2 - frame["b"] ** 2
    x = np.ones((len(d), 1))
    import statsmodels.api as sm
    fit = sm.OLS(d.to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"dm_stat": float(fit.tvalues[0]), "dm_p": float(fit.pvalues[0]), "n": int(len(d))}
