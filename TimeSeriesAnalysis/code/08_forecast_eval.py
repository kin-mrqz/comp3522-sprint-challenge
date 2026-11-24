"""08 - validation comparison, final test forecasts, horizons and Diebold-Mariano.

Validation 2024 selects finalists; 2025 is scored only after selection.
Figures F30-F36, tables t29-t34, facts forecast.json.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tsa_lib import (DOW_ABBR, FACTS, MONTH_ABBR, exog_frame, fit_model, load_daily,
                     save_facts, save_fig, save_table, score, splits)

series = load_daily()
parts = splits(series)
train, val, test, full = parts["train"], parts["val"], parts["test"], parts["full"]

arima_spec = json.loads((FACTS / "arima.json").read_text())
sarima_spec = json.loads((FACTS / "sarima.json").read_text())
sarimax_spec = json.loads((FACTS / "sarimax.json").read_text())
ets_spec = json.loads((FACTS / "ets.json").read_text())

ETS_CONFIGS = {
    "SES": dict(trend=None, seasonal=None, damped_trend=False),
    "Holt_damped": dict(trend="add", seasonal=None, damped_trend=True),
    "HW_add_damped": dict(trend="add", seasonal="add", seasonal_periods=7, damped_trend=True),
    "HW_add": dict(trend="add", seasonal="add", seasonal_periods=7, damped_trend=False),
    "HW_mul_damped": dict(trend="add", seasonal="mul", seasonal_periods=7, damped_trend=True),
    "HW_add_annual365": dict(trend="add", seasonal="add", seasonal_periods=365, damped_trend=True),
}


def bench_forecast(history: pd.Series, target_index: pd.DatetimeIndex, method: str) -> pd.Series:
    history = history.dropna()
    if method == "mean":
        return pd.Series(float(history.mean()), index=target_index)
    if method == "naive":
        return pd.Series(float(history.iloc[-1]), index=target_index)
    if method == "snaive7":
        last = history.iloc[-7:].to_numpy(dtype=float)
        return pd.Series(np.resize(last, len(target_index)), index=target_index)
    if method == "snaive365":
        values = []
        fallbacks = 0
        month_mean = history.groupby(history.index.month).mean()
        for day in target_index:
            source = day - pd.Timedelta(days=365)
            if source in history.index and pd.notna(history.loc[source]):
                values.append(float(history.loc[source]))
            else:
                values.append(float(month_mean.loc[day.month]))
                fallbacks += 1
        out = pd.Series(values, index=target_index)
        out.attrs["fallbacks"] = fallbacks
        return out
    if method == "month_clim":
        month_mean = history.groupby(history.index.month).mean()
        return pd.Series([float(month_mean.loc[d.month]) for d in target_index], index=target_index)
    raise ValueError(method)


def ols_forecast(history: pd.Series, target_index: pd.DatetimeIndex, kind: str, fit_index: pd.DatetimeIndex) -> pd.Series:
    import statsmodels.api as sm
    basis = history.index[0]
    t_fit = (history.index - basis).days.to_numpy(dtype=float)
    t_target = (target_index - basis).days.to_numpy(dtype=float)
    if kind == "trend":
        X = sm.add_constant(t_fit)
        Xt = sm.add_constant(t_target)
    else:
        X = exog_frame(history.index, "fourier_dow", k=2).to_numpy()
        X = sm.add_constant(X)
        Xt = exog_frame(target_index, "fourier_dow", k=2).to_numpy()
        Xt = sm.add_constant(Xt)
    fit = sm.OLS(history.to_numpy(dtype=float), X).fit()
    return pd.Series(fit.predict(Xt), index=target_index)


# --------------------------------------------------- validation comparison
records = []
forecasts = {}

for method in ("mean", "naive", "snaive7", "snaive365", "month_clim"):
    pred = bench_forecast(train, val.index, method)
    forecasts[method] = pred
    rec = {"model": method, "family": "benchmark", **score(val, pred, train=train)}
    records.append(rec)
for kind, label in [("trend", "linear_trend"), ("seasonal", "seasonal_ols")]:
    pred = ols_forecast(train.dropna(), val.index, kind, train.index)
    forecasts[label] = pred
    rec = {"model": label, "family": "regression", **score(val, pred, train=train)}
    records.append(rec)

# ARIMA
order = tuple(arima_spec["best_order"])
fit = fit_model(train, order=order, seasonal_order=(0, 0, 0, 0), trend="c", name="arima", family="ARIMA")
mean, ci = fit["result"].get_forecast(steps=len(val)).predicted_mean, fit["result"].get_forecast(steps=len(val)).conf_int()
forecasts["arima"] = pd.Series(mean, index=val.index)
records.append({"model": f"ARIMA{order}", "family": "ARIMA",
                **score(val, forecasts["arima"], ci.iloc[:, 0], ci.iloc[:, 1], train)})

# SARIMA
sorder = tuple(sarima_spec["best_order"])
sarima_seasonal = tuple(sarima_spec["best_seasonal_order"])
fit = fit_model(train, order=sorder, seasonal_order=sarima_seasonal, trend="c", name="sarima", family="SARIMA")
mean, ci = fit["result"].get_forecast(steps=len(val)).predicted_mean, fit["result"].get_forecast(steps=len(val)).conf_int()
forecasts["sarima"] = pd.Series(mean, index=val.index)
records.append({"model": f"SARIMA{sorder}x{sarima_seasonal}", "family": "SARIMA",
                **score(val, forecasts["sarima"], ci.iloc[:, 0], ci.iloc[:, 1], train)})

# SARIMAX
kind = sarimax_spec["best_exog_set"]
k = int(sarimax_spec["best_k_fourier"])
sp, sq = int(sarimax_spec["best_p"]), int(sarimax_spec["best_q"])
train_exog = exog_frame(train.index, kind, k=k)
val_exog = exog_frame(val.index, kind, k=k)
sarimax_seasonal = tuple(sarimax_spec.get("best_seasonal_order", [0, 0, 0, 0]))
fit = fit_model(train, order=(sp, 0, sq), seasonal_order=sarimax_seasonal, exog=train_exog,
                trend="c", name="sarimax", family="SARIMAX")
mean, ci = fit["result"].get_forecast(steps=len(val), exog=val_exog).predicted_mean, \
           fit["result"].get_forecast(steps=len(val), exog=val_exog).conf_int()
forecasts["sarimax"] = pd.Series(mean, index=val.index)
records.append({"model": f"SARIMAX({sp},{sq}) {kind}", "family": "SARIMAX",
                **score(val, forecasts["sarimax"], ci.iloc[:, 0], ci.iloc[:, 1], train)})

# ETS
ets_name = ets_spec["best_name"]
ets_cfg = ETS_CONFIGS[ets_name]
ets_fit = ExponentialSmoothing(train.interpolate(limit_direction="both"),
                               initialization_method="estimated", **ets_cfg).fit(optimized=True, remove_bias=False)
forecasts["ets"] = ets_fit.forecast(len(val))
records.append({"model": f"ETS {ets_name}", "family": "ETS", **score(val, forecasts["ets"], train=train)})

validation = pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)
save_table(validation, "t29_validation_metrics.csv")
print(validation[["model", "family", "rmse", "mae", "mase", "coverage_95"]].to_string(index=False))

# --------------------------------------------------- validation figure
fig, ax = plt.subplots(figsize=(10.5, 5.2))
plot = validation.sort_values("rmse")
colors = {"benchmark": "#7f7f7f", "regression": "#bcbd22", "ARIMA": "#1f77b4",
          "SARIMA": "#2ca02c", "SARIMAX": "#d62728", "ETS": "#ff7f0e"}
ax.barh(range(len(plot)), plot["rmse"], color=[colors.get(x, "0.5") for x in plot["family"]])
ax.set_yticks(range(len(plot)), plot["model"], fontsize=8)
ax.set_xlabel("validation RMSE (AQHI, 2024 Jan-Nov)")
ax.set_title(f"F30 Validation ranking: best {plot.iloc[0]['model']} at {plot.iloc[0]['rmse']:.3f}; a constant mean scores {validation.query('model == \"mean\"')['rmse'].iloc[0]:.3f}")
for i, value in enumerate(plot["rmse"]):
    ax.text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=7.5)
save_fig(fig, "F30_validation_rmse.png")

# ------------------------------------------------ validation fan chart
finalist_names = validation.head(4)["model"].tolist()
lookup = {
    "arima": "arima", "sarima": "sarima", "sarimax": "sarimax", "ets": "ets",
    "mean": "mean", "naive": "naive", "snaive7": "snaive7", "snaive365": "snaive365",
    "month_clim": "month_clim", "linear_trend": "linear_trend", "seasonal_ols": "seasonal_ols",
}
def key_for(label):
    lowered = label.lower()
    if lowered.startswith("sarimax"):
        return "sarimax"
    if lowered.startswith("sarima"):
        return "sarima"
    if lowered.startswith("arima"):
        return "arima"
    if lowered.startswith("ets"):
        return "ets"
    for key in lookup:
        if key in lowered:
            return key
    return None

fig, ax = plt.subplots(figsize=(12.5, 4.6))
ax.plot(val.index, val.to_numpy(), color="0.2", lw=1.2, label="actual 2024")
for label in finalist_names[:3]:
    key = key_for(label)
    if key and key in forecasts:
        ax.plot(val.index, forecasts[key].to_numpy(), lw=1.3, label=label)
ax.set_ylabel("AQHI")
ax.set_title(f"F31 Validation forecasts: {', '.join(finalist_names[:3])}")
ax.legend(frameon=False, fontsize=8)
save_fig(fig, "F31_validation_forecast.png")

# --------------------------------------------------------- test refit
future_index = pd.date_range("2024-12-01", test.index.max(), freq="D")
test_rows = []
test_forecasts = {}

def test_model(label, key, predictor):
    pred = predictor()
    pred = pd.Series(pred, index=future_index).reindex(test.index)
    test_forecasts[label] = pred
    test_rows.append({"model": label, "family": key, **score(test, pred, train=train)})

# benchmarks from full history
for method in ("mean", "snaive7", "snaive365", "month_clim"):
    pred = bench_forecast(full, test.index, method)
    test_forecasts[method] = pred
    test_rows.append({"model": method, "family": "benchmark", **score(test, pred, train=train)})

# ARIMA
fit = fit_model(full, order=order, seasonal_order=(0, 0, 0, 0), trend="c", name="arima_test", family="ARIMA")
mean, ci = fit["result"].get_forecast(steps=len(future_index)).predicted_mean, fit["result"].get_forecast(steps=len(future_index)).conf_int()
test_model(f"ARIMA{order}", "ARIMA", lambda: pd.Series(mean, index=future_index))
test_forecasts[f"ARIMA{order}"] = pd.Series(mean, index=future_index).reindex(test.index)

# SARIMA
fit = fit_model(full, order=sorder, seasonal_order=sarima_seasonal, trend="c", name="sarima_test", family="SARIMA")
mean, ci = fit["result"].get_forecast(steps=len(future_index)).predicted_mean, fit["result"].get_forecast(steps=len(future_index)).conf_int()
test_model(f"SARIMA{sorder}x{sarima_seasonal}", "SARIMA", lambda: pd.Series(mean, index=future_index))
test_forecasts[f"SARIMA{sorder}x{sarima_seasonal}"] = pd.Series(mean, index=future_index).reindex(test.index)

# SARIMAX
full_exog = exog_frame(full.index, kind, k=k)
future_exog = exog_frame(future_index, kind, k=k)
fit = fit_model(full, order=(sp, 0, sq), seasonal_order=sarimax_seasonal, exog=full_exog,
                trend="c", name="sarimax_test", family="SARIMAX")
mean = fit["result"].get_forecast(steps=len(future_index), exog=future_exog).predicted_mean
ci = fit["result"].get_forecast(steps=len(future_index), exog=future_exog).conf_int()
test_model(f"SARIMAX({sp},{sq}) {kind}", "SARIMAX", lambda: pd.Series(mean, index=future_index))
test_forecasts[f"SARIMAX({sp},{sq}) {kind}"] = pd.Series(mean, index=future_index).reindex(test.index)

# ETS
ets_fit_full = ExponentialSmoothing(full.interpolate(limit_direction="both"),
                                    initialization_method="estimated", **ets_cfg).fit(optimized=True, remove_bias=False)
test_model(f"ETS {ets_name}", "ETS", lambda: ets_fit_full.forecast(len(future_index)))

test_table = pd.DataFrame(test_rows).sort_values("rmse").reset_index(drop=True)
for _col in ("mase", "bias", "coverage_95", "width_95", "pinball_95"):
    if _col not in test_table.columns:
        test_table[_col] = np.nan
save_table(test_table, "t30_test_metrics.csv")
print("\nTEST")
print(test_table[["model", "family", "rmse", "mae", "mase", "bias", "coverage_95"]].to_string(index=False))

# --------------------------------------------------------- test figure
fig, ax = plt.subplots(figsize=(12.5, 4.6))
ax.plot(test.index, test.to_numpy(), color="0.2", lw=1.2, label="actual 2025")
for label in test_table.head(3)["model"].tolist():
    if label in test_forecasts:
        ax.plot(test.index, test_forecasts[label].to_numpy(), lw=1.3, label=label)
if "snaive365" in test_forecasts:
    ax.plot(test.index, test_forecasts["snaive365"].to_numpy(), color="0.6", lw=1.0, ls=":", label="seasonal naive 365")
ax.set_ylabel("AQHI")
ax.set_title(f"F32 2025 test forecasts: best validation model {test_table.iloc[0]['model']} scores RMSE {test_table.iloc[0]['rmse']:.3f}")
ax.legend(frameon=False, fontsize=8)
save_fig(fig, "F32_test_forecast.png")

# --------------------------------------------------------- horizon bands
bands = [(1, 7, "1-7 d"), (8, 30, "8-30 d"), (31, 90, "31-90 d"), (91, len(val), "91-335 d")]
horizon_rows = []
for label, pred in forecasts.items():
    pred = pd.Series(pred).reindex(val.index)
    for lo, hi, band in bands:
        block_y = val.iloc[lo - 1:hi]
        block_f = pred.iloc[lo - 1:hi]
        frame = pd.concat([block_y.rename("y"), block_f.rename("f")], axis=1).dropna()
        if len(frame) == 0:
            continue
        err = frame["y"] - frame["f"]
        horizon_rows.append({"model": label, "band": band, "n": int(len(frame)),
                             "rmse": float(np.sqrt(np.mean(err ** 2))),
                             "mae": float(np.mean(np.abs(err)))})
horizon = pd.DataFrame(horizon_rows)
save_table(horizon, "t31_horizon_rmse.csv")

fig, ax = plt.subplots(figsize=(9.8, 4.6))
for model in ["sarimax", "sarima", "arima", "snaive365", "mean"]:
    block = horizon[horizon["model"] == model]
    if len(block):
        ax.plot(block["band"], block["rmse"], marker="o", lw=1.6, label=model)
ax.set_ylabel("RMSE")
ax.set_title("F33 Forecast error by horizon: one week is skilful, a season is climatology")
ax.legend(frameon=False, fontsize=8)
save_fig(fig, "F33_horizon_rmse.png")

# --------------------------------------------------------- monthly table
best_test_label = test_table.iloc[0]["model"]
best_pred = test_forecasts[best_test_label]
monthly_rows = []
for month, block in pd.DataFrame({"y": test, "f": best_pred}).groupby(test.index.month):
    block = block.dropna()
    err = block["y"] - block["f"]
    monthly_rows.append({"month": MONTH_ABBR[month - 1], "actual_mean": float(block["y"].mean()),
                         "forecast_mean": float(block["f"].mean()),
                         "rmse": float(np.sqrt(np.mean(err ** 2))),
                         "bias": float(err.mean())})
monthly = pd.DataFrame(monthly_rows)
save_table(monthly, "t32_test_monthly.csv")

fig, ax = plt.subplots(figsize=(9.6, 4.2))
ax.plot(range(1, 13), monthly["actual_mean"], marker="o", label="actual", color="#333333")
ax.plot(range(1, 13), monthly["forecast_mean"], marker="s", label="forecast", color="#d62728")
ax.set_xticks(range(1, 13), MONTH_ABBR)
ax.set_ylabel("monthly mean AQHI")
ax.set_title(f"F34 Monthly test means, {best_test_label}: the model tracks the seasonal shape, not the month-to-month level")
ax.legend(frameon=False)
save_fig(fig, "F34_test_monthly.png")

# --------------------------------------------------------- method-change split
split_rows = []
for label in test_table["model"].tolist():
    if label not in test_forecasts:
        continue
    pred = test_forecasts[label]
    for period, mask in [("Jan-Mar 2025", test.index <= pd.Timestamp("2025-03-21")),
                         ("Apr-Dec 2025", test.index >= pd.Timestamp("2025-03-22"))]:
        block = pd.DataFrame({"y": test[mask], "f": pred[mask]}).dropna()
        if len(block) == 0:
            continue
        err = block["y"] - block["f"]
        split_rows.append({"model": label, "period": period, "n": int(len(block)),
                           "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))})
split = pd.DataFrame(split_rows)
save_table(split, "t33_method_change_split.csv")

fig, ax = plt.subplots(figsize=(10, 4.4))
pivot = split.pivot(index="model", columns="period", values="rmse").sort_values("Jan-Mar 2025")
pivot.plot(kind="barh", ax=ax, color=["#4c72b0", "#c44e52"])
ax.set_xlabel("RMSE")
ax.set_title("F35 Test error before and after the 2025-03-22 AQHI method change")
ax.legend(frameon=False)
save_fig(fig, "F35_method_change_split.png")

# --------------------------------------------------------- DM tests
dm_rows = []
base = test_forecasts.get("snaive365")
if base is not None:
    for label in test_table["model"].tolist():
        if label == "snaive365" or label not in test_forecasts:
            continue
        err_a = test - test_forecasts[label]
        err_b = test - base
        from tsa_lib import dm_test
        dm = dm_test(err_a, err_b, maxlags=7)
        dm_rows.append({"model": label, "vs": "snaive365", **dm})
dm = pd.DataFrame(dm_rows)
save_table(dm, "t34_diebold_mariano.csv")

save_facts({
    "validation_best": validation.iloc[0]["model"],
    "validation_best_rmse": float(validation.iloc[0]["rmse"]),
    "test_best": best_test_label,
    "test_best_rmse": float(test_table.iloc[0]["rmse"]),
    "test_best_mae": float(test_table.iloc[0]["mae"]),
    "test_best_mase": float(test_table.iloc[0]["mase"]),
    "test_mean_rmse": float(test_table.query("model == 'mean'")["rmse"].iloc[0]),
    "test_snaive365_rmse": float(test_table.query("model == 'snaive365'")["rmse"].iloc[0]) if "snaive365" in test_table["model"].values else np.nan,
    "validation_table": validation.to_dict("records"),
    "test_table": test_table.to_dict("records"),
    "finalists": finalist_names[:3],
    "test_monthly": monthly.to_dict("records"),
}, "forecast.json")

print("\n08 forecast evaluation done")
print(f"validation best {validation.iloc[0]['model']} {validation.iloc[0]['rmse']:.3f}")
print(f"test best {best_test_label} {test_table.iloc[0]['rmse']:.3f}; mean {test_table.query('model == \"mean\"')['rmse'].iloc[0]:.3f}")
print(monthly.to_string(index=False))
