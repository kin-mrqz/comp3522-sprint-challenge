"""06 - exponential smoothing benchmarks (Chapter 9), including Holt-Winters.

Figures F24-F25, tables t23-t24, facts ets.json.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tsa_lib import (ljung_box_adjusted, load_daily, save_facts, save_fig, save_table, score,
                     splits)

series = load_daily()
parts = splits(series)
train, val = parts["train"], parts["val"]
train_filled = train.interpolate(limit_direction="both")

configs = [
    ("SES", dict(trend=None, seasonal=None, damped_trend=False)),
    ("Holt_damped", dict(trend="add", seasonal=None, damped_trend=True)),
    ("HW_add_damped", dict(trend="add", seasonal="add", seasonal_periods=7, damped_trend=True)),
    ("HW_add", dict(trend="add", seasonal="add", seasonal_periods=7, damped_trend=False)),
    ("HW_mul_damped", dict(trend="add", seasonal="mul", seasonal_periods=7, damped_trend=True)),
    ("HW_add_annual365", dict(trend="add", seasonal="add", seasonal_periods=365, damped_trend=True)),
]

rows, results = [], {}
for name, cfg in configs:
    try:
        model = ExponentialSmoothing(train_filled, initialization_method="estimated", **cfg)
        res = model.fit(optimized=True, remove_bias=False)
        forecast = res.forecast(len(val))
        metrics = score(val, forecast, train=train)
        resid = pd.Series(res.resid).dropna()
        rec = {
            "name": name,
            "config": str(cfg),
            "aic": float(getattr(res, "aic", np.nan)),
            "bic": float(getattr(res, "bic", np.nan)),
            "aicc": float(getattr(res, "aicc", np.nan)),
            "sse": float(res.sse),
            "params": str(res.params),
            "status": "ok",
        }
        rec.update({f"val_{k}": v for k, v in metrics.items()})
        rec.update(ljung_box_adjusted(resid, m=0))
        results[name] = {"fit": res, "forecast": forecast}
    except Exception as exc:  # noqa: BLE001
        rec = {"name": name, "config": str(cfg), "status": f"failed: {type(exc).__name__}: {exc}"}
    rows.append(rec)
grid = pd.DataFrame(rows)
save_table(grid, "t23_ets_grid.csv")

ok = grid[grid["status"] == "ok"].sort_values("val_rmse")
best = ok.iloc[0]
best_name = str(best["name"])
forecast = results[best_name]["forecast"]

fig, ax = plt.subplots(figsize=(12.5, 4.4))
ax.plot(train.index[-180:], train.iloc[-180:].to_numpy(), color="0.5", lw=0.8, label="train (last 180 d)")
ax.plot(val.index, val.to_numpy(), color="#333333", lw=1.0, label="validation")
ax.plot(forecast.index, forecast.to_numpy(), color="#ff7f0e", lw=1.4, label=f"{best_name} forecast")
ax.set_ylabel("AQHI")
ax.set_title(f"F24 Best exponential smoothing ({best_name}): validation RMSE {best['val_rmse']:.3f}, MAE {best['val_mae']:.3f}")
ax.legend(frameon=False, fontsize=8.5)
save_fig(fig, "F24_ets_forecast.png")

fig, ax = plt.subplots(figsize=(9.2, 4.2))
plot = grid[grid["status"] == "ok"].sort_values("val_rmse")
ax.barh(range(len(plot)), plot["val_rmse"], color="#ff7f0e")
ax.set_yticks(range(len(plot)), plot["name"], fontsize=8.5)
ax.set_xlabel("validation RMSE")
ax.set_title(f"F25 Exponential smoothing grid: RMSE {plot['val_rmse'].min():.3f} to {plot['val_rmse'].max():.3f}")
for i, value in enumerate(plot["val_rmse"]):
    ax.text(value + 0.002, i, f"{value:.3f}", va="center", fontsize=7.5)
save_fig(fig, "F25_ets_rmse.png")

save_table(pd.DataFrame([{
    "name": best_name,
    "aic": float(best["aic"]),
    "bic": float(best["bic"]),
    "val_rmse": float(best["val_rmse"]),
    "val_mae": float(best["val_mae"]),
    "val_coverage": float(best.get("val_coverage_95", np.nan)),
    "params": str(best["params"]),
}]), "t24_ets_selected.csv")

save_facts({
    "best_name": best_name,
    "best_aic": float(best["aic"]),
    "best_bic": float(best["bic"]),
    "best_val_rmse": float(best["val_rmse"]),
    "best_val_mae": float(best["val_mae"]),
    "best_val_coverage": float(best.get("val_coverage_95", np.nan)),
    "best_lb_30": float(best["lb_p_30"]),
    "grid": ok[["name", "aic", "val_rmse", "val_mae"]].to_dict("records"),
}, "ets.json")

print("06 ETS done")
print(grid[["name", "status", "aic", "val_rmse", "val_mae", "lb_p_30"]].to_string(index=False))
