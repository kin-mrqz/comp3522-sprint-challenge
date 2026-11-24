"""03 - ARIMA(p,d,q) grid, d = 0 and d = 1, with validation scoring.

Figures F14-F16, tables t13-t15, facts arima.json.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from tsa_lib import (fit_model, forecast_result, load_daily, save_facts, save_fig, save_table,
                     score, splits)

series = load_daily()
parts = splits(series)
train, val = parts["train"], parts["val"]

rows, results = [], {}
for d in (0, 1):
    for p in range(4):
        for q in range(4):
            name = f"arima_{p}{d}{q}"
            fit = fit_model(train, order=(p, d, q), seasonal_order=(0, 0, 0, 0), trend="c",
                            name=name, family="ARIMA", maxiter=300)
            rec = fit["record"]
            rec["d"] = d
            rec["p"] = p
            rec["q"] = q
            if fit["result"] is not None:
                try:
                    mean, ci = forecast_result(fit["result"], len(val))
                    rec.update({f"val_{k}": v for k, v in score(val, mean, ci["lower"], ci["upper"], train).items()})
                except Exception as exc:  # noqa: BLE001
                    rec["val_error"] = f"{type(exc).__name__}: {exc}"
                results[name] = fit
            rows.append(rec)
grid = pd.DataFrame(rows)
save_table(grid, "t13_arima_grid.csv")

d0 = grid[grid["d"] == 0].copy()
d1 = grid[grid["d"] == 1].copy()

# No plain ARIMA is expected to whiten a daily series with an annual cycle.
# Carry the lowest-AIC non-boundary fit as the ARIMA representative and report the failure.
def adequate(frame):
    return frame[(frame["status"] == "ok") & (~frame["near_unit"].fillna(True))
                 & (frame["lb_p_14"] > 0.05) & (frame["lb_p_21"] > 0.05) & (frame["lb_p_30"] > 0.05)]

ad0 = adequate(d0)
candidates = d0[(d0["status"] == "ok") & (~d0["near_unit"].fillna(True))]
best = candidates.sort_values("aic").iloc[0]
arima_adequate = bool(len(ad0) > 0)
if arima_adequate:
    best = ad0.sort_values("aic").iloc[0]
    rule = "Ljung-Box K=14,21,30 all p > 0.05; no boundary parameter"
else:
    rule = "No plain ARIMA passed Ljung-Box K=14,21,30; the lowest-AIC non-boundary fit is carried only as the ARIMA representative"
print("ARIMA selection rule:", rule)
print("best ARIMA:", best["name"], "AIC", best["aic"], "val RMSE", best.get("val_rmse"))

# AIC heatmap d=0
fig, ax = plt.subplots(figsize=(7.6, 5.2))
piv = d0.pivot(index="p", columns="q", values="aic")
im = ax.imshow(np.clip(piv.to_numpy(), None, 1830), cmap="viridis", aspect="auto", vmin=1795, vmax=1830)
ax.set_xticks(range(len(piv.columns)), piv.columns)
ax.set_yticks(range(len(piv.index)), piv.index)
ax.set_xlabel("q")
ax.set_ylabel("p")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        value = piv.iloc[i, j]
        ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7,
                color="white" if value < np.nanpercentile(piv.to_numpy(), 60) else "black")
ax.set_title(f"F14 ARIMA(p,0,q) AIC: overall minimum {piv.min().min():.1f} at p={piv.stack().idxmin()[0]}, q={piv.stack().idxmin()[1]}; usable minimum {best['aic']:.1f} at p={int(best['p'])}, q={int(best['q'])}")
fig.colorbar(im, ax=ax, label="AIC (colour clipped at 1830; cell labels are exact)")
save_fig(fig, "F14_arima_aic.png")

# validation RMSE heatmap
fig, ax = plt.subplots(figsize=(7.6, 5.2))
pivv = d0.pivot(index="p", columns="q", values="val_rmse")
im = ax.imshow(pivv.to_numpy(), cmap="magma_r", aspect="auto")
ax.set_xticks(range(len(pivv.columns)), pivv.columns)
ax.set_yticks(range(len(pivv.index)), pivv.index)
ax.set_xlabel("q")
ax.set_ylabel("p")
for i in range(pivv.shape[0]):
    for j in range(pivv.shape[1]):
        value = pivv.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                color="white" if value > np.nanpercentile(pivv.to_numpy(), 40) else "black")
ax.set_title(f"F15 ARIMA(p,0,q) validation RMSE: best {pivv.min().min():.3f} at p={pivv.stack().idxmin()[0]}, q={pivv.stack().idxmin()[1]}")
fig.colorbar(im, ax=ax, label="RMSE (AQHI)")
save_fig(fig, "F15_arima_rmse.png")

# residual diagnostics for the selected d=0 model
best_name = str(best["name"])
res = results[best_name]["result"]
resid = pd.Series(res.resid, index=train.index).dropna()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(resid, lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title(f"residual ACF, {best_name}")
axes[1].hist(resid, bins=45, color="#4c72b0", edgecolor="white")
axes[1].set_title(f"residual histogram, sd {resid.std():.3f}")
fig.suptitle(f"F16 Selected ARIMA residual diagnostics: Ljung-Box K=30 p = {best['lb_p_30']:.3f}, ARCH-LM p = {best['arch_lm_p_10']:.3g}",
             fontweight="bold", y=1.02)
save_fig(fig, "F16_arima_residuals.png")

# parameter table
params = pd.DataFrame({
    "parameter": res.params.index,
    "estimate": res.params.to_numpy(),
    "std_error": res.bse.to_numpy(),
    "z": res.params.to_numpy() / res.bse.to_numpy(),
})
params["p_value"] = 2 * stats.norm.sf(np.abs(params["z"]))
save_table(params, "t14_arima_selected_params.csv")

# overparameterisation around the leader
p0, q0 = int(best["p"]), int(best["q"])
over_rows = []
for label, order in [(f"({p0},0,{q0})", (p0, 0, q0)),
                     (f"({p0+1},0,{q0})", (p0 + 1, 0, q0)),
                     (f"({p0},0,{q0+1})", (p0, 0, q0 + 1)),
                     (f"({p0+1},0,{q0+1})", (p0 + 1, 0, q0 + 1))]:
    fit = fit_model(train, order=order, seasonal_order=(0, 0, 0, 0), trend="c",
                    name=f"over_{order}", family="ARIMA", maxiter=300)
    rec = fit["record"]
    rec["base"] = f"({p0},0,{q0})"
    over_rows.append(rec)
over = pd.DataFrame(over_rows)
save_table(over, "t15_arima_overparameterisation.csv")

save_facts({
    "best_name": best_name,
    "best_order": [int(best["p"]), 0, int(best["q"])],
    "best_aic": float(best["aic"]),
    "best_bic": float(best["bic"]),
    "best_lb_14": float(best["lb_p_14"]),
    "best_lb_21": float(best["lb_p_21"]),
    "best_lb_30": float(best["lb_p_30"]),
    "best_arch_p": float(best["arch_lm_p_10"]),
    "best_val_rmse": float(best["val_rmse"]),
    "best_val_mae": float(best["val_mae"]),
    "best_val_coverage": float(best["val_coverage_95"]),
    "n_adequate": int(len(ad0)),
    "arima_adequate": arima_adequate,
    "selection_rule": rule,
    "d0_aic_min": float(d0["aic"].min()),
    "d1_aic_min": float(d1["aic"].min()),
}, "arima.json")

print("03 ARIMA done")
print(grid[["name", "order", "aic", "bic", "lb_p_14", "lb_p_21", "lb_p_30", "val_rmse", "near_unit", "status"]].sort_values("aic").head(12).to_string(index=False))
print(over[["order", "aic", "bic", "lb_p_14", "near_unit"]].to_string(index=False))
