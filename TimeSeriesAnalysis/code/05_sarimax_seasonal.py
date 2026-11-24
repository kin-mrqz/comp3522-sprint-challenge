"""05 - SARIMAX with deterministic annual seasonality and ARMA errors.

The annual cycle is modelled with Fourier terms (or month dummies), the weekly
cycle with day-of-week dummies, and the remainder with ARMA errors.

Figures F20-F23, tables t19-t22, facts sarimax.json.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

from tsa_lib import (DATA, DOW_ABBR, MONTH_ABBR, exog_frame, fit_model, forecast_result,
                     load_daily, save_facts, save_fig, save_table, score, splits)

series = load_daily()
parts = splits(series)
train, val = parts["train"], parts["val"]

specs = []
for k in (1, 2, 3):
    for p, q in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]:
        specs.append((f"fourier{k}_dow", "fourier_dow", k, p, q))
for p, q in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    specs.append(("month_dow", "month_dow", 0, p, q))
specs.append(("fourier2_dow_trend", "fourier_dow_trend", 2, 1, 1))
specs.append(("fourier4_dow", "fourier_dow", 4, 1, 1))

rows, results = [], {}
for label, kind, k, p, q in specs:
    name = f"sarimax_{label}_a{p}{q}"
    train_exog = exog_frame(train.index, kind, k=k)
    val_exog = exog_frame(val.index, kind, k=k)
    fit = fit_model(train, order=(p, 0, q), seasonal_order=(0, 0, 0, 0), exog=train_exog,
                    trend="c", name=name, family="SARIMAX", exog_kind=label, k_fourier=k,
                    maxiter=400)
    rec = fit["record"]
    rec.update({"p": p, "q": q, "exog_set": label, "n_exog": int(train_exog.shape[1])})
    if fit["result"] is not None:
        try:
            mean, ci = forecast_result(fit["result"], len(val), exog=val_exog)
            rec.update({f"val_{kk}": v for kk, v in score(val, mean, ci["lower"], ci["upper"], train).items()})
        except Exception as exc:  # noqa: BLE001
            rec["val_error"] = f"{type(exc).__name__}: {exc}"
        results[name] = {"fit": fit, "train_exog": train_exog, "val_exog": val_exog}
    rows.append(rec)

# combined model: annual Fourier + DOW + weekly SARMA errors.
# The weekly seasonal AR is what removes the negative lag-7 residual autocorrelation.
combined = []
for k in (2, 3):
    for p, q in [(1, 2), (1, 3)]:
        for P, Q in [(1, 0), (2, 0)]:
            combined.append((f"fourier{k}_dow_sarma", "fourier_dow", k, p, q, (P, 0, Q, 7)))
for label, kind_, k, p, q, seasonal in combined:
    name = f"sarimax_{label}_a{p}{q}_P{seasonal[0]}"
    train_exog = exog_frame(train.index, kind_, k=k)
    val_exog = exog_frame(val.index, kind_, k=k)
    fit = fit_model(train, order=(p, 0, q), seasonal_order=seasonal, exog=train_exog,
                    trend="c", name=name, family="SARIMAX_weekly", exog_kind=label,
                    k_fourier=k, maxiter=500)
    rec = fit["record"]
    rec.update({"p": p, "q": q, "exog_set": label, "n_exog": int(train_exog.shape[1]),
                "P": seasonal[0], "Q": seasonal[2], "D": seasonal[1],
                "seasonal_order": str(seasonal)})
    if fit["result"] is not None:
        try:
            mean, ci = forecast_result(fit["result"], len(val), exog=val_exog)
            rec.update({f"val_{kk}": v for kk, v in score(val, mean, ci["lower"], ci["upper"], train).items()})
        except Exception as exc:  # noqa: BLE001
            rec["val_error"] = f"{type(exc).__name__}: {exc}"
        results[name] = {"fit": fit, "train_exog": train_exog, "val_exog": val_exog}
    rows.append(rec)
grid = pd.DataFrame(rows)
save_table(grid, "t19_sarimax_grid.csv")

ok = grid[(grid["status"] == "ok") & (~grid["near_unit"].fillna(True))]
adequate = ok[(ok["lb_p_14"] > 0.05) & (ok["lb_p_21"] > 0.05) & (ok["lb_p_30"] > 0.05)]
candidate_pool = adequate if not adequate.empty else ok
if not candidate_pool.empty:
    near = candidate_pool[candidate_pool["val_rmse"] <= candidate_pool["val_rmse"].min() + 0.01]
    best = near.sort_values("aic").iloc[0]
else:
    best = ok.sort_values("val_rmse").iloc[0]
print("adequate SARIMAX fits:", len(adequate), "of", len(ok))
print("best:", best["name"], "val RMSE", best["val_rmse"], "AIC", best["aic"], "LB30", best["lb_p_30"])
print(adequate[["name", "exog_set", "p", "q", "aic", "val_rmse", "lb_p_30"]].sort_values("val_rmse").head(12).to_string(index=False))

# ------------------------------------------------------- F20 AIC by family
fig, ax = plt.subplots(figsize=(11, 4.8))
labels = grid["name"].tolist()
ypos = np.arange(len(grid))
colors = {"fourier1_dow": "#9ecae1", "fourier2_dow": "#4c72b0", "fourier3_dow": "#08519c",
          "fourier4_dow": "#08306b", "month_dow": "#ff7f0e", "fourier2_dow_trend": "#d62728",
          "fourier2_dow_sarma": "#2ca02c", "fourier3_dow_sarma": "#006d2c"}
ax.barh(ypos, grid["aic"], color=[colors.get(x, "#7f7f7f") for x in grid["exog_set"]])
ax.set_yticks(ypos, [n.replace("sarimax_", "") for n in labels], fontsize=6.5)
ax.set_xlabel("AIC (comparable only within one exog set)")
ax.set_title(f"F20 SARIMAX AIC: the best exog set is {best['exog_set']}; AIC is not comparable across regressor sets")
save_fig(fig, "F20_sarimax_aic.png")

# --------------------------------------------------- F21 validation RMSE
ordered = grid.sort_values("val_rmse").head(20)
fig, ax = plt.subplots(figsize=(9.8, 5.2))
ax.barh(range(len(ordered)), ordered["val_rmse"],
        color=[colors.get(x, "#7f7f7f") for x in ordered["exog_set"]])
ax.set_yticks(range(len(ordered)), [n.replace("sarimax_", "") for n in ordered["name"]], fontsize=7)
ax.set_xlabel("validation RMSE (AQHI)")
ax.set_title(f"F21 Validation RMSE by SARIMAX specification: best {ordered['val_rmse'].min():.3f} ({ordered.iloc[0]['name']})")
for i, (value, flag) in enumerate(zip(ordered["val_rmse"], ordered["near_unit"])):
    ax.text(value + 0.002, i, f"{value:.3f}" + (" *" if flag else ""), va="center", fontsize=7)
save_fig(fig, "F21_sarimax_rmse.png")

# ------------------------------------------------- residual and QQ plots
best_name = str(best["name"])
res = results[best_name]["fit"]["result"]
resid = pd.Series(res.resid, index=train.index).dropna()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
plot_acf(resid, lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title(f"residual ACF, {best_name}")
qqplot(resid, line="45", ax=axes[1], marker=".", markersize=2)
axes[1].set_title("normal Q-Q")
fig.suptitle(f"F22 Selected SARIMAX residuals: Ljung-Box K=30 p = {best['lb_p_30']:.3f}, ARCH-LM p = {best['arch_lm_p_10']:.3g}, JB p = {best['jb_p']:.3g}",
             fontweight="bold", y=1.02)
save_fig(fig, "F22_sarimax_residuals.png")

# --------------------------------------------- F23 fitted annual cycle and DOW
params = res.params
param_rows = []
for name, value, se in zip(params.index, params.to_numpy(), res.bse.to_numpy()):
    param_rows.append({"parameter": name, "estimate": float(value), "std_error": float(se),
                       "z": float(value / se), "p_value": float(2 * stats.norm.sf(abs(value / se)))})
param_table = pd.DataFrame(param_rows)
save_table(param_table, "t20_sarimax_selected_params.csv")

k_best = int(best["k_fourier"]) if pd.notna(best["k_fourier"]) else 0
t_grid = np.arange(0, 366)
annual = np.zeros_like(t_grid, dtype=float)
for kk in range(1, k_best + 1):
    sin_name, cos_name = f"sin{kk}", f"cos{kk}"
    if sin_name in params.index:
        annual += params[sin_name] * np.sin(2 * np.pi * kk * t_grid / 365.25)
        annual += params[cos_name] * np.cos(2 * np.pi * kk * t_grid / 365.25)
dow_effects = [0.0] + [float(params.get(f"dow{d}", 0.0)) for d in range(1, 7)]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
axes[0].plot(t_grid, annual, color="#4c72b0", lw=1.8)
axes[0].set_xlabel("day of year")
axes[0].set_ylabel("AQHI relative to baseline")
axes[0].set_title(f"fitted annual cycle, Fourier K={k_best}, amplitude {annual.max()-annual.min():.2f}")
axes[1].bar(range(7), dow_effects, color="#c44e52")
axes[1].set_xticks(range(7), DOW_ABBR)
axes[1].set_ylabel("AQHI relative to Monday")
axes[1].set_title(f"fitted day-of-week effects, range {max(dow_effects)-min(dow_effects):.3f}")
fig.suptitle("F23 Deterministic seasonal components in the selected SARIMAX", fontweight="bold", y=1.02)
save_fig(fig, "F23_sarimax_seasonal_components.png")

# --------------------------------------------------------- trend in model
trend_rows = grid[grid["exog_set"] == "fourier2_dow_trend"].copy()
if not trend_rows.empty:
    trend_name = trend_rows.iloc[0]["name"]
    trend_res = results[trend_name]["fit"]["result"]
    if "trend" in trend_res.params.index:
        coef = float(trend_res.params["trend"])
        se = float(trend_res.bse["trend"])
        pval = float(2 * stats.norm.sf(abs(coef / se)))
        save_table(pd.DataFrame([{"model": trend_name, "trend_per_year": coef, "std_error": se,
                                  "z": coef / se, "p_value": pval}]), "t21_trend_in_sarimax.csv")
    else:
        pval = np.nan
        coef = np.nan
else:
    pval = np.nan
    coef = np.nan

# save the selected residuals for GARCH
resid.rename("resid").to_csv(DATA / "mean_resid.csv", index_label="date")
save_table(resid.rename("resid").reset_index(), "t22_selected_residuals.csv")

save_facts({
    "best_name": best_name,
    "best_exog_set": str(best["exog_set"]),
    "best_k_fourier": int(best["k_fourier"]) if pd.notna(best["k_fourier"]) else 0,
    "best_p": int(best["p"]), "best_q": int(best["q"]),
    "best_seasonal_order": [int(best.get("P", 0)), int(best.get("D", 0)), int(best.get("Q", 0)), 7],
    "best_aic": float(best["aic"]),
    "best_bic": float(best["bic"]),
    "best_lb_14": float(best["lb_p_14"]),
    "best_lb_21": float(best["lb_p_21"]),
    "best_lb_30": float(best["lb_p_30"]),
    "best_arch_p": float(best["arch_lm_p_10"]),
    "best_jb_p": float(best["jb_p"]),
    "best_val_rmse": float(best["val_rmse"]),
    "best_val_mae": float(best["val_mae"]),
    "best_val_coverage": float(best["val_coverage_95"]),
    "n_adequate": int(len(adequate)),
    "trend_in_model_per_year": coef,
    "trend_in_model_p": pval,
    "dow_effects": {DOW_ABBR[d]: float(dow_effects[d]) for d in range(7)},
    "annual_amplitude": float(annual.max() - annual.min()),
}, "sarimax.json")

print("05 SARIMAX done")
print(f"best {best_name}: AIC {best['aic']:.1f}, val RMSE {best['val_rmse']:.3f}, LB14/21/30 {best['lb_p_14']:.3f}/{best['lb_p_21']:.3f}/{best['lb_p_30']:.3f}")
print(f"trend in model {coef:+.3f}/yr, p={pval:.3f}" if pd.notna(coef) else "trend model missing")
print(f"annual amplitude {annual.max()-annual.min():.3f}, DOW range {max(dow_effects)-min(dow_effects):.3f}")
