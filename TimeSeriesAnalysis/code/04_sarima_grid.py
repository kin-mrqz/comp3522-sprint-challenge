"""04 - SARIMA(p,d,q)(P,D,Q)_7 grid with weekly seasonality.

Figures F17-F19, tables t16-t18, facts sarima.json.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from tsa_lib import (fit_model, forecast_result, load_daily, save_facts, save_fig, save_table,
                     score, splits)

series = load_daily()
parts = splits(series)
train, val = parts["train"], parts["val"]

rows, results = [], {}
for D in (0, 1):
    for p in (0, 1, 2):
        for q in (0, 1, 2):
            for P in (0, 1):
                for Q in (0, 1):
                    name = f"sarima_{p}{q}_{P}{Q}_D{D}"
                    fit = fit_model(train, order=(p, 0, q), seasonal_order=(P, D, Q, 7), trend="c",
                                    name=name, family="SARIMA", maxiter=400)
                    rec = fit["record"]
                    rec.update({"p": p, "q": q, "P": P, "Q": Q, "D": D})
                    if fit["result"] is not None:
                        try:
                            mean, ci = forecast_result(fit["result"], len(val))
                            rec.update({f"val_{k}": v for k, v in score(val, mean, ci["lower"], ci["upper"], train).items()})
                        except Exception as exc:  # noqa: BLE001
                            rec["val_error"] = f"{type(exc).__name__}: {exc}"
                        results[name] = fit
                    rows.append(rec)
grid = pd.DataFrame(rows)

# refinement around the leader: the 2x2 seasonal grid may stop at the boundary
refine_specs = [
    ((1, 0, 2), (2, 0, 0, 7)), ((1, 0, 2), (3, 0, 0, 7)),
    ((1, 0, 2), (2, 0, 1, 7)), ((2, 0, 2), (2, 0, 0, 7)),
    ((1, 0, 3), (2, 0, 0, 7)), ((2, 0, 1), (2, 0, 0, 7)),
]
for order, seasonal in refine_specs:
    name = f"sarima_refine_{order[0]}{order[2]}_{seasonal[0]}{seasonal[2]}_D{seasonal[1]}"
    fit = fit_model(train, order=order, seasonal_order=seasonal, trend="c",
                    name=name, family="SARIMA_refined", maxiter=400)
    rec = fit["record"]
    rec.update({"p": order[0], "q": order[2], "P": seasonal[0], "Q": seasonal[2],
                "D": seasonal[1], "refined": True})
    if fit["result"] is not None:
        try:
            mean, ci = forecast_result(fit["result"], len(val))
            rec.update({f"val_{k}": v for k, v in score(val, mean, ci["lower"], ci["upper"], train).items()})
        except Exception as exc:  # noqa: BLE001
            rec["val_error"] = f"{type(exc).__name__}: {exc}"
        results[name] = fit
    rows.append(rec)
grid = pd.DataFrame(rows)
grid["refined"] = grid.get("refined", False).fillna(False)
save_table(grid, "t16_sarima_grid.csv")

leaders = []
for D in (0, 1):
    sub = grid[(grid["D"] == D) & (grid["status"] == "ok") & (~grid["near_unit"].fillna(True))]
    if sub.empty:
        continue
    adequate = sub[(sub["lb_p_14"] > 0.05) & (sub["lb_p_21"] > 0.05) & (sub["lb_p_30"] > 0.05)]
    pick = (adequate if not adequate.empty else sub).sort_values("aic").iloc[0]
    leaders.append(pick)
leaders = pd.DataFrame(leaders).sort_values("val_rmse")
best = leaders.iloc[0]
sarima_adequate = bool(len(grid[(grid["status"] == "ok") & (~grid["near_unit"].fillna(True))
                               & (grid["lb_p_14"] > 0.05) & (grid["lb_p_21"] > 0.05) & (grid["lb_p_30"] > 0.05)]))
print("SARIMA leaders by D:")
print(leaders[["name", "order", "seasonal_order", "D", "aic", "lb_p_14", "lb_p_21", "lb_p_30", "val_rmse", "near_unit"]].to_string(index=False))

# --------------------------------------------------------- F17 AIC landscape
top = grid[(grid["status"] == "ok")].sort_values("aic").head(20).iloc[::-1]
colors = ["#d62728" if flag else "#4c72b0" for flag in top["near_unit"]]
fig, ax = plt.subplots(figsize=(9.6, 5.6))
delta = top["aic"] - grid["aic"].min()
ax.barh(range(len(top)), delta, color=colors)
ax.set_yticks(range(len(top)), [f"{r['order']} x {r['seasonal_order'][:9]} D={r['D']}" for _, r in top.iterrows()],
              fontsize=7)
ax.set_xlabel("AIC minus the best AIC")
ax.set_title(f"F17 Delta-AIC of the 20 lowest SARIMA fits: {int(top['near_unit'].sum())} of 20 sit on a near-unit parameter (red)")
save_fig(fig, "F17_sarima_aic.png")

# --------------------------------------------------- F18 AIC vs validation
fig, ax = plt.subplots(figsize=(8.0, 5.0))
for D, marker in [(0, "o"), (1, "s")]:
    sub = grid[(grid["D"] == D) & (grid["status"] == "ok")]
    ax.scatter(sub["aic"], sub["val_rmse"], s=32, marker=marker, alpha=0.75,
               label=f"D={D}", color="#4c72b0" if D == 0 else "#ff7f0e")
    flagged = sub[sub["near_unit"]]
    ax.scatter(flagged["aic"], flagged["val_rmse"], s=40, marker="x", color="#d62728")
ax.set_xlabel("AIC (within D)")
ax.set_ylabel("validation RMSE")
ax.set_title(f"F18 AIC does not order validation error: Spearman {grid['aic'].corr(grid['val_rmse'], method='spearman'):.2f}")
ax.legend(frameon=False)
save_fig(fig, "F18_sarima_aic_vs_rmse.png")

# ------------------------------------------------------- residual diagnostics
best_name = str(best["name"])
res = results[best_name]["result"]
resid = pd.Series(res.resid, index=train.index).dropna()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
plot_acf(resid, lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title(f"residual ACF, {best_name}")
axes[1].hist(resid, bins=45, color="#4c72b0", edgecolor="white")
axes[1].set_title(f"residual histogram, sd {resid.std():.3f}")
fig.suptitle(f"F19 Selected SARIMA diagnostics: Ljung-Box K=30 p = {best['lb_p_30']:.3f}, ARCH-LM p = {best['arch_lm_p_10']:.3g}",
             fontweight="bold", y=1.02)
save_fig(fig, "F19_sarima_residuals.png")

params = pd.DataFrame({
    "parameter": res.params.index,
    "estimate": res.params.to_numpy(),
    "std_error": res.bse.to_numpy(),
    "z": res.params.to_numpy() / res.bse.to_numpy(),
})
params["p_value"] = 2 * stats.norm.sf(np.abs(params["z"]))
save_table(params, "t17_sarima_selected_params.csv")

# seasonal overparameterisation: add one seasonal AR and one seasonal MA
p0, q0, P0, Q0, D0 = int(best["p"]), int(best["q"]), int(best["P"]), int(best["Q"]), int(best["D"])
over_rows = []
for label, seasonal in [(f"({P0},{D0},{Q0})_7", (P0, D0, Q0, 7)),
                        (f"({P0+1},{D0},{Q0})_7", (P0 + 1, D0, Q0, 7)),
                        (f"({P0},{D0},{Q0+1})_7", (P0, D0, Q0 + 1, 7)),
                        (f"({P0+1},{D0},{Q0+1})_7", (P0 + 1, D0, Q0 + 1, 7))]:
    fit = fit_model(train, order=(p0, 0, q0), seasonal_order=seasonal, trend="c",
                    name=f"over_{label}", family="SARIMA", maxiter=400)
    rec = fit["record"]
    rec["seasonal_order_tested"] = label
    over_rows.append(rec)
over = pd.DataFrame(over_rows)
save_table(over, "t18_sarima_overparameterisation.csv")

save_facts({
    "best_name": best_name,
    "best_order": [p0, 0, q0],
    "best_seasonal_order": [P0, D0, Q0, 7],
    "best_aic": float(best["aic"]),
    "best_bic": float(best["bic"]),
    "best_lb_14": float(best["lb_p_14"]),
    "best_lb_21": float(best["lb_p_21"]),
    "best_lb_30": float(best["lb_p_30"]),
    "best_arch_p": float(best["arch_lm_p_10"]),
    "best_val_rmse": float(best["val_rmse"]),
    "best_val_mae": float(best["val_mae"]),
    "sarima_adequate": sarima_adequate,
    "leaders": leaders[["name", "aic", "val_rmse", "D"]].to_dict("records"),
}, "sarima.json")

print("04 SARIMA done")
print(f"best {best_name} AIC {best['aic']:.1f} val RMSE {best['val_rmse']:.3f} LB14 {best['lb_p_14']:.3f}")
print(f"any adequate: {sarima_adequate}")
print(over[["seasonal_order_tested", "aic", "bic", "lb_p_14", "near_unit"]].to_string(index=False))
