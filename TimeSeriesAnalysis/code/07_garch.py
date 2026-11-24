"""07 - GARCH family on the selected mean-model residuals, plus a 2025 VaR filter.

Figures F26-F29, tables t25-t28, facts garch.json.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tsa_lib import (DATA, FACTS, exog_frame, load_daily, save_facts, save_fig, save_table,
                     score, splits)

series = load_daily()
parts = splits(series)
train, val, test, full = parts["train"], parts["val"], parts["test"], parts["full"]

resid_path = DATA / "mean_resid.csv"
if not resid_path.exists():
    raise SystemExit("run 05_sarimax_seasonal.py first: results/data/mean_resid.csv is missing")
resid = pd.read_csv(resid_path, parse_dates=["date"]).set_index("date")["resid"].astype("float64").dropna()
print(f"residuals: {len(resid)} from {resid.index.min().date()} to {resid.index.max().date()}")

# ------------------------------------------------------ ARCH tests on mean
arch_rows = []
for lag in (5, 10, 20):
    stat, pvalue, fstat, fpvalue = het_arch(resid, nlags=lag)
    arch_rows.append({"test": "ARCH-LM", "lag": lag, "stat": float(stat), "p_value": float(pvalue),
                      "f_stat": float(fstat), "f_p_value": float(fpvalue)})
lb_sq = acorr_ljungbox(resid ** 2, lags=[7, 14, 21, 30], return_df=True)
for lag in (7, 14, 21, 30):
    arch_rows.append({"test": "Ljung-Box on squared residuals", "lag": lag,
                      "stat": float(lb_sq.loc[lag, "lb_stat"]), "p_value": float(lb_sq.loc[lag, "lb_pvalue"]),
                      "f_stat": np.nan, "f_p_value": np.nan})
arch_table = pd.DataFrame(arch_rows)
save_table(arch_table, "t25_arch_tests.csv")

# ------------------------------------------------------------- GARCH grid
specs = [
    ("Constant", dict(vol="Constant")),
    ("ARCH1", dict(vol="ARCH", p=1)),
    ("GARCH11", dict(vol="Garch", p=1, q=1)),
    ("GARCH12", dict(vol="Garch", p=1, q=2)),
    ("GARCH21", dict(vol="Garch", p=2, q=1)),
    ("GARCH11_t", dict(vol="Garch", p=1, q=1, dist="t")),
    ("GJR11", dict(vol="Garch", p=1, o=1, q=1)),
    ("EGARCH11", dict(vol="EGARCH", p=1, o=1, q=1)),
]
rows, results = [], {}
for name, kwargs in specs:
    try:
        am = arch_model(resid, mean="Zero", **kwargs)
        res = am.fit(disp="off", show_warning=False)
        std = pd.Series(res.std_resid).dropna()
        lb = acorr_ljungbox(std ** 2, lags=[14, 21, 30], return_df=True)
        params = res.params
        if "alpha[1]" in params.index:
            alpha = float(params["alpha[1]"])
        elif "alpha[1]" not in params.index and "alpha" in params.index:
            alpha = float(params["alpha"])
        else:
            alpha = 0.0
        beta = float(params.get("beta[1]", 0.0))
        gamma = float(params.get("gamma[1]", 0.0))
        if kwargs.get("vol") == "EGARCH":
            persistence = beta
        elif "o" in kwargs:
            persistence = alpha + beta + 0.5 * gamma
        elif kwargs.get("vol") == "Garch":
            persistence = alpha + beta
        elif kwargs.get("vol") == "ARCH":
            persistence = alpha
        else:
            persistence = 0.0
        half_life = float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else np.inf
        omega = float(params.get("omega", np.nan))
        unconditional_var = omega / (1 - persistence) if 0 < persistence < 1 else np.nan
        rows.append({
            "name": name,
            "aic": float(res.aic),
            "bic": float(res.bic),
            "loglik": float(res.loglikelihood),
            "params": str(dict(params)),
            "persistence": float(persistence),
            "half_life_days": half_life,
            "unconditional_vol": float(np.sqrt(unconditional_var)) if np.isfinite(unconditional_var) and unconditional_var >= 0 else np.nan,
            "lb_sq_14": float(lb.loc[14, "lb_pvalue"]),
            "lb_sq_21": float(lb.loc[21, "lb_pvalue"]),
            "lb_sq_30": float(lb.loc[30, "lb_pvalue"]),
            "arch_p_10": float(het_arch(std, nlags=10)[1]),
            "status": "ok",
        })
        results[name] = res
    except Exception as exc:  # noqa: BLE001
        rows.append({"name": name, "status": f"failed: {type(exc).__name__}: {exc}"})
grid = pd.DataFrame(rows)
save_table(grid, "t26_garch_grid.csv")

ok = grid[grid["status"] == "ok"].copy()
adequate = ok[(ok["lb_sq_14"] > 0.05) & (ok["lb_sq_21"] > 0.05) & (ok["lb_sq_30"] > 0.05)
              & (ok["persistence"] < 1)]
best = (adequate if not adequate.empty else ok).sort_values("aic").iloc[0]
best_name = str(best["name"])
best_res = results[best_name]
# The manual one-step variance recursion is written for GARCH/GJR. If EGARCH wins,
# use the best GARCH-family fit for the VaR filter and say so.
var_name = best_name if not best_name.startswith("EGARCH") else "GARCH11"
var_res = results[var_name]
print("GARCH adequate:", len(adequate), "of", len(ok))
print("best:", best_name, "AIC", best["aic"], "persistence", best["persistence"])

# ------------------------------------------------- F26 residual and squared ACF
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
plot_acf(resid, lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title(f"mean-model residual ACF: r1 = {resid.autocorr(1):.3f}")
plot_acf(resid ** 2, lags=40, ax=axes[1], alpha=0.05)
axes[1].set_title(f"squared residual ACF: r1 = {(resid**2).autocorr(1):.3f}, ARCH-LM p = {arch_table.iloc[1]['p_value']:.2g}")
fig.suptitle("F26 The mean model is not the whole story: level residuals are clean, squared residuals are not",
             fontweight="bold", y=1.02)
save_fig(fig, "F26_arch_effects.png")

# ------------------------------------------- F27 conditional volatility
vol = pd.Series(best_res.conditional_volatility, index=resid.index)
fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.0), sharex=True)
axes[0].plot(train.index, train.to_numpy(), color="0.55", lw=0.6)
axes[0].set_ylabel("AQHI")
axes[0].set_title(f"training AQHI and {best_name} conditional volatility")
axes[1].plot(vol.index, vol.to_numpy(), color="#d62728", lw=0.8)
axes[1].set_ylabel("conditional SD")
axes[1].set_xlabel("date")
axes[1].axhline(float(np.sqrt(best["unconditional_vol"] ** 2)), color="0.4", ls="--", lw=0.8,
                label="unconditional SD")
axes[1].legend(frameon=False)
fig.suptitle(f"F27 Conditional volatility clusters on the episode months (persistence {best['persistence']:.3f}, half-life {best['half_life_days']:.1f} d)",
             fontweight="bold", y=1.0)
fig.tight_layout()
save_fig(fig, "F27_conditional_volatility.png")

# --------------------------------------------------- F28 term structure
horizon = 365
fc = best_res.forecast(horizon=horizon, reindex=False)
path = np.sqrt(fc.variance.iloc[-1].to_numpy(dtype=float))
uncond = float(np.sqrt(best["unconditional_vol"] ** 2)) if np.isfinite(best["unconditional_vol"]) else np.nan
fig, ax = plt.subplots(figsize=(9.6, 4.2))
ax.plot(range(1, horizon + 1), path, color="#d62728", lw=1.8, label="GARCH variance forecast")
ax.axhline(uncond, color="0.4", ls="--", lw=1.0, label="unconditional SD")
ax.set_xlabel("horizon (days)")
ax.set_ylabel("forecast SD")
ax.set_title(f"F28 Volatility term structure from 2023-12-31: half the gap to unconditional closes in {best['half_life_days']:.1f} days")
ax.legend(frameon=False)
save_fig(fig, "F28_vol_term_structure.png")

# ------------------------------------------------- 2025 mean model refit
spec = json.loads((FACTS / "sarimax.json").read_text())
kind = spec["best_exog_set"]
k = int(spec["best_k_fourier"])
p, q = int(spec["best_p"]), int(spec["best_q"])
full_exog = exog_frame(full.index, kind, k=k)
future_index = pd.date_range("2024-12-01", test.index.max(), freq="D")
future_exog = exog_frame(future_index, kind, k=k)
seasonal = tuple(spec.get("best_seasonal_order", [0, 0, 0, 0]))
mean_model = SARIMAX(full, exog=full_exog, order=(p, 0, q), seasonal_order=seasonal,
                     trend="c", enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=400)
mean_fc = pd.Series(mean_model.get_forecast(steps=len(future_index), exog=future_exog).predicted_mean,
                    index=future_index)
mu = mean_fc.reindex(test.index)
errors = test - mu
print(f"2025 mean forecast RMSE {np.sqrt(np.mean(errors.dropna()**2)):.3f}")

# one-step GARCH filter with fixed parameters
params = var_res.params
omega = float(params.get("omega", np.nan))
alpha = float(params.get("alpha[1]", 0.0))
beta = float(params.get("beta[1]", 0.0))
gamma = float(params.get("gamma[1]", 0.0))
h = float(var_res.conditional_volatility.iloc[-1] ** 2)
sigma_path = []
eps_prev = float(resid.iloc[-1])
for day in test.index:
    sigma_path.append(np.sqrt(h))
    eps = float(errors.loc[day]) if pd.notna(errors.loc[day]) else 0.0
    h = omega + alpha * eps_prev ** 2 + beta * h
    if "o" in specs[[s[0] for s in specs].index(var_name)][1] and eps_prev < 0:
        h += gamma * eps_prev ** 2
    eps_prev = eps
sigma_path = np.asarray(sigma_path)
sigma_const = float(resid.std())
mu_arr = mu.to_numpy(dtype=float)
y_arr = test.to_numpy(dtype=float)

var_rows = []
for label, sigma in [(var_name, sigma_path), ("Constant", np.full(len(test), sigma_const))]:
    nu = float(var_res.params.get("nu", np.nan))
    for coverage in (0.95, 0.99):
        if np.isfinite(nu) and nu > 2:
            z = float(stats.t.ppf(coverage, nu) * np.sqrt((nu - 2) / nu))
        else:
            z = float(stats.norm.ppf(coverage))
        q_hat = mu_arr + z * sigma
        exceed = y_arr > q_hat
        n = len(y_arr)
        x = int(exceed.sum())
        p = 1 - coverage
        if 0 < x < n:
            lr = -2 * ((n - x) * np.log((1 - p) / (1 - x / n)) + x * np.log(p / (x / n)))
            kupiec_p = float(stats.chi2.sf(lr, 1))
        else:
            kupiec_p = np.nan
        loss = np.where(y_arr >= q_hat, coverage * (y_arr - q_hat), (1 - coverage) * (q_hat - y_arr))
        var_rows.append({"model": label, "coverage": coverage, "exceedances": x, "n": n,
                         "expected": float(n * p), "rate": float(x / n),
                         "kupiec_p": kupiec_p, "pinball": float(loss.mean()),
                         "mean_sigma": float(np.mean(sigma))})
var_table = pd.DataFrame(var_rows)
save_table(var_table, "t28_var_backtest.csv")

fig, ax = plt.subplots(figsize=(12.5, 4.4))
ax.plot(test.index, y_arr, color="#333333", lw=0.9, label="AQHI 2025")
z95 = stats.norm.ppf(0.95)
ax.plot(test.index, mu_arr + z95 * sigma_path, color="#d62728", lw=1.0, label="GARCH 95% upper VaR")
ax.plot(test.index, mu_arr + z95 * np.full(len(test), sigma_const), color="#4c72b0", lw=1.0, ls="--",
        label="constant-variance 95% upper VaR")
ax.set_ylabel("AQHI")
ax.set_title(f"F29 2025 one-step upper VaR: {var_name} exceeds {var_table.iloc[0]['exceedances']} times, constant variance {var_table.iloc[2]['exceedances']} times")
ax.legend(frameon=False, fontsize=8.5)
save_fig(fig, "F29_var_backtest.png")

# parameter table
param_table = pd.DataFrame({
    "parameter": best_res.params.index,
    "estimate": best_res.params.to_numpy(dtype=float),
    "std_error": best_res.std_err.to_numpy(dtype=float) if hasattr(best_res, "std_err") else np.nan,
})
param_table["z"] = param_table["estimate"] / param_table["std_error"]
param_table["p_value"] = 2 * stats.norm.sf(np.abs(param_table["z"]))
save_table(param_table, "t27_garch_params.csv")

save_facts({
    "best_name": best_name,
    "best_aic": float(best["aic"]),
    "best_bic": float(best["bic"]),
    "persistence": float(best["persistence"]),
    "half_life_days": float(best["half_life_days"]),
    "unconditional_vol": float(best["unconditional_vol"]),
    "arch_lm_p_10": float(arch_table.iloc[1]["p_value"]),
    "n_adequate": int(len(adequate)),
    "var": var_table.to_dict("records"),
    "mean_forecast_rmse_2025": float(np.sqrt(np.mean(errors.dropna() ** 2))),
}, "garch.json")

print("07 GARCH done")
print(grid[["name", "aic", "bic", "persistence", "half_life_days", "lb_sq_14", "lb_sq_30", "status"]].to_string(index=False))
print(var_table.to_string(index=False))
