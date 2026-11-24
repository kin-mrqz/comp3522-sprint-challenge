"""Gain, a short permutation check, and TreeSHAP or linear SHAP on a sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import TABLES, ensure_dirs


def final_estimator(model):
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model


def gain_table(model, feature_names) -> pd.DataFrame:
    """Tree gain, or mean absolute coefficient when the model is linear."""
    estimator = final_estimator(model)
    names = [str(name) for name in feature_names]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float).ravel()
        kind = "gain"
    elif hasattr(estimator, "coef_"):
        coef = np.abs(np.asarray(estimator.coef_, dtype=float))
        values = coef.mean(axis=0).ravel() if coef.ndim == 2 else coef.ravel()
        kind = "mean_abs_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "kind"])
    if len(values) != len(names):
        raise ValueError(
            f"importance length {len(values)} does not match {len(names)} feature names"
        )
    table = pd.DataFrame({"feature": names, "importance": values, "kind": kind})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_table(
    model,
    X,
    y,
    feature_names,
    task: str,
    max_rows: int = 2000,
    n_repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance on a validation sample. Three repeats, so it stays cheap."""
    from sklearn.inspection import permutation_importance

    scoring = {
        "binary": "average_precision",
        "multiclass": "balanced_accuracy",
        "regression": "neg_mean_absolute_error",
    }[task]
    x = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    if len(x) > max_rows:
        take = np.sort(np.random.default_rng(seed).choice(len(x), size=max_rows, replace=False))
        x = x[take]
        labels = labels[take]
    result = permutation_importance(
        model,
        x,
        labels,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=scoring,
    )
    table = pd.DataFrame(
        {
            "feature": [str(name) for name in feature_names],
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def _shap_frame(values, feature_names) -> pd.DataFrame:
    names = [str(name) for name in feature_names]
    if isinstance(values, list):
        if len(values) == 2:
            array = np.asarray(values[1])
        else:
            array = np.mean([np.abs(np.asarray(part)) for part in values], axis=0)
    else:
        array = np.asarray(values)
    if array.ndim == 3:
        array = np.mean(np.abs(array), axis=2)
    if array.ndim == 2 and array.shape[1] == len(names) + 1:
        array = array[:, :-1]
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError(f"unexpected SHAP shape {getattr(array, 'shape', None)}")
    importance = np.mean(np.abs(array), axis=0)
    table = pd.DataFrame({"feature": names, "mean_abs_shap": importance})
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def shap_table(model, X, feature_names, max_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Mean absolute SHAP on at most a few thousand rows. Linear models fall back to coefficients."""
    import shap

    estimator = final_estimator(model)
    x = np.asarray(X, dtype=float)
    limit = max_rows if x.shape[1] <= 100 else min(max_rows, 500)
    if len(x) > limit:
        take = np.sort(np.random.default_rng(seed).choice(len(x), size=limit, replace=False))
        x = x[take]
    name = estimator.__class__.__name__
    tree_tokens = ("XGB", "LGBM", "CatBoost", "GradientBoosting", "RandomForest", "DecisionTree")
    try:
        if any(token in name for token in tree_tokens):
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(x)
        else:
            explainer = shap.LinearExplainer(estimator, x)
            values = explainer.shap_values(x)
        return _shap_frame(values, feature_names)
    except Exception as exc:
        fallback = gain_table(estimator, feature_names)
        if fallback.empty:
            raise
        fallback = fallback.rename(columns={"importance": "mean_abs_shap"})
        fallback["note"] = f"shap fallback: {type(exc).__name__}: {exc}"
        return fallback


def save_importance_bar(table: pd.DataFrame, path: Path, title: str, value_column: str = "importance") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"].astype(str), top[value_column])
    ax.set_title(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def write_explanations(model, X, y, feature_names, task: str, max_rows: int = 2000) -> dict[str, Path]:
    """Write gain, permutation, and SHAP tables plus one bar chart."""
    ensure_dirs()
    gain = gain_table(model, feature_names)
    perm = permutation_table(model, X, y, feature_names, task=task, max_rows=max_rows)
    shap = shap_table(model, X, feature_names, max_rows=max_rows)
    paths = {
        "gain": TABLES / "gain_importance.csv",
        "permutation": TABLES / "permutation_importance.csv",
        "shap": TABLES / "shap_values.csv",
    }
    gain.to_csv(paths["gain"], index=False)
    perm.to_csv(paths["permutation"], index=False)
    shap.to_csv(paths["shap"], index=False)
    value_column = "importance" if "importance" in gain.columns else "mean_abs_shap"
    paths["figure"] = save_importance_bar(
        gain,
        TABLES.parent / "figures" / "importance_top15.png",
        "Validation importance",
        value_column=value_column,
    )
    return paths
