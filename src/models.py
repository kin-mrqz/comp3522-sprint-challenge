"""Small, pre-registered grids for linear models, XGBoost, LightGBM, and CatBoost.

Validation chooses the configuration. The test split is never passed into these
functions. Reported CSI is tuned on the same validation scores, so it is a
model-selection diagnostic, not a test result. The selection metric is PR-AUC
for binary targets and negative MAE for regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

LINEAR_GRID = {
    "regression": [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}],
    "binary": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    "multiclass": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
}
TREE_GRID = [
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 120, "subsample": 0.9, "colsample_bytree": 0.9},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.9, "colsample_bytree": 0.9},
]
CATBOOST_GRID = [
    {"depth": 4, "learning_rate": 0.1, "iterations": 120},
    {"depth": 6, "learning_rate": 0.05, "iterations": 200},
]
GRIDS = {
    "linear": LINEAR_GRID,
    "xgboost": {"regression": TREE_GRID, "binary": TREE_GRID, "multiclass": TREE_GRID},
    "lightgbm": {"regression": TREE_GRID, "binary": TREE_GRID, "multiclass": TREE_GRID},
    "catboost": {"regression": CATBOOST_GRID, "binary": CATBOOST_GRID, "multiclass": CATBOOST_GRID},
}
ALERT_RATES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5)
ROW_COLUMNS = (
    "family",
    "ok",
    "task",
    "params",
    "selection_metric",
    "val_score",
    "val_pr_auc",
    "val_roc_auc",
    "val_csi",
    "val_mae",
    "val_rmse",
    "val_balanced_accuracy",
    "train_subsampled",
    "val_subsampled",
    "error",
)


def detect_task(y) -> str:
    """Binary, multiclass, or regression. Pass ``task=`` when a 3–8 level integer is really a measurement."""
    labels = pd.Series(np.asarray(y)).dropna()
    if labels.empty:
        raise ValueError("target has no labelled rows")
    if pd.api.types.is_bool_dtype(labels):
        return "binary"
    nunique = int(labels.nunique())
    if pd.api.types.is_numeric_dtype(labels):
        numeric = pd.to_numeric(labels, errors="coerce")
        whole = numeric.notna().all() and np.all(np.mod(numeric.to_numpy(dtype=float), 1) == 0)
        if nunique <= 2:
            return "binary"
        if whole and nunique <= 8:
            return "multiclass"
        return "regression"
    if nunique <= 2:
        return "binary"
    if nunique <= 8:
        return "multiclass"
    raise ValueError(
        f"target has {nunique} non-numeric classes. Recode it, or pass task= if that is really the label."
    )


def design_matrix(frame, feature_columns, train_positions, transform_positions) -> dict:
    """Train-fit impute, scale, and one-hot. High-cardinality text is dropped, not encoded.

    Numeric columns are standardized on the training rows. Tree and linear models
    then share one matrix. The test positions may be included in
    ``transform_positions`` only after the preprocessor has been fit on train.
    """
    train_positions = np.asarray(train_positions, dtype=int)
    transform_positions = np.asarray(transform_positions, dtype=int)
    frame = frame.copy()
    present = [column for column in feature_columns if column in frame.columns]
    missing = [column for column in feature_columns if column not in frame.columns]
    train = frame.iloc[train_positions]
    numeric, categorical, notes = [], [], []
    for column in present:
        series = train[column]
        if int(series.nunique(dropna=True)) <= 1:
            notes.append({"column": column, "reason": "constant_on_train"})
            continue
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            numeric.append(column)
            continue
        levels = int(series.nunique(dropna=True))
        if levels > 40:
            notes.append({"column": column, "reason": "too_many_levels_aggregate_first", "nunique": levels})
            continue
        categorical.append(column)
    if not numeric and not categorical:
        raise ValueError("no usable feature columns after the train-only screen")
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    x_train = np.asarray(preprocessor.fit_transform(frame.iloc[train_positions]), dtype=float)
    x_other = np.asarray(preprocessor.transform(frame.iloc[transform_positions]), dtype=float)
    return {
        "X_train": x_train,
        "X_other": x_other,
        "feature_names": [str(name) for name in preprocessor.get_feature_names_out()],
        "preprocessor": preprocessor,
        "dropped": notes,
        "numeric": numeric,
        "categorical": categorical,
        "missing_columns": missing,
    }


def _csi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    hits = int(((y_true == 1) & (y_pred == 1)).sum())
    misses = int(((y_true == 1) & (y_pred == 0)).sum())
    false_alarms = int(((y_true == 0) & (y_pred == 1)).sum())
    denom = hits + misses + false_alarms
    if denom == 0:
        return float("nan")
    return hits / denom


def best_csi(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Highest CSI across the project's alert rates. Tuned on the scores it is given."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    y_true, scores = y_true[ok], scores[ok]
    if len(scores) == 0 or int(y_true.sum()) == 0:
        return float("nan")
    best = -1.0
    for rate in ALERT_RATES:
        threshold = float(np.quantile(scores, 1.0 - rate))
        score = _csi(y_true, (scores >= threshold).astype(int))
        if np.isfinite(score) and score > best:
            best = score
    return float(best) if best >= 0 else float("nan")


def _blank_row(family: str, task: str) -> dict:
    row = {column: np.nan for column in ROW_COLUMNS}
    row["family"] = family
    row["task"] = task
    row["ok"] = False
    row["error"] = ""
    row["params"] = ""
    row["selection_metric"] = ""
    row["train_subsampled"] = False
    row["val_subsampled"] = False
    return row


def _labelled_pair(X, y):
    x = np.asarray(X, dtype=float)
    series = pd.Series(np.asarray(y))
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        mask = np.isfinite(numeric.to_numpy(dtype=float))
        y_out = numeric.to_numpy(dtype=float)
    else:
        mask = series.notna().to_numpy()
        y_out = series.to_numpy()
    if int(mask.sum()) == 0:
        raise ValueError("no labelled rows in this split")
    return x[mask], y_out[mask]


def _subsample(X, y, max_rows: int, seed: int):
    if len(X) <= max_rows:
        return X, y, False
    take = np.sort(np.random.default_rng(seed).choice(len(X), size=max_rows, replace=False))
    return X[take], np.asarray(y)[take], True


def _binary_scale(y) -> float:
    values, counts = np.unique(np.asarray(y), return_counts=True)
    if len(values) != 2:
        return 1.0
    try:
        numeric = {float(value) for value in values}
    except (TypeError, ValueError):
        numeric = set()
    if numeric <= {0.0, 1.0}:
        lookup = {float(value): int(count) for value, count in zip(values, counts)}
        positive = lookup.get(1.0, 0)
        negative = lookup.get(0.0, 0)
        return float(negative / positive) if positive else 1.0
    minority = int(counts.min())
    majority = int(counts.max())
    return float(majority / minority) if minority else 1.0


def _positive_scores(model, X, y):
    classes = list(model.classes_)
    proba = model.predict_proba(X)
    index = 1 if proba.shape[1] > 1 else 0
    try:
        numeric = {float(value) for value in classes}
    except (TypeError, ValueError):
        numeric = set()
    if numeric <= {0.0, 1.0} and 1.0 in {float(value) for value in classes}:
        index = [float(value) for value in classes].index(1.0)
    positive = classes[index]
    scores = proba[:, index]
    labels = (np.asarray(y) == positive).astype(int)
    return scores, labels


def _metrics(model, X, y, task: str) -> dict:
    x, y = _labelled_pair(X, y)
    if task == "regression":
        pred = np.asarray(model.predict(x), dtype=float)
        mae = float(mean_absolute_error(y, pred))
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        return {
            "selection_metric": "neg_mae",
            "val_score": -mae,
            "val_mae": mae,
            "val_rmse": rmse,
        }
    if task == "binary":
        scores, labels = _positive_scores(model, x, y)
        pr_auc = float(average_precision_score(labels, scores))
        try:
            roc = float(roc_auc_score(labels, scores))
        except ValueError:
            roc = float("nan")
        return {
            "selection_metric": "pr_auc",
            "val_score": pr_auc,
            "val_pr_auc": pr_auc,
            "val_roc_auc": roc,
            "val_csi": best_csi(labels, scores),
        }
    pred = model.predict(x)
    balanced = float(balanced_accuracy_score(y, pred))
    return {
        "selection_metric": "balanced_accuracy",
        "val_score": balanced,
        "val_balanced_accuracy": balanced,
    }


def _make_model(family: str, params: dict, task: str, y_train, seed: int):
    if family == "linear":
        if task == "regression":
            return Ridge(alpha=float(params.get("alpha", 1.0)))
        solver = "liblinear" if task == "binary" else "lbfgs"
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            solver=solver,
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        )
    if family == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        common = dict(params)
        common.update(random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist")
        if task == "regression":
            return XGBRegressor(objective="reg:squarederror", **common)
        if task == "binary":
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=_binary_scale(y_train),
                **common,
            )
        return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **common)
    if family == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        common = dict(params)
        common.update(random_state=seed, n_jobs=-1, verbosity=-1)
        if task == "regression":
            return LGBMRegressor(**common)
        return LGBMClassifier(class_weight="balanced", **common)
    if family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        common = dict(params)
        common.update(random_seed=seed, verbose=False, allow_writing_files=False)
        if task == "regression":
            return CatBoostRegressor(**common)
        return CatBoostClassifier(auto_class_weights="Balanced", **common)
    raise ValueError(f"unknown family {family!r}")


def _fit_family(family, X_train, y_train, X_val, y_val, task, seed, max_rows):
    x_train, y_train = _labelled_pair(X_train, y_train)
    x_val, y_val = _labelled_pair(X_val, y_val)
    x_train, y_train, train_sub = _subsample(x_train, y_train, max_rows, seed)
    x_val, y_val, val_sub = _subsample(x_val, y_val, max_rows, seed + 1)
    grid = GRIDS[family][task]
    rows = []
    best_model = None
    best_score = -np.inf
    for params in grid:
        row = _blank_row(family, task)
        row["params"] = json.dumps(params)
        row["train_subsampled"] = train_sub
        row["val_subsampled"] = val_sub
        try:
            model = _make_model(family, params, task, y_train, seed)
            model.fit(x_train, y_train)
            metrics = _metrics(model, x_val, y_val, task)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        row.update(metrics)
        row["ok"] = True
        rows.append(row)
        if float(metrics["val_score"]) > best_score:
            best_score = float(metrics["val_score"])
            best_model = model
    if not rows:
        rows.append(_blank_row(family, task))
    return best_model, pd.DataFrame(rows)


def _missing_family(family, task, exc):
    row = _blank_row(family, task or "")
    row["error"] = f"ImportError: {exc}"
    return None, pd.DataFrame([row])


def fit_linear_or_logistic(X_train, y_train, X_val, y_val, task=None, seed=42, max_rows=80000):
    """Ridge for a measurement target, L2 logistic regression for a class target."""
    task = task or detect_task(y_train)
    return _fit_family("linear", X_train, y_train, X_val, y_val, task, seed, max_rows)


def fit_xgboost(X_train, y_train, X_val, y_val, task=None, seed=42, max_rows=80000):
    task = task or detect_task(y_train)
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        return _missing_family("xgboost", task, exc)
    return _fit_family("xgboost", X_train, y_train, X_val, y_val, task, seed, max_rows)


def fit_lightgbm(X_train, y_train, X_val, y_val, task=None, seed=42, max_rows=80000):
    task = task or detect_task(y_train)
    try:
        import lightgbm  # noqa: F401
    except ImportError as exc:
        return _missing_family("lightgbm", task, exc)
    return _fit_family("lightgbm", X_train, y_train, X_val, y_val, task, seed, max_rows)


def fit_catboost(X_train, y_train, X_val, y_val, task=None, seed=42, max_rows=80000):
    task = task or detect_task(y_train)
    try:
        import catboost  # noqa: F401
    except ImportError as exc:
        return _missing_family("catboost", task, exc)
    return _fit_family("catboost", X_train, y_train, X_val, y_val, task, seed, max_rows)


FITTERS = {
    "linear": fit_linear_or_logistic,
    "xgboost": fit_xgboost,
    "lightgbm": fit_lightgbm,
    "catboost": fit_catboost,
}


def compare(X_train, y_train, X_val, y_val, families=("linear", "xgboost"), task=None, seed=42, max_rows=80000):
    """Fit each requested family. Returns the long comparison table and the best model of each family.

    ``families`` defaults to linear and XGBoost. Pass lightgbm and catboost only
    while the clock is still before 14:05 and the question is predictive.
    """
    task = task or detect_task(y_train)
    frames = []
    chosen = {}
    for family in families:
        if family not in FITTERS:
            raise ValueError(f"unknown family {family!r}")
        model, table = FITTERS[family](
            X_train, y_train, X_val, y_val, task=task, seed=seed, max_rows=max_rows
        )
        frames.append(table)
        if model is not None:
            chosen[family] = model
    report = pd.concat(frames, ignore_index=True)
    return report, chosen


def best_row(report: pd.DataFrame) -> pd.Series | None:
    ok = report[report["ok"] == True]  # noqa: E712
    if ok.empty:
        return None
    return ok.sort_values("val_score", ascending=False).iloc[0]


def save_bundle(path, model, preprocessor, feature_names, task, target, family, params) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "feature_names": list(feature_names),
            "task": task,
            "target": target,
            "family": family,
            "params": params,
        },
        path,
    )
    return path


def load_bundle(path):
    import joblib

    return joblib.load(path)
