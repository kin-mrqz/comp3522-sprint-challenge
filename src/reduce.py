"""PCA, and LDA when a labelled target really has two or more classes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def numeric_block(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Keep numeric columns that vary. Constants are listed and left out."""
    kept = []
    for column in columns:
        if column not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
            kept.append(column)
    block = df.loc[:, kept].apply(pd.to_numeric, errors="coerce")
    varying = [column for column in kept if block[column].nunique(dropna=True) > 1]
    dropped = [column for column in kept if column not in varying]
    return block.loc[:, varying].copy(), dropped


def run_pca(
    df: pd.DataFrame,
    columns: list[str],
    n_components: int = 2,
    scope: str = "descriptive_full_frame",
) -> dict:
    """Fit PCA on the rows in ``df``. The caller decides whether those rows are train-only.

    A plot that uses the full analysis frame must keep ``scope`` as
    ``descriptive_full_frame`` and must not be written back as a model feature.
    """
    block, dropped = numeric_block(df, columns)
    if block.shape[1] == 0 or block.shape[0] == 0:
        return {
            "ok": False,
            "reason": "no numeric columns with variation",
            "dropped_constant": dropped,
            "scope": scope,
        }
    width = int(min(n_components, block.shape[0], block.shape[1]))
    if width < 1:
        return {"ok": False, "reason": "not enough rows", "dropped_constant": dropped, "scope": scope}
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=width, random_state=42)),
        ]
    )
    scores = pipeline.fit_transform(block)
    model = pipeline.named_steps["pca"]
    loadings = pd.DataFrame(
        model.components_.T,
        index=block.columns,
        columns=[f"PC{i + 1}" for i in range(width)],
    )
    return {
        "ok": True,
        "scores": scores,
        "explained_variance_ratio": [float(value) for value in model.explained_variance_ratio_],
        "loadings": loadings,
        "scope": scope,
        "columns": list(block.columns),
        "dropped_constant": dropped,
        "pipeline": pipeline,
    }


def run_lda(
    df: pd.DataFrame,
    columns: list[str],
    target: str,
    scope: str = "descriptive_full_frame",
) -> dict:
    """LDA on the rows in ``df``. Returns a reason instead of fitting when the target is not a class label."""
    if target not in df.columns:
        return {"ok": False, "reason": f"target column {target!r} is not in the frame", "scope": scope}
    work = df.reset_index(drop=True)
    labels = work[target]
    labelled = labels.notna()
    usable = labels[labelled]
    if usable.nunique() < 2:
        return {"ok": False, "reason": "LDA needs at least two classes", "scope": scope}
    counts = usable.value_counts()
    rare = set(counts[counts < 2].index)
    keep = labelled & ~labels.isin(rare)
    if labels[keep].nunique() < 2:
        return {"ok": False, "reason": "fewer than two classes have two rows", "scope": scope}
    block, dropped = numeric_block(work.loc[keep], columns)
    if block.shape[1] < 1:
        return {
            "ok": False,
            "reason": "LDA needs at least one numeric feature",
            "scope": scope,
            "dropped_constant": dropped,
        }
    n_classes = int(labels[keep].nunique())
    width = min(n_classes - 1, block.shape[1], block.shape[0] - n_classes)
    if width < 1:
        return {"ok": False, "reason": "not enough rows for LDA", "scope": scope}
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(n_components=width)),
        ]
    )
    try:
        scores = pipeline.fit_transform(block, labels[keep])
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "scope": scope}
    return {
        "ok": True,
        "scores": np.asarray(scores),
        "scope": scope,
        "n_components": int(width),
        "classes_dropped": sorted(rare, key=str),
        "columns": list(block.columns),
        "dropped_constant": dropped,
        "pipeline": pipeline,
        "row_positions": np.flatnonzero(keep.to_numpy()),
    }
