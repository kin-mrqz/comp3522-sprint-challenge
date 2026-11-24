"""All-station risk score. Hour, month, and station only.

This file does not write outputs/answers.md or the sprint deck.
The three-station weather score stays where it is.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .paths import ANALYSIS_PATH, OUTPUTS, TABLES

EXT = OUTPUTS / "extension"
FIGURES = EXT / "figures"
FPR_CAPS = (0.10, 0.20, 0.30)
SLIDE_FPR = 0.20
SEASON_OF = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def _detection(y_poor: np.ndarray, stay: np.ndarray) -> dict:
    y = np.asarray(y_poor, dtype=bool)
    called = np.asarray(stay, dtype=bool)
    n_poor = int(y.sum())
    n_safe = int((~y).sum())
    caught = int((called & y).sum())
    false_pos = int((called & ~y).sum())
    return {
        "n": int(len(y)),
        "n_poor": n_poor,
        "tpr": float(caught / n_poor) if n_poor else float("nan"),
        "fpr": float(false_pos / n_safe) if n_safe else float("nan"),
        "go_share": float((~called).mean()) if len(y) else float("nan"),
    }


def _curve(y_poor: np.ndarray, proba: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        row = _detection(y_poor, proba >= threshold)
        row["threshold"] = float(threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def _pick_threshold(curve: pd.DataFrame, cap: float) -> tuple[pd.Series, bool]:
    under = curve.loc[curve["fpr"] <= cap + 1e-12]
    if under.empty:
        best = curve.sort_values(["fpr", "tpr"], ascending=[True, False]).iloc[0]
        return best, False
    best = under.sort_values(["tpr", "go_share", "threshold"], ascending=[False, False, False]).iloc[0]
    return best, True


def _upper_edges(values: np.ndarray, bins: int) -> list[float]:
    ordered = np.sort(np.asarray(values, dtype=float))
    n = len(ordered)
    edges = []
    for i in range(bins):
        last = min(n - 1, max(0, (i + 1) * n // bins - 1))
        edges.append(float(ordered[last]))
    return edges


def _assign_scores(proba: np.ndarray, threshold: float, go_edges: list[float], stay_edges: list[float]) -> np.ndarray:
    scores = np.empty(len(proba), dtype=int)
    go_mask = proba < threshold
    go_values = proba[go_mask]
    go_scores = np.full(go_values.shape, 5, dtype=int)
    for i, edge in enumerate(go_edges[:4]):
        go_scores = np.where(go_values <= edge, np.minimum(go_scores, i + 1), go_scores)
    scores[go_mask] = go_scores
    stay_values = proba[~go_mask]
    stay_scores = np.full(stay_values.shape, 10, dtype=int)
    for i, edge in enumerate(stay_edges[:4]):
        stay_scores = np.where(stay_values <= edge, np.minimum(stay_scores, i + 6), stay_scores)
    scores[~go_mask] = stay_scores
    return scores


def _pct(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{100 * value:.1f}%"


def _notes(result: dict) -> str:
    locked = result["locked"]
    train = result["train"]
    test = result["test"]
    old = result["old_rule_test"]
    weights = result["weights"]
    high = weights.head(3)
    low = weights.tail(3).iloc[::-1]
    high_bits = ", ".join(f"{row.feature} {row.coef:+.2f}" for row in high.itertuples())
    low_bits = ", ".join(f"{row.feature} {row.coef:+.2f}" for row in low.itertuples())
    station_w = weights.loc[weights["feature"].str.startswith("cat__station_")].copy()
    station_w["name"] = station_w["feature"].str.replace("cat__station_", "", regex=False)
    up_stations = ", ".join(f"{row.name} {row.coef:+.2f}" for row in station_w.head(3).itertuples())
    down_stations = ", ".join(f"{row.name} {row.coef:+.2f}" for row in station_w.tail(3).iloc[::-1].itertuples())
    caught = result["by_station"].head(3)
    caught_bits = ", ".join(
        f"{row.level} TPR {_pct(float(row.tpr))} FPR {_pct(float(row.fpr))}" for row in caught.itertuples()
    )
    bins = result["bins"]
    test_bins = bins.loc[:, ["score", "test_n", "test_poor_share"]]
    bin_bits = ", ".join(
        f"{int(row.score)} {_pct(float(row.test_poor_share))} (n={int(row.test_n):,})"
        for row in test_bins.itertuples()
    )
    return "\n".join(
        [
            "# All-station score",
            "",
            "This note is separate from outputs/answers.md and the sprint deck.",
            "The score uses hour, month, and station at all 18 stations. It does not use temperature or rainfall.",
            "",
            f"Reference case for the weights: {result['reference']}.",
            f"Largest upward weights: {high_bits}.",
            f"Largest downward weights: {low_bits}.",
            f"Largest upward station weights versus the reference station: {up_stations}.",
            f"Largest downward station weights: {down_stations}.",
            f"On 2024–2025 the same cutoff catches the most poor hours at {caught_bits}. Those stations also spend more safe hours indoors.",
            "",
            f"Stay in when the probability is at least {locked['threshold']:.4f}.",
            "That cutoff was chosen on 2021–2023: the highest share of AQHI ≥ 7 hours caught with a false-positive rate at or under 20%.",
            f"Fit years: TPR {_pct(train['tpr'])}, FPR {_pct(train['fpr'])}, outdoors {_pct(train['go_share'])}.",
            f"2024–2025: TPR {_pct(test['tpr'])}, FPR {_pct(test['fpr'])}, outdoors {_pct(test['go_share'])}.",
            f"Old month-and-hour flags on the same hours: TPR {_pct(old['tpr'])}, FPR {_pct(old['fpr'])}, outdoors {_pct(old['go_share'])}.",
            "",
            "Scores 1–5 are the hours below that cutoff, split into five equal training groups. 1 is the safest. 5 is the last go-outside score.",
            "Scores 6–10 are the hours at or above the cutoff. 6 is the first stay-in score. 10 is the highest risk.",
            "The bin edges were frozen on 2021–2023 and applied to 2024–2025.",
            "1–5 go outside, 6–10 stay in.",
            "",
            f"Poor-hour share on 2024–2025 by score: {bin_bits}.",
            "",
        ]
    )


def run_extension() -> dict:
    EXT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(ANALYSIS_PATH)
    frame = frame.loc[frame["year"].between(2021, 2025)].copy()
    frame["hour"] = frame["hour"].astype(int)
    frame["month"] = frame["month"].astype(int)
    frame["season"] = frame["month"].map(SEASON_OF)
    train = frame.loc[frame["year"].between(2021, 2023)].copy()
    test = frame.loc[frame["year"].between(2024, 2025)].copy()
    features = ["hour", "month", "station"]
    model = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                            features,
                        )
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )
    model.fit(train[features], train["poor"].astype(int))
    positive = list(model.named_steps["clf"].classes_).index(1)
    train["proba"] = model.predict_proba(train[features])[:, positive]
    test["proba"] = model.predict_proba(test[features])[:, positive]

    encoder = model.named_steps["pre"].named_transformers_["cat"]
    dropped = [cats[0] for cats in encoder.categories_]
    reference = f"hour {int(dropped[0])}, month {int(dropped[1])}, station {dropped[2]}"
    names = model.named_steps["pre"].get_feature_names_out()
    weights = pd.DataFrame({"feature": names, "coef": model.named_steps["clf"].coef_.ravel()})
    weights = weights.sort_values("coef", ascending=False)
    weights.to_csv(EXT / "coefficients.csv", index=False)

    thresholds = np.unique(
        np.concatenate([[0.0, 1.0], np.quantile(train["proba"].to_numpy(), np.linspace(0, 1, 401))])
    )
    train_curve = _curve(train["poor"].to_numpy(), train["proba"].to_numpy(), thresholds)
    test_curve = _curve(test["poor"].to_numpy(), test["proba"].to_numpy(), thresholds)
    caps = []
    for cap in FPR_CAPS:
        chosen, met = _pick_threshold(train_curve, cap)
        test_at = test_curve.iloc[(test_curve["threshold"] - chosen["threshold"]).abs().argmin()]
        caps.append(
            {
                "cap": cap,
                "met_on_train": bool(met),
                "threshold": float(chosen["threshold"]),
                "train_tpr": float(chosen["tpr"]),
                "train_fpr": float(chosen["fpr"]),
                "train_go_share": float(chosen["go_share"]),
                "test_tpr": float(test_at["tpr"]),
                "test_fpr": float(test_at["fpr"]),
                "test_go_share": float(test_at["go_share"]),
            }
        )
    pd.DataFrame(caps).to_csv(EXT / "threshold_caps.csv", index=False)
    locked = next(row for row in caps if abs(row["cap"] - SLIDE_FPR) < 1e-9)
    threshold = locked["threshold"]

    rule = json.loads((TABLES / "aqhi_rule.json").read_text(encoding="utf-8"))
    old_stay = test["month"].isin(rule["bad_months"]) | test["hour"].isin(rule["bad_hours"]) | test["roadside"]
    old_rule_test = _detection(test["poor"].to_numpy(), old_stay.to_numpy())

    go_edges = _upper_edges(train.loc[train["proba"] < threshold, "proba"].to_numpy(), 5)
    stay_edges = _upper_edges(train.loc[train["proba"] >= threshold, "proba"].to_numpy(), 5)
    train["score"] = _assign_scores(train["proba"].to_numpy(), threshold, go_edges, stay_edges)
    test["score"] = _assign_scores(test["proba"].to_numpy(), threshold, go_edges, stay_edges)
    bin_rows = []
    for score in range(1, 11):
        train_part = train.loc[train["score"].eq(score)]
        test_part = test.loc[test["score"].eq(score)]
        edges = go_edges if score <= 5 else stay_edges
        edge_index = score - 1 if score <= 5 else score - 6
        bin_rows.append(
            {
                "score": score,
                "side": "go" if score <= 5 else "stay",
                "p_high": edges[edge_index],
                "train_n": int(len(train_part)),
                "train_poor_share": float(train_part["poor"].mean()) if len(train_part) else float("nan"),
                "test_n": int(len(test_part)),
                "test_poor_share": float(test_part["poor"].mean()) if len(test_part) else float("nan"),
            }
        )
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(EXT / "score_bins.csv", index=False)

    test["stay"] = test["proba"] >= threshold
    pieces = []
    for column in ("season", "station"):
        grouped = test.groupby(column, sort=False)
        for label, part in grouped:
            row = _detection(part["poor"].to_numpy(), part["stay"].to_numpy())
            row["slice"] = column
            row["level"] = str(label)
            pieces.append(row)
    by_slice = pd.DataFrame(pieces)
    by_slice.loc[by_slice["slice"].eq("station")].drop(columns="slice").sort_values("tpr", ascending=False).to_csv(
        EXT / "by_station.csv", index=False
    )
    season_order = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
    seasons = by_slice.loc[by_slice["slice"].eq("season")].copy()
    seasons["_order"] = seasons["level"].map(season_order)
    seasons.sort_values("_order").drop(columns=["slice", "_order"]).to_csv(EXT / "by_season.csv", index=False)

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25, "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(test_curve["fpr"], test_curve["tpr"], color="#0369a1", label="2024–2025")
    ax.plot(train_curve["fpr"], train_curve["tpr"], color="#94a3b8", label="2021–2023")
    for row, marker in zip(caps, ["o", "s", "D"]):
        ax.scatter(row["test_fpr"], row["test_tpr"], marker=marker, s=40, zorder=3, label=f"{int(row['cap'] * 100)}% cap")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Share of AQHI ≥ 7 hours caught")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("All 18 stations, cutoff chosen on 2021–2023")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "threshold_curve.png", dpi=140)
    plt.close(fig)

    station_weights = weights.loc[weights["feature"].str.startswith("cat__station_")].copy()
    station_weights["station"] = station_weights["feature"].str.replace("cat__station_", "", regex=False)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ordered = station_weights.sort_values("coef")
    colors = np.where(ordered["coef"] >= 0, "#b45309", "#0369a1")
    ax.barh(ordered["station"], ordered["coef"], color=colors)
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_xlabel(f"Log-odds versus {dropped[2]}")
    ax.set_title("Station weights, hour and month held fixed")
    fig.tight_layout()
    fig.savefig(FIGURES / "station_weights.png", dpi=140)
    plt.close(fig)

    result = {
        "reference": reference,
        "weights": weights,
        "locked": locked,
        "train": {"tpr": locked["train_tpr"], "fpr": locked["train_fpr"], "go_share": locked["train_go_share"], "n": int(len(train))},
        "test": _detection(test["poor"].to_numpy(), test["stay"].to_numpy()),
        "old_rule_test": old_rule_test,
        "bins": bins,
        "by_station": pd.read_csv(EXT / "by_station.csv"),
    }
    (EXT / "notes.md").write_text(_notes(result), encoding="utf-8")
    print((EXT / "notes.md").read_text(encoding="utf-8"))
    return result
