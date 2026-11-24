"""Native-graph decision, geographic edges, and leakage-safe neighbour features.

A feature-space kNN graph is not built here. Neighbour label rates read training
labels only. Degree and neighbour means use an unweighted graph so the columns
stay easy to explain. Edge weights are still stored on the edge table.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .paths import GATE_PATH, ensure_dirs

KINDS = ("edge_list", "coordinates", "shared_id")


def write_gate(
    native_graph: bool | None,
    reason: str,
    kind: str | None = None,
    evidence: str = "",
    path: Path | None = None,
) -> Path:
    """Record the team's yes or no. ``None`` means the decision is not made yet."""
    if native_graph not in (True, False, None):
        raise ValueError("native_graph must be true, false, or null")
    if kind not in (None, "") and kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    ensure_dirs()
    path = Path(path) if path else GATE_PATH
    payload = {
        "native_graph": native_graph,
        "reason": reason,
        "kind": kind or None,
        "evidence": evidence,
        "decided_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_gate(path: Path | None = None) -> dict:
    path = Path(path) if path else GATE_PATH
    if not path.exists():
        return {
            "native_graph": None,
            "reason": "graph gate has not been written",
            "kind": None,
            "evidence": "",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def gate_is_yes(payload: dict | None = None) -> bool:
    if payload is None:
        payload = read_gate()
    return payload.get("native_graph") is True


def pairwise_haversine_km(lat, lon) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if len(lat) != len(lon):
        raise ValueError("lat and lon must have the same length")
    if len(lat) > 1500:
        raise ValueError(
            "Refusing a dense distance matrix above 1500 nodes. "
            "Aggregate to entities, or pass an edge list."
        )
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _components(n: int, src: np.ndarray, dst: np.ndarray):
    parent = list(range(n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in zip(src, dst):
        ra, rb = find(int(left)), find(int(right))
        if ra != rb:
            parent[rb] = ra
    labels = [find(node) for node in range(n)]
    remap = {old: new for new, old in enumerate(sorted(set(labels)))}
    return len(remap), np.asarray([remap[label] for label in labels], dtype=int)


def edges_from_distance(dist, k: int = 4, max_dist: float = 8.0) -> dict:
    """Mutual k-nearest neighbours, plus every pair closer than ``max_dist``.

    Weights are a Gaussian kernel. Sigma is the median edge length, fixed from
    distance before any label is read.
    """
    dist = np.asarray(dist, dtype=float).copy()
    n = dist.shape[0]
    if dist.shape != (n, n):
        raise ValueError("distance matrix must be square")
    np.fill_diagonal(dist, np.inf)
    connect = np.zeros((n, n), dtype=bool)
    if k > 0 and n > 1:
        kk = min(int(k), n - 1)
        knn = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        is_knn = np.zeros((n, n), dtype=bool)
        rows = np.repeat(np.arange(n), kk)
        is_knn[rows, knn.reshape(-1)] = True
        connect |= is_knn & is_knn.T
    close = np.isfinite(dist) & (dist <= max_dist)
    np.fill_diagonal(close, False)
    connect |= close
    connect = np.logical_or(connect, connect.T)
    np.fill_diagonal(connect, False)
    src, dst = np.where(np.triu(connect, 1))
    distances = dist[src, dst].astype(float)
    if len(distances) == 0:
        return {
            "src": src.astype(np.int64),
            "dst": dst.astype(np.int64),
            "distance_km": distances,
            "weight": np.array([], dtype=float),
            "sigma_km": np.nan,
            "n_components": int(n),
            "component": np.arange(n),
            "n_nodes": int(n),
            "n_edges": 0,
        }
    finite = distances[np.isfinite(distances)]
    sigma = float(np.median(finite)) if len(finite) else float("nan")
    if not np.isfinite(sigma) or sigma <= 0:
        weights = np.ones(len(distances), dtype=float)
        sigma = float(np.mean(finite)) if len(finite) else 1.0
    else:
        weights = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
    n_components, component = _components(n, src, dst)
    return {
        "src": src.astype(np.int64),
        "dst": dst.astype(np.int64),
        "distance_km": distances,
        "weight": weights.astype(float),
        "sigma_km": float(sigma),
        "n_components": int(n_components),
        "component": component,
        "n_nodes": int(n),
        "n_edges": int(len(src)),
    }


def edges_from_coordinates(lat, lon, k: int = 4, max_km: float = 8.0) -> dict:
    """Geographic edges on rows with finite coordinates. Other rows stay isolated."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    positions = np.flatnonzero(ok)
    if len(positions) == 0:
        raise ValueError("no finite coordinates")
    packed = edges_from_distance(
        pairwise_haversine_km(lat[positions], lon[positions]),
        k=k,
        max_dist=max_km,
    )
    packed["src"] = positions[packed["src"]].astype(np.int64)
    packed["dst"] = positions[packed["dst"]].astype(np.int64)
    component = np.full(len(lat), -1, dtype=int)
    component[positions] = packed["component"]
    packed["component"] = component
    packed["n_nodes"] = int(len(lat))
    packed["n_dropped_missing_coordinates"] = int((~ok).sum())
    return packed


def edge_table(packed: dict, node_ids=None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "src": packed["src"],
            "dst": packed["dst"],
            "distance_km": packed["distance_km"],
            "weight": packed["weight"],
        }
    )
    if node_ids is not None and len(frame):
        ids = pd.Series(node_ids).astype(str).reset_index(drop=True)
        frame["src_id"] = ids.iloc[frame["src"].to_numpy()].to_numpy()
        frame["dst_id"] = ids.iloc[frame["dst"].to_numpy()].to_numpy()
    return frame


def encode_edge_list(edge_df: pd.DataFrame, node_ids, src_col: str = "src", dst_col: str = "dst"):
    """Map an edge list of entity ids onto positions in ``node_ids``."""
    ids = pd.Series(node_ids).astype(str).reset_index(drop=True)
    lookup = {key: position for position, key in enumerate(ids)}
    src, dst = [], []
    dropped = 0
    for left, right in zip(edge_df[src_col].astype(str), edge_df[dst_col].astype(str)):
        if left not in lookup or right not in lookup or left == right:
            dropped += 1
            continue
        src.append(lookup[left])
        dst.append(lookup[right])
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), int(dropped)


def _adjacency(src: np.ndarray, dst: np.ndarray, n: int):
    if len(src) == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float64)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    matrix.sum_duplicates()
    matrix.data = np.ones_like(matrix.data)
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def graph_features(
    src,
    dst,
    n: int,
    X=None,
    y=None,
    train_mask=None,
    feature_names=None,
    max_mean_cols: int = 30,
) -> pd.DataFrame:
    """Degree, two-hop count, train-only neighbour label rate, and neighbour means.

    ``X`` must not contain the label. Isolated nodes get a neighbour-label rate
    of -1, which means "no labelled neighbour", not a rate of zero.
    """
    src = np.asarray(src, dtype=np.int64).ravel()
    dst = np.asarray(dst, dtype=np.int64).ravel()
    if len(src) != len(dst):
        raise ValueError("src and dst must have the same length")
    if len(src):
        if src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n:
            raise ValueError("an edge endpoint is outside the node range")
        left = np.minimum(src, dst)
        right = np.maximum(src, dst)
        keep = left != right
        pairs = (
            np.unique(np.stack([left[keep], right[keep]], axis=1), axis=0)
            if keep.any()
            else np.zeros((0, 2), dtype=np.int64)
        )
        src, dst = pairs[:, 0], pairs[:, 1]
    matrix = _adjacency(src, dst, n)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    out = {
        "g_degree": degree.astype(np.float32),
        "g_has_edges": (degree > 0).astype(np.float32),
        "g_log_degree": np.log1p(degree).astype(np.float32),
        "g_two_hop": np.asarray(matrix @ degree.reshape(-1, 1)).ravel().astype(np.float32),
    }
    if y is not None:
        if train_mask is None:
            raise ValueError(
                "Neighbour label rates need train_mask. Refusing to read every label."
            )
        labels = np.asarray(y, dtype=float)
        mask = np.asarray(train_mask, dtype=bool)
        if len(labels) != n or len(mask) != n:
            raise ValueError("y and train_mask must have one entry per node")
        known = mask & np.isfinite(labels)
        label_vector = np.where(known, np.nan_to_num(labels), 0.0)
        count_vector = known.astype(np.float64)
        labelled_neighbours = np.asarray(matrix @ count_vector.reshape(-1, 1)).ravel()
        bad_neighbours = np.asarray(matrix @ label_vector.reshape(-1, 1)).ravel()
        rate = np.full(n, -1.0, dtype=np.float64)
        np.divide(bad_neighbours, labelled_neighbours, out=rate, where=labelled_neighbours > 0)
        out["g_nb_train_labelled"] = labelled_neighbours.astype(np.float32)
        out["g_nb_bad_count"] = bad_neighbours.astype(np.float32)
        out["g_nb_bad_rate"] = rate.astype(np.float32)
    if X is not None:
        block = np.asarray(X, dtype=np.float64)
        if block.ndim != 2 or block.shape[0] != n:
            raise ValueError("X must have one row per node")
        names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(block.shape[1])]
        if len(names) < block.shape[1]:
            raise ValueError("feature_names is shorter than X")
        chosen = list(range(min(block.shape[1], max_mean_cols)))
        values = block[:, chosen]
        finite = np.isfinite(values)
        filled = np.where(finite, values, 0.0)
        counts = np.asarray(matrix @ finite.astype(np.float64))
        sums = np.asarray(matrix @ filled)
        means = np.full(values.shape, np.nan, dtype=np.float64)
        np.divide(sums, counts, out=means, where=counts > 0)
        for offset, column in enumerate(chosen):
            safe = str(names[column]).replace(" ", "_")
            out[f"g_nbmean_{safe}"] = means[:, offset].astype(np.float32)
            out[f"g_nbdiff_{safe}"] = (values[:, offset] - means[:, offset]).astype(np.float32)
    frame = pd.DataFrame(out)
    frame.index.name = "node"
    return frame


def map_node_features(node_features: pd.DataFrame, row_node_positions) -> pd.DataFrame:
    """Repeat node-level graph features onto row-level observations."""
    positions = np.asarray(row_node_positions, dtype=int)
    return node_features.iloc[positions].reset_index(drop=True)
