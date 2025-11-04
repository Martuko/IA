
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances

from .utils import build_preprocessor

def _score_clusters(X_trans, labels, metric: str) -> float:
    # Some metrics need at least 2 clusters and at least 2 samples per cluster
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or X_trans.shape[0] < 3:
        return -np.inf if metric == "silhouette" else np.inf
    if metric == "silhouette":
        # For speed, you can subsample if needed
        return silhouette_score(X_trans, labels, metric="euclidean")
    elif metric == "davies_bouldin":
        return -davies_bouldin_score(X_trans, labels)  # higher is better (negated DBI)
    else:
        raise ValueError(f"Métrica no soportada: {metric}")

def _dominant_label_mapping(train_labels: np.ndarray, y_train: np.ndarray) -> Dict[int, Any]:
    mapping = {}
    for c in np.unique(train_labels):
        mask = (train_labels == c)
        vals, counts = np.unique(y_train[mask], return_counts=True)
        mapping[int(c)] = vals[np.argmax(counts)]
    return mapping

def _assign_nearest_center(X_trans, centers: np.ndarray) -> np.ndarray:
    # Assign each sample to nearest center by Euclidean distance
    d = pairwise_distances(X_trans, centers, metric="euclidean")
    return np.argmin(d, axis=1)

def run_clustering(cfg: Dict[str, Any], X_train, y_train, X_test, y_test, num_cols, cat_cols):
    score_metric = cfg["clustering"].get("score_metric", "silhouette")
    results = []

    # Preprocessor (fit only on X_train to avoid data leakage)
    pre = build_preprocessor(num_cols, cat_cols)
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)

    # 1) KMeans + KMeans++ (init random vs k-means++)
    for params in cfg["clustering"]["kmeans"]["configs"]:
        km = KMeans(
            n_clusters=params["n_clusters"],
            init=params.get("init", "k-means++"),
            n_init=params.get("n_init", 10),
            max_iter=params.get("max_iter", 300),
            random_state=cfg.get("random_state", 42),
        )
        # Fit on training
        train_labels = km.fit_predict(Xtr)
        score = _score_clusters(Xtr, train_labels, metric=score_metric)

        # Map dominant Y label per cluster (using training)
        dom_map = _dominant_label_mapping(train_labels, np.array(y_train))

        # Apply to test set
        test_labels = km.predict(Xte)
        # Convert clusters -> dominant class
        y_pred_dom = np.array([dom_map.get(int(c), None) for c in test_labels])
        acc_dom = np.mean(y_pred_dom == np.array(y_test))

        results.append({
            "technique": "KMeans" if params.get("init", "k-means++") == "random" else "KMeans++",
            "name": params["name"],
            "n_clusters": params["n_clusters"],
            "params": params,
            "score_metric": score_metric,
            "train_score": float(score),
            "test_dominant_label_acc": float(acc_dom),
        })

    # 2) MeanShift (fit on training, predict by nearest center)
    for params in cfg["clustering"]["meanshift"]["configs"]:
        # MeanShift does not accept sparse -> densify
        Xtr_dense = Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr
        Xte_dense = Xte.toarray() if hasattr(Xte, "toarray") else Xte

        bandwidth = params.get("bandwidth", None)
        if bandwidth is None:
            q = params.get("quantile", 0.2)
            n_samples = params.get("n_samples", None)
            bandwidth = estimate_bandwidth(Xtr_dense, quantile=q, n_samples=n_samples, random_state=cfg.get("random_state", 42))

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=params.get("bin_seeding", True), n_jobs=None)
        train_labels = ms.fit_predict(Xtr_dense)
        score = _score_clusters(Xtr_dense, train_labels, metric=score_metric)

        centers = ms.cluster_centers_
        test_labels = _assign_nearest_center(Xte_dense, centers)

        dom_map = _dominant_label_mapping(train_labels, np.array(y_train))
        y_pred_dom = np.array([dom_map.get(int(c), None) for c in test_labels])
        acc_dom = np.mean(y_pred_dom == np.array(y_test))

        results.append({
            "technique": "MeanShift",
            "name": params["name"],
            "bandwidth": float(bandwidth) if bandwidth is not None else None,
            "params": params,
            "score_metric": score_metric,
            "train_score": float(score),
            "test_dominant_label_acc": float(acc_dom),
        })

    df_results = pd.DataFrame(results).sort_values(by="train_score", ascending=False).reset_index(drop=True)
    # Top-3 según métrica en train
    top3 = df_results.head(3).copy()
    return df_results, top3, pre
