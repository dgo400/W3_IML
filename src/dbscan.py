"""
DBSCAN Clustering implementation and evaluation for Work 3 - IML.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
groups points that are closely packed together (points with many nearby neighbors)
and marks points in low-density regions as outliers/noise.

Key Hyperparameters:
- eps (epsilon): The maximum distance between two samples for one to be considered
  in the neighborhood of the other.
- min_samples: The number of samples in a neighborhood for a point to be considered
  a core point (including the point itself).
- metric: The distance metric used (e.g., 'euclidean', 'cosine', 'manhattan').
- algorithm: The nearest neighbor search algorithm ('auto', 'ball_tree', 'kd_tree', 'brute').
"""

import sys
import os
import time
import numpy as np

from sklearn.cluster import DBSCAN

# Support execution both from root and src directory
try:
    from src.preprocessing import load_dataset
except ModuleNotFoundError:
    from preprocessing import load_dataset


VALID_ALGORITHMS = ("auto", "ball_tree", "kd_tree", "brute")
VALID_METRICS = ("euclidean", "cosine", "manhattan")


def validate_dbscan_params(metric, algorithm):
    """
    Validate metric and algorithm compatibility.
    
    Note: In scikit-learn, cosine distance violates the triangle inequality,
    so tree-based partitioning (KDTree and BallTree) is unsupported.
    Cosine distance is only compatible with algorithm='brute'.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {VALID_METRICS}.")
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from {VALID_ALGORITHMS}.")

    if metric == "cosine" and algorithm != "brute":
        raise ValueError(
            f"Incompatible configuration: metric='cosine' only works with algorithm='brute', "
            f"but received algorithm='{algorithm}'."
        )


def run_dbscan(X, eps=0.5, min_samples=5, metric="euclidean", algorithm="auto"):
    """
    Execute DBSCAN clustering with parameter validation and timing.
    
    Args:
        X: np.ndarray of shape (n_samples, n_features)
        eps: neighborhood radius float
        min_samples: minimum neighborhood samples int
        metric: distance metric string ('euclidean', 'cosine', 'manhattan')
        algorithm: search algorithm string ('auto', 'ball_tree', 'kd_tree', 'brute')
        
    Returns:
        labels: cluster labels (-1 indicates noise)
        elapsed_sec: execution time in seconds
    """
    validate_dbscan_params(metric, algorithm)
    start_time = time.perf_counter()
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm
    )
    db.fit(X)
    elapsed_sec = time.perf_counter() - start_time
    return db.labels_, elapsed_sec


def compute_purity(y_true, labels, ignore_noise=True):
    """
    Compute clustering Purity: sum(max_class_per_cluster) / N.
    """
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    if ignore_noise:
        mask = (labels != -1)
        if not np.any(mask):
            return 0.0
        y_true = y_true[mask]
        labels = labels[mask]
    if len(y_true) == 0:
        return 0.0

    correct = 0
    for cluster_id in np.unique(labels):
        cluster_mask = (labels == cluster_id)
        _, counts = np.unique(y_true[cluster_mask], return_counts=True)
        correct += counts.max()
    return float(correct / len(y_true))


def compute_f_measure(y_true, labels, ignore_noise=True):
    """
    Compute clustering F-measure: majority class mapped F1 score.
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    if ignore_noise:
        mask = (labels != -1)
        if not np.any(mask):
            return 0.0
        y_true = y_true[mask]
        labels = labels[mask]
    if len(y_true) == 0:
        return 0.0

    mapped_pred = np.empty_like(y_true)
    for cluster_id in np.unique(labels):
        cluster_mask = (labels == cluster_id)
        classes, counts = np.unique(y_true[cluster_mask], return_counts=True)
        majority_class = classes[np.argmax(counts)]
        mapped_pred[cluster_mask] = majority_class

    return float(f1_score(y_true, mapped_pred, average="weighted", zero_division=0))


def evaluate_clustering(X, y_true, labels, metric="euclidean"):
    """
    Compute comprehensive internal and external clustering validation metrics.
    
    Internal:
      - silhouette: separation & cohesion (range [-1, 1], higher is better)
      - davies_bouldin: cluster similarity (lower is better, >= 0)
      - calinski_harabasz: variance ratio (higher is better)
    External:
      - ari (Adjusted Rand Index): agreement with ground truth adjusted for chance
      - purity: fraction of dominant ground-truth class per cluster
      - f_measure: weighted F1-measure under optimal majority mapping
    """
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score,
        adjusted_rand_score,
    )

    n_samples = len(labels)
    unique_labels = set(labels)
    has_noise = -1 in unique_labels
    n_clusters = len(unique_labels) - (1 if has_noise else 0)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / n_samples if n_samples > 0 else 0.0

    results = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": round(noise_ratio, 4),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "ari": None,
        "purity": None,
        "f_measure": None,
    }

    # External metrics
    results["ari"] = round(float(adjusted_rand_score(y_true, labels)), 4)
    results["purity"] = round(compute_purity(y_true, labels, ignore_noise=True), 4)
    results["f_measure"] = round(compute_f_measure(y_true, labels, ignore_noise=True), 4)

    # Internal metrics (require at least 2 non-noise clusters and >= 2 points)
    clustered_mask = labels != -1
    clustered_count = np.sum(clustered_mask)

    if n_clusters >= 2 and clustered_count >= 2:
        X_clustered = X[clustered_mask]
        labels_clustered = labels[clustered_mask]

        # Ensure more than 1 distinct label exists among non-noise points
        if len(set(labels_clustered)) >= 2:
            try:
                # Use specified metric for silhouette when feasible
                sil_metric = metric if metric in ("euclidean", "cosine", "manhattan") else "euclidean"
                results["silhouette"] = round(
                    float(silhouette_score(X_clustered, labels_clustered, metric=sil_metric)), 4
                )
            except Exception:
                results["silhouette"] = None

            try:
                results["davies_bouldin"] = round(
                    float(davies_bouldin_score(X_clustered, labels_clustered)), 4
                )
            except Exception:
                results["davies_bouldin"] = None

            try:
                results["calinski_harabasz"] = round(
                    float(calinski_harabasz_score(X_clustered, labels_clustered)), 2
                )
            except Exception:
                results["calinski_harabasz"] = None

    return results


def run_experiments(dataset_names=("kr-vs-kp", "horse-colic"), output_csv="results/dbscan_results.csv"):
    """
    Run systematic DBSCAN parameter sweep across datasets, metrics, and algorithms.
    """
    import pandas as pd

    records = []

    # Dataset-specific parameter grids
    grid = {
        "kr-vs-kp": {
            "metrics": ["euclidean", "manhattan", "cosine"],
            "algorithms": ["auto", "ball_tree", "kd_tree", "brute"],
            "min_samples_list": [3, 5, 10],
            "eps_map": {
                "euclidean": [1.4, 1.8, 2.0, 2.4],
                "manhattan": [2.0, 4.0, 6.0, 8.0],
                "cosine": [0.03, 0.05, 0.08, 0.12],
            },
        },
        "horse-colic": {
            "metrics": ["euclidean", "manhattan", "cosine"],
            "algorithms": ["auto", "ball_tree", "kd_tree", "brute"],
            "min_samples_list": [3, 5, 8],
            "eps_map": {
                "euclidean": [2.5, 3.2, 3.8, 4.5],
                "manhattan": [8.0, 11.0, 14.0, 17.0],
                "cosine": [0.15, 0.22, 0.28, 0.35],
            },
        },
    }

    for ds_name in dataset_names:
        print(f"\n==================================================")
        print(f" Running DBSCAN Experiments on: {ds_name.upper()}")
        print(f"==================================================")
        X, y_true, feats = load_dataset(ds_name)
        cfg = grid[ds_name]

        for metric in cfg["metrics"]:
            # Cosine only works with brute algorithm
            valid_algos = ["brute"] if metric == "cosine" else cfg["algorithms"]

            for algo in valid_algos:
                for min_samples in cfg["min_samples_list"]:
                    for eps in cfg["eps_map"][metric]:
                        try:
                            labels, elapsed_sec = run_dbscan(
                                X, eps=eps, min_samples=min_samples,
                                metric=metric, algorithm=algo
                            )
                            metrics_dict = evaluate_clustering(X, y_true, labels, metric=metric)
                            record = {
                                "dataset": ds_name,
                                "metric": metric,
                                "algorithm": algo,
                                "eps": eps,
                                "min_samples": min_samples,
                                "time_ms": round(elapsed_sec * 1000, 2),
                                **metrics_dict,
                            }
                            records.append(record)
                        except Exception as e:
                            # E.g. KDTree does not support certain metrics
                            pass

    df_results = pd.DataFrame(records)

    # Save to CSV
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_results.to_csv(output_csv, index=False)
        print(f"\nSaved {len(df_results)} experiment runs to: {output_csv}")

    return df_results


def print_summary_table(df_results, dataset_name, metric):
    """
    Print a concise terminal comparison table for a given dataset and metric.
    """
    subset = df_results[(df_results["dataset"] == dataset_name) & (df_results["metric"] == metric)]
    if subset.empty:
        return
    # Select default algorithm (auto or brute)
    target_algo = "brute" if metric == "cosine" else "auto"
    view = subset[subset["algorithm"] == target_algo][
        ["eps", "min_samples", "n_clusters", "n_noise", "noise_ratio", "silhouette", "ari", "purity", "time_ms"]
    ]
    print(f"\n--- {dataset_name} | Metric: {metric.upper()} (algorithm={target_algo}) ---")
    print(view.to_string(index=False))


if __name__ == "__main__":
    results_df = run_experiments(dataset_names=["kr-vs-kp", "horse-colic"])

    # Print summary highlights
    for ds in ["kr-vs-kp", "horse-colic"]:
        for m in ["euclidean", "cosine", "manhattan"]:
            print_summary_table(results_df, ds, m)
