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


def summarize_clustering(labels):
    """
    Summarize cluster and noise counts.
    
    Returns:
        n_clusters (int): Number of discovered clusters (excluding noise)
        n_noise (int): Number of noise points
        noise_ratio (float): Ratio of noise points to total samples
    """
    n_samples = len(labels)
    unique_labels = set(labels)
    has_noise = -1 in unique_labels
    n_clusters = len(unique_labels) - (1 if has_noise else 0)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / n_samples if n_samples > 0 else 0.0
    return n_clusters, n_noise, noise_ratio


if __name__ == "__main__":
    print("=== Testing DBSCAN Multi-Metric Execution on kr-vs-kp ===")
    X, y, feats = load_dataset("kr-vs-kp")
    
    test_configs = [
        {"metric": "euclidean", "algorithm": "auto", "eps": 2.0, "min_samples": 5},
        {"metric": "manhattan", "algorithm": "ball_tree", "eps": 4.0, "min_samples": 5},
        {"metric": "cosine", "algorithm": "brute", "eps": 0.05, "min_samples": 5},
    ]
    
    for cfg in test_configs:
        labels, elapsed = run_dbscan(
            X, eps=cfg["eps"], min_samples=cfg["min_samples"],
            metric=cfg["metric"], algorithm=cfg["algorithm"]
        )
        n_clusters, n_noise, noise_ratio = summarize_clustering(labels)
        print(f"[{cfg['metric'].upper():9s} | {cfg['algorithm']:9s} | eps={cfg['eps']}]: "
              f"Clusters={n_clusters:2d}, Noise={n_noise:4d} ({noise_ratio*100:5.1f}%), Time={elapsed*1000:6.1f}ms")
    
    print("\nVerifying that cosine with kd_tree raises expected validation error...")
    try:
        run_dbscan(X, eps=0.05, min_samples=5, metric="cosine", algorithm="kd_tree")
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly caught: {e}")
