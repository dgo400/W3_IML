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


def run_dbscan(X, eps=0.5, min_samples=5, metric="euclidean", algorithm="auto"):
    """
    Execute DBSCAN clustering with parameter validation and timing.
    
    Args:
        X: np.ndarray of shape (n_samples, n_features)
        eps: neighborhood radius float
        min_samples: minimum neighborhood samples int
        metric: distance metric string
        algorithm: search algorithm string
        
    Returns:
        labels: cluster labels (-1 indicates noise)
        elapsed_sec: execution time in seconds
    """
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
    print("Testing DBSCAN baseline with Euclidean metric...")
    X, y, feats = load_dataset("kr-vs-kp")
    labels, elapsed = run_dbscan(X, eps=2.0, min_samples=5, metric="euclidean", algorithm="auto")
    n_clusters, n_noise, noise_ratio = summarize_clustering(labels)
    print(f"Dataset: kr-vs-kp | Samples: {len(X)}")
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise} ({noise_ratio * 100:.2f}%)")
    print(f"Runtime: {elapsed * 1000:.2f} ms")
