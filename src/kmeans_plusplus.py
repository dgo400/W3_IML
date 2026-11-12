

import os
import time
import pandas as pd

from parser import load_arff
from preprocessing import preprocess
from metrics import evaluate

import numpy as np



DATASETS = ["satimage", "kr-vs-kp", "horse-colic"]
N_CLUSTERS = [2, 3, 4, 6, 8]
MAX_ITER = 100
SEEDS = [0, 1, 2, 3, 4]


def init_kmeans_plusplus(X, k, seed=None):
    """Choose initial centroids using K-Means++."""
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if not 1 <= k <= len(X):
        raise ValueError("k must be between 1 and the number of samples.")

    rng = np.random.default_rng(seed)

    # Choose the first centroid randomly.
    first_index = rng.integers(len(X))
    centroids = [X[first_index].copy()]

    # Calculate squared distances to the first centroid.
    closest_dist_sq = np.sum(
        (X - centroids[0]) ** 2, axis=1
    )

    # Choose the remaining centroids.
    for _ in range(1, k):
        total_distance = np.sum(closest_dist_sq)

        if total_distance == 0:
            break

        probabilities = closest_dist_sq / total_distance
        next_index = rng.choice(len(X), p=probabilities)

        centroids.append(X[next_index].copy())

        new_dist_sq = np.sum(
            (X - X[next_index]) ** 2, axis=1
        )

        closest_dist_sq = np.minimum(
            closest_dist_sq, new_dist_sq
        )

    # Handle datasets where all points are identical.
    while len(centroids) < k:
        centroids.append(X[0].copy())

    return np.array(centroids)


class KMeansPlusPlus:
    """K-Means clustering using K-Means++ initialization."""

    def __init__(self, n_clusters, max_iter=100, seed=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2D array.")

        if not np.isfinite(X).all():
            raise ValueError("X must not contain missing or infinite values.")

        if not 1 <= self.n_clusters <= len(X):
            raise ValueError("Invalid number of clusters.")

        # Your K-Means++ initialization.
        self.centroids = init_kmeans_plusplus(
            X, self.n_clusters, self.seed
        )

        for iteration in range(self.max_iter):
            # Assign every point to its nearest centroid.
            distances = np.sum(
                (X[:, None, :] - self.centroids[None, :, :]) ** 2,
                axis=2
            )
            labels = np.argmin(distances, axis=1)

            # Calculate new centroids.
            new_centroids = self.centroids.copy()

            for cluster_id in range(self.n_clusters):
                points = X[labels == cluster_id]

                if len(points) > 0:
                    new_centroids[cluster_id] = np.mean(
                        points, axis=0
                    )

            # Stop if centroids no longer move significantly.
            if np.allclose(
                self.centroids, new_centroids, rtol=1e-5, atol=1e-8
            ):
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

        self.n_iter_ = iteration + 1

        # Calculate final labels using the final centroids.
        distances = np.sum(
            (X[:, None, :] - self.centroids[None, :, :]) ** 2,
            axis=2
        )
        self.labels_ = np.argmin(distances, axis=1)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        distances = np.sum(
            (X[:, None, :] - self.centroids[None, :, :]) ** 2,
            axis=2
        )

        return np.argmin(distances, axis=1)


def run_kmeans_plusplus(X, n_clusters, max_iter=100, seed=None):
    """Fit K-Means++ and return the cluster labels."""
    model = KMeansPlusPlus(
        n_clusters=n_clusters,
        max_iter=max_iter,
        seed=seed
    )
    model.fit(X)
    return model.labels_


if __name__ == "__main__":
    rows = []

    for name in DATASETS:
        print(f"\nProcessing dataset: {name}")

        X, y = load_arff("data/" + name + ".arff")
        X = preprocess(X)

        for k in N_CLUSTERS:
            for seed in SEEDS:
                start = time.time()

                model = KMeansPlusPlus(
                    n_clusters=k,
                    max_iter=MAX_ITER,
                    seed=seed
                )
                model.fit(X)

                elapsed = time.time() - start

                row = {
                    "dataset": name,
                    "k": k,
                    "max_iter": MAX_ITER,
                    "seed": seed,
                    "n_iter": model.n_iter_,
                    "time": elapsed
                }

                row.update(evaluate(X, y, model.labels_))
                rows.append(row)

                print(
                    f"k={k}, seed={seed}, "
                    f"iterations={model.n_iter_}, "
                    f"time={elapsed:.3f}s"
                )

    results = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    output_file = "results/kmeans_plusplus.csv"
    results.to_csv(output_file, index=False)

    print("\nResults saved to:", output_file)
    print(results.head())