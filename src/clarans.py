"""
CLARANS (Ng & Han, 1994): k-medoids searched as a randomized walk on a graph.

Each node is a set of k medoids and its neighbours differ in one medoid. From a
random node we try random neighbours and move to the first one with lower cost;
after `maxneighbor` failed tries the node is a local minimum. This is repeated
`numlocal` times and the best local minimum is kept.

Dissimilarity: Euclidean distance on the same preprocessed data as K-Means.
Cost: sum of the distances of every point to its closest medoid.
"""

import os
import time

import numpy as np
import pandas as pd

from parser import load_arff
from preprocessing import preprocess
from metrics import evaluate

DATASETS = ["satimage", "kr-vs-kp", "horse-colic"]

# Parameter values to test
N_CLUSTERS = [2, 3, 4, 6, 8, 10]
NUMLOCAL = [1, 2]
MAXNEIGHBOR = [50, 250, 1000]
SEEDS = [0, 1, 2, 3, 4]


def distances_to(X, points):
    """Euclidean distance from every row of X to every point, shape (n, len(points))."""
    return np.sqrt(np.sum((X[:, None, :] - points[None, :, :]) ** 2, axis=2))


class CLARANS:
    def __init__(self, n_clusters, numlocal=2, maxneighbor=250, seed=None):
        self.n_clusters = n_clusters
        self.numlocal = numlocal
        self.maxneighbor = maxneighbor
        self.seed = seed

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        n, k = len(X), self.n_clusters
        self.cost_ = np.inf
        self.n_moves_ = 0  # accepted swaps over all local searches

        for _ in range(self.numlocal):
            medoids = rng.choice(n, size=k, replace=False)
            D = distances_to(X, X[medoids])  # (n, k), column i = distance to medoid i
            cost = D.min(axis=1).sum()
            tries = 0
            while tries < self.maxneighbor:
                # Random neighbour: replace medoid i by the non-medoid h
                i = rng.integers(k)
                h = rng.integers(n)
                if h in medoids:
                    continue
                D_new = D.copy()
                D_new[:, i] = distances_to(X, X[h][None])[:, 0]
                new_cost = D_new.min(axis=1).sum()
                if new_cost < cost:
                    medoids[i], D, cost = h, D_new, new_cost
                    self.n_moves_ += 1
                    tries = 0
                else:
                    tries += 1
            if cost < self.cost_:
                self.cost_ = float(cost)
                self.medoid_indices_ = medoids.copy()
                self.labels_ = D.argmin(axis=1)

        self.medoids_ = X[self.medoid_indices_]
        return self

    def predict(self, X):
        return distances_to(np.asarray(X, dtype=float), self.medoids_).argmin(axis=1)


def run_clarans(X, n_clusters, numlocal, maxneighbor, seed):
    """Fit CLARANS on X and return the labels, the cost and the number of accepted swaps."""
    model = CLARANS(n_clusters=n_clusters, numlocal=numlocal, maxneighbor=maxneighbor, seed=seed)
    model.fit(X)
    return model.labels_, model.cost_, model.n_moves_


if __name__ == "__main__":
    rows = []
    for name in DATASETS:
        X, y = load_arff("data/" + name + ".arff")
        X = preprocess(X)
        print("Processing dataset:", name)

        for n_clusters in N_CLUSTERS:
            for numlocal in NUMLOCAL:
                for maxneighbor in MAXNEIGHBOR:
                    for seed in SEEDS:
                        start = time.time()
                        labels, cost, n_moves = run_clarans(X, n_clusters, numlocal, maxneighbor, seed)
                        elapsed = time.time() - start

                        row = {"dataset": name, "k": n_clusters, "numlocal": numlocal,
                               "maxneighbor": maxneighbor, "seed": seed}
                        row.update(evaluate(X, y, labels))
                        row["cost"] = cost
                        row["iterations"] = n_moves
                        row["time"] = elapsed
                        rows.append(row)

    results = pd.DataFrame(rows)
    print(results)

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/clarans.csv", index=False)
