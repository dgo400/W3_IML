import os
import time

import numpy as np
import pandas as pd

from parser import load_arff
from preprocessing import preprocess
from metrics import evaluate

DATASETS = ["satimage", "kr-vs-kp", "horse-colic"]

# Parameter values to test
N_CLUSTERS = [2, 3, 4, 6, 8]
MAX_ITER = 100
SEEDS = [0, 1, 2, 3, 4] # K-Means depends on the random initialisation, so each setting is repeated

def init_random_clusters(X, k, seed): # X (samples, features)
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(X), size = k, replace = False)
    return X[indexes].copy()

def euclidean(point, data):
    # point (m,); data (n,m); output size (n,)
    return np.sqrt(np.sum((point - data) ** 2, axis = 1))

class KMeans:
    def __init__(self, n_clusters, max_iter, seed = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X):
        X = np.asarray(X, dtype = float)
        iteration = 0
        self.centroids = init_random_clusters(X, self.n_clusters, self.seed) # (k, features)
        prev_centroids = None
        converged = False
        while not converged and iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.n_clusters)] # a list of points for each cluster [[points c1], [points c2], ..., [points c_k]]
            for x in X:
                dists = euclidean(x, self.centroids)  # list of distances to each centroid (k,)
                centroid_idx = np.argmin(dists) # select cluster ci => dist(x, ci) is minimal
                sorted_points[centroid_idx].append(x) # append to the corresponding cluster
            prev_centroids = self.centroids # keep the previous centroids to compare with the new ones
            # new centroids from the mean of the cluster points; centroids having no points are kept at the previous centroid
            self.centroids = np.array([np.mean(cluster, axis = 0) if len(cluster) > 0 else prev_centroids[i] for i, cluster in enumerate(sorted_points)])
            converged = np.array_equal(self.centroids, prev_centroids) # stop when the centroids no longer move
            iteration += 1
        self.n_iter_ = iteration
        self.labels_ = self.predict(X)
        return self.centroids, self.labels_

    def predict(self, X):
        X = np.asarray(X, dtype = float)
        return np.array([np.argmin(euclidean(x, self.centroids)) for x in X]) # closest centroid index for each sample

def run_kmeans(X, n_clusters, max_iter, seed):
    """Fit our K-Means on X and return the cluster label of each sample."""
    model = KMeans(n_clusters = n_clusters, max_iter = max_iter, seed = seed)
    model.fit(X)
    return model.labels_

if __name__ == "__main__":
    rows = []
    for name in DATASETS:
        X, y = load_arff("data/" + name + ".arff")
        X = preprocess(X)

        for n_clusters in N_CLUSTERS:
            for seed in SEEDS:
                start = time.time()
                labels = run_kmeans(X, n_clusters, MAX_ITER, seed)
                elapsed = time.time() - start

                row = {"dataset": name, "k": n_clusters, "max_iter": MAX_ITER, "seed": seed}
                row.update(evaluate(X, y, labels))
                row["time"] = elapsed
                rows.append(row)

    results = pd.DataFrame(rows)
    print(results)

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/kmeans.csv", index=False)
