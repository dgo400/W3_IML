import os
import time

import pandas as pd
from sklearn.cluster import Birch

from parser import load_arff
# from preprocessing import preprocess
# from metrics import evaluate

DATASETS = ["satimage", "kr-vs-kp", "horse-colic"]

# Parameter values to test (None = no global clustering step, the subclusters are the clusters)
THRESHOLDS = [0.1, 0.3, 0.5, 1.0]
BRANCHING_FACTORS = [10, 50, 100]
N_CLUSTERS = [None, 2, 4, 6, 8]


def run_birch(X, threshold, branching_factor, n_clusters):
    """Fit BIRCH on X and return the cluster label of each sample."""
    model = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters)
    return model.fit_predict(X)


if __name__ == "__main__":
    rows = []
    for name in DATASETS:
        X, _ = load_arff("data/" + name + ".arff")
        X = preprocess(X)

        for threshold in THRESHOLDS:
            for branching_factor in BRANCHING_FACTORS:
                for n_clusters in N_CLUSTERS:
                    start = time.time()
                    labels = run_birch(X, threshold, branching_factor, n_clusters)
                    elapsed = time.time() - start

                    row = {"dataset": name, "threshold": threshold,
                           "branching_factor": branching_factor, "n_clusters": n_clusters}
                    row.update(evaluate(X, y, labels))
                    row["time"] = elapsed
                    rows.append(row)

    results = pd.DataFrame(rows)
    print(results)

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/birch.csv", index=False)
