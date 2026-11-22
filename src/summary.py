"""
Put the results of every algorithm together.

1. Read every results/*.csv and add an `algorithm` column with the file name.
2. Group by algorithm, dataset and parameters (all seeds together) and compute
   mean and std of every metric and of the time.
3. For each algorithm and dataset keep the best configuration by ARI
   (and also by Silhouette).

Outputs: results/summary.csv and results/best_configs.csv
"""

import glob
import os

import pandas as pd

RESULTS_DIR = "results"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
BEST_CSV = os.path.join(RESULTS_DIR, "best_configs.csv")

# Files produced by the analysis scripts, not by an algorithm run
DERIVED_FILES = {"summary.csv", "best_configs.csv", "kmeans_selected_k.csv", "kmeans_comparison.csv"}

# Everything that is a result of the run. Any other column (except seed) is a parameter.
METRICS = ["ARI", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Purity", "F-measure",
           "n_clusters", "noise_points", "noise_ratio", "sse", "cost", "iterations", "time"]
NOT_PARAMS = set(METRICS) | {"algorithm", "dataset", "seed"}

# Criteria used to choose the best configuration (higher is better for both)
BEST_BY = ["ARI", "Silhouette"]


def load_all_results(results_dir=RESULTS_DIR):
    """
    Read every algorithm CSV and add the `algorithm` column from its file name.
    Also return the parameter columns of each algorithm, e.g. {"birch": ["threshold", ...]}.
    """
    frames = []
    params_of = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        file_name = os.path.basename(path)
        if file_name in DERIVED_FILES:
            continue
        df = pd.read_csv(path)
        # "dbscan_results.csv" -> "dbscan", "kmeans_plusplus.csv" -> "kmeans_plusplus"
        algorithm = file_name[:-len(".csv")].removesuffix("_results")
        df.insert(0, "algorithm", algorithm)
        params_of[algorithm] = [c for c in df.columns if c not in NOT_PARAMS]
        frames.append(df)
    return pd.concat(frames, ignore_index=True), params_of


def format_value(value):
    """4.0 -> '4', 0.5 -> '0.5', NaN -> 'None' (e.g. BIRCH without final k)."""
    if pd.isna(value):
        return "None"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def summarise(results, params_of):
    """Mean and std over seeds for each (algorithm, dataset, parameter setting)."""
    param_cols = [c for c in results.columns if c not in NOT_PARAMS]
    metric_cols = [c for c in METRICS if c in results.columns]
    group_cols = ["algorithm", "dataset"] + param_cols

    # dropna=False: a NaN parameter (another algorithm's parameter, or BIRCH k=None)
    # must still form a group instead of the row being thrown away
    grouped = results.groupby(group_cols, dropna=False, sort=False)
    summary = grouped[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]  # ARI_mean, ARI_std...
    summary.insert(0, "n_runs", grouped.size())
    summary = summary.reset_index()

    # Only the parameters that belong to each algorithm, as readable text: "k=4, max_iter=100"
    summary.insert(2, "params", summary.apply(
        lambda row: ", ".join(f"{c}={format_value(row[c])}" for c in params_of[row["algorithm"]]),
        axis=1))
    return summary


def best_configs(summary, criteria=BEST_BY):
    """For each algorithm and dataset, the row with the highest mean of each criterion."""
    best = []
    for metric in criteria:
        col = f"{metric}_mean"
        if col not in summary.columns:
            continue
        valid = summary.dropna(subset=[col])  # e.g. Silhouette is NaN with a single cluster
        idx = valid.groupby(["algorithm", "dataset"])[col].idxmax()
        rows = summary.loc[idx].copy()
        rows.insert(0, "criterion", metric)
        best.append(rows)
    return pd.concat(best, ignore_index=True)


if __name__ == "__main__":
    results, params_of = load_all_results()
    summary = summarise(results, params_of)
    best = best_configs(summary)

    summary.to_csv(SUMMARY_CSV, index=False)
    best.to_csv(BEST_CSV, index=False)

    cols = ["criterion", "algorithm", "dataset", "params", "ARI_mean", "ARI_std",
            "Silhouette_mean", "Purity_mean", "time_mean"]
    pd.set_option("display.width", 200)
    print(best[[c for c in cols if c in best.columns]].to_string(index=False))
    print(f"\nSaved {len(summary)} configurations to {SUMMARY_CSV}")
    print(f"Saved best configurations to {BEST_CSV}")
