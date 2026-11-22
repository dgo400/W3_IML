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
    from src.parser import load_arff
    from src.preprocessing import preprocess
    from src.metrics import evaluate
except ModuleNotFoundError:
    from parser import load_arff
    from preprocessing import preprocess
    from metrics import evaluate


DATASETS = ["satimage", "kr-vs-kp", "horse-colic"]
VALID_ALGORITHMS = ("auto", "ball_tree", "kd_tree", "brute")
VALID_METRICS = ("euclidean", "cosine", "manhattan")


def load_data(name):
    """Same preprocessing as the other algorithms: preprocessed X and class labels y."""
    X, y = load_arff("data/" + name + ".arff")
    return preprocess(X), y.to_numpy()


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


def run_experiments(dataset_names=DATASETS, output_csv="results/dbscan_results.csv"):
    """
    Run systematic DBSCAN parameter sweep across datasets, metrics, and algorithms.
    """
    import pandas as pd

    records = []

    # Dataset-specific parameter grids (eps ranges chosen from the k-distance plots)
    grid = {
        "satimage": {
            "metrics": ["euclidean", "manhattan", "cosine"],
            "algorithms": ["auto", "ball_tree", "kd_tree", "brute"],
            "min_samples_list": [5, 10, 20],
            "eps_map": {
                "euclidean": [1.0, 1.5, 2.0, 2.5],
                "manhattan": [4.0, 6.0, 8.0, 10.0],
                "cosine": [0.03, 0.05, 0.08, 0.12],
            },
        },
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
        X, y_true = load_data(ds_name)
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
                            metrics_dict = evaluate(X, y_true, labels)
                            record = {
                                "dataset": ds_name,
                                "metric": metric,
                                "nn_algorithm": algo,
                                "eps": eps,
                                "min_samples": min_samples,
                                **metrics_dict,
                                "time": elapsed_sec,
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
    view = subset[subset["nn_algorithm"] == target_algo][
        ["eps", "min_samples", "n_clusters", "noise_points", "noise_ratio", "Silhouette", "ARI", "Purity", "time"]
    ]
    print(f"\n--- {dataset_name} | Metric: {metric.upper()} (algorithm={target_algo}) ---")
    print(view.to_string(index=False))


def parse_arguments():
    """
    Parse command-line arguments for single-run or sweep experiments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="DBSCAN Clustering evaluation for Work 3 - IML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Dataset name for targeted single run.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Neighborhood radius (eps).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Minimum neighborhood samples for core points.",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine", "manhattan"],
        default="euclidean",
        help="Distance metric.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["auto", "ball_tree", "kd_tree", "brute"],
        default="auto",
        help="Nearest neighbor algorithm (note: cosine requires brute).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run full parameter sweep across all datasets and hyperparameters.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save k-distance knee plots and sensitivity curves.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/dbscan_results.csv",
        help="Destination path for saving parameter sweep CSV.",
    )
    return parser.parse_args()


def generate_plots(output_dir="results/figures", csv_path="results/dbscan_results.csv"):
    """
    Generate diagnostic and report plots:
      1. k-distance (knee) plots for heuristic eps determination.
      2. Sensitivity curves (eps vs clusters, noise ratio, silhouette).
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # 1. k-distance plots
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.5 * len(DATASETS), 5))
    k = 5

    for idx, ds_name in enumerate(DATASETS):
        ax = axes[idx]
        X, _ = load_data(ds_name)
        for metric in ["euclidean", "cosine", "manhattan"]:
            nbrs = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="brute").fit(X)
            distances, _ = nbrs.kneighbors(X)
            k_distances = np.sort(distances[:, -1])
            ax.plot(k_distances, label=f"{metric.capitalize()}")
        ax.set_title(f"k-Distance Plot: {ds_name} (k={k})")
        ax.set_xlabel("Sorted Points (Ascending)")
        ax.set_ylabel(f"{k}-NN Distance")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    k_dist_path = os.path.join(output_dir, "k_distance_plots.png")
    plt.savefig(k_dist_path, dpi=300)
    plt.close()
    print(f"Saved k-distance plots to: {k_dist_path}")

    # 2. Parameter sensitivity plots from CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for ds_name in DATASETS:
            # The neighbour search algorithm does not change the labels, only the time
            sub_ds = df[(df["dataset"] == ds_name) & (df["nn_algorithm"].isin(["auto", "brute"]))]
            if sub_ds.empty:
                continue

            metrics = sub_ds["metric"].unique()
            fig, axes = plt.subplots(len(metrics), 4, figsize=(20, 3.5 * len(metrics)), sharex=False)
            if len(metrics) == 1:
                axes = np.expand_dims(axes, 0)

            for m_idx, m in enumerate(metrics):
                sub_m = sub_ds[sub_ds["metric"] == m]
                for ms in sorted(sub_m["min_samples"].unique()):
                    curve = sub_m[sub_m["min_samples"] == ms].sort_values("eps")
                    axes[m_idx, 0].plot(curve["eps"], curve["n_clusters"], marker="o", label=f"min_samples={ms}")
                    axes[m_idx, 1].plot(curve["eps"], curve["noise_ratio"], marker="s", label=f"min_samples={ms}")
                    axes[m_idx, 2].plot(curve["eps"], curve["Silhouette"], marker="^", label=f"min_samples={ms}")
                    axes[m_idx, 3].plot(curve["eps"], curve["ARI"], marker="d", label=f"min_samples={ms}")

                axes[m_idx, 0].set_ylabel(f"{m.upper()}\nClusters (n)")
                axes[m_idx, 0].grid(True, linestyle="--", alpha=0.5)
                axes[m_idx, 0].legend()

                axes[m_idx, 1].set_ylabel("Noise Ratio")
                axes[m_idx, 1].grid(True, linestyle="--", alpha=0.5)
                axes[m_idx, 1].legend()

                axes[m_idx, 2].set_ylabel("Silhouette")
                axes[m_idx, 2].grid(True, linestyle="--", alpha=0.5)
                axes[m_idx, 2].legend()

                axes[m_idx, 3].set_ylabel("ARI")
                axes[m_idx, 3].grid(True, linestyle="--", alpha=0.5)
                axes[m_idx, 3].legend()

                if m_idx == len(metrics) - 1:
                    for col in range(4):
                        axes[m_idx, col].set_xlabel("eps")

            fig.suptitle(f"DBSCAN Parameter Sensitivity: {ds_name}", fontsize=14)
            plt.tight_layout()
            sens_path = os.path.join(output_dir, f"sensitivity_{ds_name}.png")
            plt.savefig(sens_path, dpi=300)
            plt.close()
            print(f"Saved sensitivity curves to: {sens_path}")


if __name__ == "__main__":
    args = parse_arguments()

    if args.plot:
        generate_plots(output_dir="results/figures", csv_path=args.output_csv)
        if not args.sweep and not (args.dataset and args.eps is not None):
            sys.exit(0)

    # If user specifies targeted run parameters
    if args.dataset and args.eps is not None and args.min_samples is not None:
        # Cosine requires brute
        algo = "brute" if args.metric == "cosine" else args.algorithm
        print(f"\n=== DBSCAN Single Run: {args.dataset.upper()} ===")
        print(f"Parameters: eps={args.eps}, min_samples={args.min_samples}, metric={args.metric}, algorithm={algo}")
        X, y_true = load_data(args.dataset)
        labels, elapsed_sec = run_dbscan(
            X, eps=args.eps, min_samples=args.min_samples,
            metric=args.metric, algorithm=algo
        )
        metrics = evaluate(X, y_true, labels)
        print(f"\nExecution time: {elapsed_sec * 1000:.2f} ms")
        print("Clustering Results & Metrics:")
        for k, v in metrics.items():
            print(f"  {k:18s}: {v}")
    else:
        # Default / --sweep behavior
        results_df = run_experiments(
            dataset_names=DATASETS,
            output_csv=args.output_csv,
        )

        # Print summary highlights
        for ds in DATASETS:
            for m in ["euclidean", "cosine", "manhattan"]:
                print_summary_table(results_df, ds, m)
