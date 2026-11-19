import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt

from parser import load_arff

RESULTS_CSV = "results/kmeans.csv"
FIGURE_PATH = "results/figures/kmeans_k_selection.png"
SELECTED_K_CSV = "results/kmeans_selected_k.csv"

# Which criterion decides the K passed to compare_kmeans.py ("silhouette" or "elbow").
CRITERION = "silhouette"
# After looking at the figure you can force a K for a dataset, e.g. {"satimage": 6}.
MANUAL_K = {}


def summarise_by_k(df):
    """Mean and std of SSE and Silhouette over the seeds, for each (dataset, k)."""
    return (df.groupby(["dataset", "k"])[["sse", "Silhouette"]]
              .agg(["mean", "std"])
              .reset_index())


def elbow_k(ks, sse):
    """Return the k where the SSE curve bends the most (furthest below the chord)."""
    ks = np.asarray(ks, dtype=float)
    sse = np.asarray(sse, dtype=float)
    # Normalise both axes to [0, 1] so the distance does not depend on the units of SSE
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (sse - sse.min()) / (sse.max() - sse.min())
    # Chord from the first to the last point: since y goes from 1 to 0, it is y = 1 - x
    chord = 1 - x
    # The elbow is the point that lies furthest below that straight line
    return int(ks[np.argmax(chord - y)])


def silhouette_k(ks, silhouette):
    """Return the k with the highest mean Silhouette."""
    return int(np.asarray(ks)[np.argmax(silhouette)])


if __name__ == "__main__":
    df = pd.read_csv(RESULTS_CSV)
    stats = summarise_by_k(df)
    datasets = list(df["dataset"].unique())

    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 8), squeeze=False)
    rows = []

    for col, name in enumerate(datasets):
        s = stats[stats["dataset"] == name].sort_values("k")
        ks = s["k"].to_numpy()
        _, y = load_arff("data/" + name + ".arff")
        n_classes = int(y.nunique())

        k_elbow = elbow_k(ks, s[("sse", "mean")])
        k_sil = silhouette_k(ks, s[("Silhouette", "mean")])
        k_chosen = MANUAL_K.get(name, k_sil if CRITERION == "silhouette" else k_elbow)
        rows.append({"dataset": name, "n_classes": n_classes, "k_elbow": k_elbow,
                     "k_silhouette": k_sil, "k_selected": k_chosen})

        # Row 0: SSE, row 1: Silhouette. Error bars = std over the seeds.
        for row, (metric, k_best, label) in enumerate([("sse", k_elbow, "elbow"),
                                                       ("Silhouette", k_sil, "max")]):
            ax = axes[row, col]
            ax.errorbar(ks, s[(metric, "mean")], yerr=s[(metric, "std")],
                        fmt="o-", capsize=4)
            ax.axvline(k_best, color="tab:red", linestyle="--", label=f"{label}: k={k_best}")
            ax.axvline(n_classes, color="tab:green", linestyle=":", label=f"classes: {n_classes}")
            ax.set_xticks(ks)
            ax.set_xlabel("k")
            ax.set_ylabel(metric if metric != "sse" else "SSE")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            if row == 0:
                ax.set_title(name)

    fig.suptitle("K-Means: choosing k (mean ± std over seeds)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)

    selected = pd.DataFrame(rows)
    selected.to_csv(SELECTED_K_CSV, index=False)
    print(selected.to_string(index=False))
    print("\nFigure saved to:", FIGURE_PATH)
    print("Selected K saved to:", SELECTED_K_CSV)
