import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt

BIRCH_CSV = "results/birch.csv"
CLARANS_CSV = "results/clarans.csv"
FIGURES_DIR = "results/figures"

# BIRCH figure with a final k: the branching factor barely changes the labels, so we fix it
BIRCH_BRANCHING = 50
# CLARANS figure: lines are maxneighbor values for this numlocal
CLARANS_NUMLOCAL = 2


def save(fig, file_name):
    path = os.path.join(FIGURES_DIR, file_name)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Figure saved to:", path)


def plot_lines(axes, df, x, line_col, line_label, metrics):
    """One subplot per metric: metric vs x, one line per value of line_col (mean ± std over seeds)."""
    for value, sub in df.groupby(line_col, dropna=False):
        stats = sub.groupby(x)[metrics].agg(["mean", "std"])
        label = f"{line_label}={'None' if pd.isna(value) else int(value)}"
        for ax, metric in zip(axes, metrics):
            # NaN mean (e.g. Silhouette with a single cluster) is left as a gap, not drawn as 0
            ax.errorbar(stats.index, stats[(metric, "mean")], yerr=stats[(metric, "std")].fillna(0),
                        fmt="o-", capsize=3, label=label)


def plot_birch(df):
    datasets = list(df["dataset"].unique())

    # 1. No global step (k=None): the threshold alone decides the number of clusters
    metrics = ["n_clusters", "ARI", "Silhouette", "time"]
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(4.5 * len(metrics), 3.5 * len(datasets)),
                             squeeze=False, sharey="col")
    for row, name in enumerate(datasets):
        sub = df[(df["dataset"] == name) & df["k"].isna()]
        plot_lines(axes[row], sub, "threshold", "branching_factor", "B", metrics)
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_title(name)
    finish_grid(axes, metrics, "threshold")
    fig.suptitle("BIRCH without global clustering (k=None)")
    save(fig, "birch_threshold_no_k.png")

    # 2. With a final k
    metrics = ["n_clusters", "ARI", "Silhouette", "Davies-Bouldin"]
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(4.5 * len(metrics), 3.5 * len(datasets)),
                             squeeze=False, sharey="col")
    for row, name in enumerate(datasets):
        sub = df[(df["dataset"] == name) & df["k"].notna() & (df["branching_factor"] == BIRCH_BRANCHING)]
        plot_lines(axes[row], sub, "threshold", "k", "k", metrics)
        axes[row, 0].set_title(name)
    finish_grid(axes, metrics, "threshold")
    fig.suptitle(f"BIRCH with a final k (branching_factor={BIRCH_BRANCHING})")
    save(fig, "birch_threshold_k.png")


def plot_clarans(df):
    datasets = list(df["dataset"].unique())
    metrics = ["cost", "ARI", "Silhouette", "time"]
    # cost and time depend on the dataset size, so only ARI and Silhouette share the y axis
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(4.5 * len(metrics), 3.5 * len(datasets)),
                             squeeze=False)
    for row, name in enumerate(datasets):
        sub = df[(df["dataset"] == name) & (df["numlocal"] == CLARANS_NUMLOCAL)]
        plot_lines(axes[row], sub, "k", "maxneighbor", "maxneighbor", metrics)
        axes[row, 0].set_title(name)
        axes[row, 0].set_xticks(sorted(sub["k"].unique()))
    for col in [1, 2]:
        low = min(ax.get_ylim()[0] for ax in axes[:, col])
        high = max(ax.get_ylim()[1] for ax in axes[:, col])
        for ax in axes[:, col]:
            ax.set_ylim(low, high)
    finish_grid(axes, metrics, "k")
    fig.suptitle(f"CLARANS (numlocal={CLARANS_NUMLOCAL}, mean ± std over seeds)")
    save(fig, "clarans_k_maxneighbor.png")


def finish_grid(axes, metrics, x_label):
    for row in axes:
        for ax, metric in zip(row, metrics):
            ax.set_ylabel(metric)
            ax.grid(True, linestyle="--", alpha=0.5)
    for ax in axes[-1]:
        ax.set_xlabel(x_label)
    axes[0, -1].legend(fontsize=8)


if __name__ == "__main__":
    plot_birch(pd.read_csv(BIRCH_CSV))
    if os.path.exists(CLARANS_CSV):
        plot_clarans(pd.read_csv(CLARANS_CSV))
