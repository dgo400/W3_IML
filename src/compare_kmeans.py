import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SELECTED_K_CSV = "results/kmeans_selected_k.csv"
RUNS = {"K-Means": "results/kmeans.csv", "K-Means++": "results/kmeans_plusplus.csv"}
COMPARISON_CSV = "results/kmeans_comparison.csv"
FIGURE_PATH = "results/figures/kmeans_vs_kmeanspp_ari.png"

METRICS = ["ARI", "Purity", "sse", "iterations", "time"]


def load_runs_at_selected_k():
    selected = pd.read_csv(SELECTED_K_CSV)[["dataset", "k_selected"]]
    frames = []
    for algorithm, path in RUNS.items():
        df = pd.read_csv(path)
        df.insert(0, "algorithm", algorithm)
        # The merge keeps a row only if its (dataset, k) is the selected one
        df = df.merge(selected, left_on=["dataset", "k"], right_on=["dataset", "k_selected"])
        frames.append(df.drop(columns="k_selected"))
    return pd.concat(frames, ignore_index=True)


def comparison_table(runs):
    table = runs.groupby(["dataset", "k", "algorithm"])[METRICS].agg(["mean", "std"])
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    return table.reset_index()


def pretty_table(table):
    out = table[["dataset", "k", "algorithm"]].copy()
    for metric in METRICS:
        decimals = 0 if metric == "sse" else 1 if metric == "iterations" else 3
        out[metric] = (table[f"{metric}_mean"].round(decimals).astype(str) + " ± "
                       + table[f"{metric}_std"].round(decimals).astype(str))
    return out


def plot_ari_boxplot(runs, path=FIGURE_PATH):
    datasets = list(runs["dataset"].unique())
    algorithms = list(RUNS)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4.5), squeeze=False)

    for ax, name in zip(axes[0], datasets):
        sub = runs[runs["dataset"] == name]
        values = [sub.loc[sub["algorithm"] == a, "ARI"] for a in algorithms]
        ax.boxplot(values, tick_labels=algorithms)
        for i, v in enumerate(values, start=1):  # each point is one seed
            ax.scatter([i] * len(v), v, alpha=0.6, zorder=3)
        ax.set_title(f"{name} (k={sub['k'].iloc[0]})")
        ax.set_ylabel("ARI")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("ARI over seeds: K-Means vs K-Means++")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def answer_question(table):
    for name, t in table.groupby("dataset"):
        km = t[t["algorithm"] == "K-Means"].iloc[0]
        pp = t[t["algorithm"] == "K-Means++"].iloc[0]
        print(f"\n{name}:")
        print(f"  higher ARI:          {pp['ARI_mean'] > km['ARI_mean']}  "
              f"({km['ARI_mean']:.3f} -> {pp['ARI_mean']:.3f})")
        print(f"  lower SSE:           {pp['sse_mean'] < km['sse_mean']}  "
              f"({km['sse_mean']:.0f} -> {pp['sse_mean']:.0f})")
        print(f"  more stable (ARI std): {pp['ARI_std'] < km['ARI_std']}  "
              f"({km['ARI_std']:.3f} -> {pp['ARI_std']:.3f})")
        print(f"  fewer iterations:    {pp['iterations_mean'] < km['iterations_mean']}  "
              f"({km['iterations_mean']:.1f} -> {pp['iterations_mean']:.1f})")


if __name__ == "__main__":
    runs = load_runs_at_selected_k()
    table = comparison_table(runs)
    table.to_csv(COMPARISON_CSV, index=False)
    plot_ari_boxplot(runs)

    pd.set_option("display.width", 200)
    print(pretty_table(table).to_string(index=False))
    answer_question(table)
    print("\nTable saved to:", COMPARISON_CSV)
    print("Figure saved to:", FIGURE_PATH)
