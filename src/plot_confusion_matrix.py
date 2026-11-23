
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parser import load_arff
from preprocessing import preprocess
from kmeans_plusplus import KMeansPlusPlus
import pandas as pd

from sklearn.metrics.cluster import contingency_matrix


def plot_confusion_matrix(y_true, cluster_labels, output_path, title="Clustering Contingency Matrix"):
    """
    Plot actual classes vs predicted clusters as a heatmap.
    """
    y_true = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)

    if len(y_true) != len(cluster_labels):
        raise ValueError("True labels and cluster labels must have the same length.")

    matrix = contingency_matrix(y_true, cluster_labels)
    classes = np.unique(y_true)
    clusters = np.unique(cluster_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(clusters)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted clusters")
    ax.set_ylabel("Actual classes")
    ax.set_title(title)

    fig.colorbar(image, ax=ax, label="Number of samples")

    # Display the number of samples in each cell.
    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > threshold else "black"
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Confusion matrix saved to: {output_path}")


if __name__ == "__main__":
    selected = pd.read_csv("results/kmeans_plusplus_selected_k.csv")

    for _, row in selected.iterrows():
        name = row["dataset"]
        k = int(row["k_selected"])

        X, y = load_arff(f"data/{name}.arff")
        X = preprocess(X)

        model = KMeansPlusPlus(n_clusters=k, max_iter=100, seed=0)
        model.fit(X)

        plot_confusion_matrix(
            y,
            model.labels_,
            f"results/figures/{name}_kmeans_plusplus_confusion.png",
            title=f"K-Means++ — {name} (K={k}, seed=0)"
        )