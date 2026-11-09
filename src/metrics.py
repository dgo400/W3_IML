
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score


def evaluate(X, y, labels):
    """
    Evaluate clustering results.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Numeric features used for clustering.
    y : array-like, shape (n_samples,)
        True class labels.
    labels : array-like, shape (n_samples,)
        Predicted cluster labels. -1 represents noise.

    Returns
    -------
    dict
        ARI, Davies-Bouldin, Purity, F-measure,
        number of clusters, and number of noise points.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("X must be a 2D numeric array.")

    if y.ndim != 1 or labels.ndim != 1:
        raise ValueError("y and labels must be 1D arrays.")

    if len(X) != len(y) or len(y) != len(labels):
        raise ValueError(
            "X, y, and labels must contain the same number of samples."
        )

    if len(labels) == 0:
        raise ValueError("The dataset cannot be empty.")

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains missing or infinite values. "
            "Preprocess the data before evaluation."
        )

    # DBSCAN uses -1 to identify noise.
    noise_mask = labels == -1
    noise_points = int(np.sum(noise_mask))

    # Count clusters without counting noise.
    cluster_labels = np.unique(labels[~noise_mask])
    n_clusters = len(cluster_labels)

    # External evaluation: compare predicted labels with true classes.
    ari = float(adjusted_rand_score(y, labels))

    # Purity: sum the majority-class count in each predicted group.
    total_correct = 0

    for cluster in np.unique(labels):
        members = y[labels == cluster]
        _, counts = np.unique(members, return_counts=True)
        total_correct += int(np.max(counts))

    purity = float(total_correct / len(y))

    # F-measure: weighted average of each class's best cluster F-score.
    classes, class_counts = np.unique(y, return_counts=True)
    predicted_groups = np.unique(labels)

    weighted_f = 0.0

    for class_label, class_count in zip(classes, class_counts):
        best_f = 0.0

        for group in predicted_groups:
            intersection = np.sum(
                (y == class_label) & (labels == group)
            )

            if intersection == 0:
                continue

            precision = intersection / np.sum(labels == group)
            recall = intersection / class_count

            f_score = (
                2 * precision * recall / (precision + recall)
            )

            best_f = max(best_f, f_score)

        weighted_f += (class_count / len(y)) * best_f

    f_measure = float(weighted_f)

    # Davies-Bouldin: exclude noise and require at least two clusters.
    valid_mask = ~noise_mask
    valid_X = X[valid_mask]
    valid_labels = labels[valid_mask]

    if (
        n_clusters >= 2
        and len(valid_X) > n_clusters
    ):
        try:
            db_index = float(
                davies_bouldin_score(valid_X, valid_labels)
            )
        except ValueError:
            db_index = float("nan")
    else:
        db_index = float("nan")

    return {
        "ARI": ari,
        "Davies-Bouldin": db_index,
        "Purity": purity,
        "F-measure": f_measure,
        "n_clusters": n_clusters,
        "noise_points": noise_points,
    }

