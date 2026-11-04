# Work 3 - IML: Clustering and visualization

This project applies and compares clustering algorithms on three .arff files:
DBSCAN and BIRCH from sklearn, plus our own K-Means, improved K-Means,
PAM/CLARA/CLARANS/K-Prototypes and Fuzzy C-Means. It also includes validation
metrics and dimensionality reduction (our own PCA, plus sklearn PCA and Isomap) for visualization.

## Project structure

```
W3_IML/
├── data/              # .arff datasets
├── src/               # Python code (parser.py, ...)
├── requirements.txt
└── pyproject.toml
```

## Setup

Requires **Python 3.11**. Run all commands from the `W3_IML/` folder.

**Option A: with [uv](https://docs.astral.sh/uv/)** (installs Python 3.11 if you don't have it):

```bash
uv python install 3.11
uv venv .venv --python 3.11
uv pip install --python .venv -r requirements.txt
```

**Option B: with pip** (Python 3.11 must already be installed):

```bash
py -3.11 -m venv .venv          # Linux/macOS: python3.11 -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Activate the environment and check that it works:

```bash
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
python src/parser.py
```

## Running DBSCAN Experiments

To execute the DBSCAN experiments across both `kr-vs-kp` and `horse-colic` datasets, testing metrics (Euclidean, Manhattan, Cosine), algorithms (`auto`, `ball_tree`, `kd_tree`, `brute`), and parameter sweeps:

```bash
python src/dbscan.py
```

Results (clusters, noise points, silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, purity, and runtime) will be saved to:
`results/dbscan_results.csv`
