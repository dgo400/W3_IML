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

### 1. Full systematic sweep (default)
Executes the sweep across `satimage`, `kr-vs-kp` and `horse-colic`, testing metrics (Euclidean, Manhattan, Cosine), algorithm options (`auto`, `ball_tree`, `kd_tree`, `brute`), and parameter ranges. The data is preprocessed with `preprocess()` and evaluated with `evaluate()` from `src/metrics.py`, the same as every other algorithm (noise points count as one more group in the external metrics). The grids are in `run_experiments()`:

```bash
python src/dbscan.py
```
Results (clusters, noise points, silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, purity, F-measure, and runtime) are saved to:
`results/dbscan_results.csv`

### 2. Single targeted run with custom hyperparameters
To test specific hyperparameters:

```bash
python src/dbscan.py --dataset satimage --eps 1.5 --min-samples 10 --metric euclidean
python src/dbscan.py --dataset kr-vs-kp --eps 2.0 --min-samples 5 --metric euclidean
python src/dbscan.py --dataset horse-colic --eps 0.22 --min-samples 5 --metric cosine
```
*(Note: metric `cosine` automatically enforces `algorithm='brute'`)*

### 3. Generate diagnostic & report plots
To generate the $k$-distance elbow curves and parameter sensitivity curves:

```bash
python src/dbscan.py --plot
```

Figures are saved to `results/figures/`:
- `k_distance_plots.png`: $k$-NN distance distributions across metrics (knee heuristic for $eps$)
- `sensitivity_<dataset>.png`: $eps$ vs clusters, noise ratio, silhouette and ARI for each dataset

## Running the other algorithms

All scripts run from the `W3_IML/` folder, loop over the three datasets and write one CSV per algorithm to `results/`. To change the hyperparameters, edit the constants at the top of each script.

| Script | Algorithm | Hyperparameters (constants) | Output |
|---|---|---|---|
| `python src/birch.py` | BIRCH (sklearn) | `THRESHOLDS`, `BRANCHING_FACTORS`, `N_CLUSTERS` (`None` = no global step) | `results/birch.csv` |
| `python src/kmeans.py` | Our K-Means | `N_CLUSTERS`, `MAX_ITER`, `SEEDS` | `results/kmeans.csv` |
| `python src/kmeans_plusplus.py` | Our K-Means++ | `N_CLUSTERS`, `MAX_ITER`, `SEEDS` | `results/kmeans_plusplus.csv` |
| `python src/clarans.py` | Our CLARANS | `N_CLUSTERS`, `NUMLOCAL`, `MAXNEIGHBOR`, `SEEDS` | `results/clarans.csv` |

To use one algorithm from code, call its `run_*` function on the preprocessed data, e.g. `run_clarans(X, n_clusters=3, numlocal=2, maxneighbor=250, seed=0)`.

## Analysis scripts

Run them after the algorithm scripts above:

```bash
python src/plot_k_selection.py   # K-Means elbow/silhouette -> results/kmeans_selected_k.csv
python src/compare_kmeans.py     # K-Means vs K-Means++ at the selected k
python src/plot_sensitivity.py   # BIRCH and CLARANS sensitivity figures
python src/summary.py            # mean ± std per configuration and best configurations
```

`summary.py` reads every algorithm CSV in `results/` and writes `results/summary.csv` and `results/best_configs.csv` (best configuration per algorithm and dataset by ARI and by Silhouette).
