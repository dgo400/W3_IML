# Analysis: DBSCAN, BIRCH and CLARANS

Draft text for the report (Work 3). All numbers come from the CSV files in `results/` and the figures in
`results/figures/`, produced by the scripts listed in the README. Random algorithms (K-Means, K-Means++,
CLARANS) are run with seeds 0–4 and reported as mean ± std; DBSCAN and BIRCH are deterministic, so they are run once.

## 1. Common experimental setup

**Data sets.** satimage (6,435 × 36, numerical, 6 classes), kr-vs-kp (3,196 × 36, nominal, 2 classes) and
horse-colic (368 × 22, mixed with 23.8 % missing values, 2 classes). The class attribute is removed before
clustering and only used afterwards for external validation.

**Preprocessing (`preprocess()`), identical for every algorithm.** Numerical attributes: median imputation and
standardisation (zero mean, unit variance), so that no attribute dominates the distance. Nominal attributes:
mode imputation and one-hot encoding. After encoding the dimensions are 36 (satimage), 73 (kr-vs-kp) and 62
(horse-colic).

**Validation metrics (`evaluate()`), identical for every algorithm.**
- External: ARI, Purity and F-measure (each class matched with its best cluster).
- Internal: Silhouette, Davies–Bouldin and Calinski–Harabasz, all with Euclidean distance on the preprocessed data.
- DBSCAN noise (label −1) counts as one more group for the external metrics, so a run that marks most points as
  noise is penalised. The internal metrics are computed only on the non-noise points, and the noise ratio is
  reported next to them, because a high Silhouette with 90 % noise would otherwise look like a good result.
- Purity grows with the number of clusters (it reaches 1 with one cluster per point), so we use ARI as the main
  external criterion and Purity/F-measure only as support.

> Note: earlier DBSCAN results used their own metric functions that ignored noise and computed Silhouette with
> DBSCAN's metric. They were replaced by the shared `evaluate()` so that all algorithms can be compared directly.

## 2. DBSCAN (sklearn)

### 2.1 Setup
- Metrics: Euclidean, Manhattan and Cosine.
- `algorithm`: `auto`, `ball_tree`, `kd_tree` and `brute`. Cosine is not a true metric (it does not satisfy the
  triangle inequality), so it is only valid with `brute`.
- `min_samples`: {3, 5, 10} for kr-vs-kp, {3, 5, 8} for horse-colic and {5, 10, 20} for satimage, which is larger.
- `eps` grid for each dataset and metric, chosen from the 5-NN distance curve (Figure `k_distance_plots.png`).
  For satimage the 5-NN distance quantiles (50 %–95 %) are 1.3–2.4 for Euclidean, 5.9–11.2 for Manhattan and
  0.03–0.15 for Cosine, giving the grids {1.0, 1.5, 2.0, 2.5}, {4, 6, 8, 10} and {0.03, 0.05, 0.08, 0.12}.

108 runs per data set (`results/dbscan_results.csv`).

### 2.2 Effect of eps and min_samples
Figures `sensitivity_satimage.png`, `sensitivity_kr-vs-kp.png` and `sensitivity_horse-colic.png` show the same
pattern for all three metrics:
- A small `eps` labels most points as noise and splits the rest into many small clusters. For satimage with
  Euclidean distance, eps=1.0 gives 79–93 % noise.
- A large `eps` merges everything into one or two clusters with little noise (eps=2.5: 2–3 clusters, 2–6 % noise).
- Increasing `min_samples` moves these curves to the right: more noise and fewer clusters for the same `eps`.
- The useful region is narrow. On satimage ARI goes from 0.02 (eps=1.0) to 0.31 (eps=1.5) and back down to 0.10
  (eps=2.0) with Euclidean distance and min_samples=10. Small changes of `eps` change the result completely.

How often a configuration is degenerate (≤ 1 cluster, or more than 50 % noise):

| Data set | runs with ≤ 1 cluster | runs with > 50 % noise |
|---|---|---|
| satimage | 4 % | 26 % |
| kr-vs-kp | 32 % | 20 % |
| horse-colic | 61 % | 31 % |

DBSCAN finds a density gap in satimage only. In horse-colic (368 points, 62 dimensions) the density is too low
and too uniform: most settings produce either one cluster or mostly noise.

### 2.3 Effect of the distance metric
Best configuration by ARI for each metric (`auto`/`brute`):

| Data set | Metric | eps | min_samples | clusters | noise | ARI | Silhouette | F-measure |
|---|---|---|---|---|---|---|---|---|
| satimage | Euclidean | 1.5 | 10 | 6 | 29 % | 0.314 | 0.010 | 0.582 |
| satimage | Manhattan | 8.0 | 10 | 5 | 18 % | **0.344** | 0.057 | 0.586 |
| satimage | Cosine | 0.05 | 10 | 6 | 31 % | 0.296 | 0.227 | 0.547 |
| kr-vs-kp | all three | — | 5 | 31 | 22 % | 0.023 | −0.102 | 0.484 |
| horse-colic | Euclidean | 2.5 | 3 | 4 | 82 % | 0.094 | 0.144 | 0.649 |
| horse-colic | Manhattan | 8.0 | 5 | 2 | 69 % | **0.122** | 0.260 | 0.676 |
| horse-colic | Cosine | 0.22 | 5 | 5 | 62 % | 0.112 | 0.279 | 0.665 |

- On satimage the three metrics give similar quality. Manhattan is slightly better, which agrees with the general
  observation that L1 is more robust than L2 in higher dimensions.
- On kr-vs-kp the three metrics give **exactly the same clustering**. After one-hot encoding every row has
  exactly 36 ones, so with *m* differing attributes the distances are Euclidean √(2m), Manhattan 2m and Cosine
  m/36. They are monotone functions of the same quantity, so eps=1.8 (Euclidean), 2.0 (Manhattan) and 0.03
  (Cosine) all mean "at most one differing attribute". For purely nominal data the choice of metric is irrelevant
  and only `eps` matters.

### 2.4 Effect of the `algorithm` parameter
`algorithm` only selects the nearest-neighbour search structure, so in theory it never changes the labels. In our
runs the labels are identical, with one exception: kr-vs-kp + Manhattan + `ball_tree`, where 1–8 points change.
There the distances are exact integers (2m) that coincide with `eps`, and the boundary test `d ≤ eps` is decided
by floating-point rounding differently in the tree.

The parameter does change the runtime (mean time per run in seconds):

| Data set (Euclidean) | auto | brute | ball_tree | kd_tree |
|---|---|---|---|---|
| horse-colic (368 × 62) | 0.001 | 0.001 | 0.005 | 0.007 |
| kr-vs-kp (3,196 × 73) | 0.011 | 0.010 | 0.344 | 0.297 |
| satimage (6,435 × 36) | 0.042 | 0.038 | 0.319 | 0.587 |

With 36–73 dimensions the trees cannot prune the search (curse of dimensionality) and are 5–30× slower than
brute force, depending on the data set. `auto` correctly picks brute force in these cases.

### 2.5 Do the metrics agree?
No. The best configuration by Silhouette is almost always a degenerate one. For example, on satimage the highest
Silhouette (0.570) is Euclidean eps=1.0, min_samples=20, which has **93 % noise** and ARI=0.002: the few remaining
points are very dense, so they are well separated. This is why Silhouette has to be read together with the noise
ratio, and why we select DBSCAN configurations by ARI.

## 3. BIRCH (sklearn)

### 3.1 Setup
- `threshold` ∈ {0.5, 1, 2, 3, 4, 5}: maximum radius of a CF subcluster.
- `branching_factor` (B) ∈ {10, 50, 100}: maximum number of children of a CF node.
- `n_clusters` (k) ∈ {None, 2, 4, 6, 8}: with `None` the leaf subclusters are the final clusters; otherwise a
  global agglomerative step merges them into k clusters.

90 runs per data set (`results/birch.csv`).

> The first version used thresholds {0.1, 0.3, 0.5, 1.0}. With standardised data these radii are far too small:
> with k=None we obtained 334–6,435 subclusters, practically one per point (on kr-vs-kp the minimum possible
> distance between different rows is √2 ≈ 1.41, so any threshold below that makes every distinct row its own
> subcluster). The range was moved up after measuring the number of subclusters per threshold.

### 3.2 Effect of threshold without a global step (k=None) — Figure `birch_threshold_no_k.png`
- The number of clusters falls exponentially with the threshold. satimage: 5,982 → 2,130 → 179–234 → 35–44 →
  15 → 9 for thresholds 0.5 to 5. kr-vs-kp and horse-colic collapse to 1–2 clusters at threshold 4.
- On satimage this is a usable clustering method on its own: threshold 4 gives 15 clusters with ARI=0.49,
  similar to the best K-Means results. On kr-vs-kp and horse-colic ARI stays ≤ 0.03 at every threshold.
- A larger threshold also makes the algorithm faster (fewer subclusters to insert and split), e.g. satimage
  0.37 s → 0.02 s with B=10.

### 3.3 Effect of threshold and k with a global step — Figure `birch_threshold_k.png`
- When the global step is used, k determines the number of clusters and the threshold only changes the
  subclusters that are merged. This still has a large effect: satimage with k=6 goes from ARI=0.27 (threshold 2,
  B=10) to 0.57 (threshold 1, B=10). BIRCH is the most threshold-sensitive of the methods we tested.
- If the threshold is so large that fewer than k subclusters exist, sklearn warns
  (`ConvergenceWarning: Number of subclusters found (1) by BIRCH is less than (k)`) and returns fewer clusters
  than requested. This happens for kr-vs-kp and horse-colic at threshold ≥ 4.
- Best k per data set (mean ARI over thresholds 0.5–3, B=50): satimage k=8 (0.43), close to the 6 true classes;
  kr-vs-kp ARI ≤ 0.03 for every k; horse-colic k=2 (0.07).

### 3.4 Effect of branching_factor
B hardly affects the quality: the std of ARI across the three B values, for a fixed threshold and k, is
0.005–0.036. Its main effect is runtime. B=10 creates a deeper tree with more node splits and is 30–40 % slower
than B=50 or B=100 (satimage 0.19 s vs 0.14 s); 50 and 100 behave the same.

### 3.5 Best configurations

| Data set | criterion | threshold | B | k | ARI | Silhouette | F-measure | time (s) |
|---|---|---|---|---|---|---|---|---|
| satimage | ARI | 1.0 | 100 | 8 | **0.578** | 0.277 | 0.744 | 0.13 |
| satimage | Silhouette | 5.0 | 10 | 2 | 0.063 | 0.500 | — | — |
| kr-vs-kp | ARI | 2.0 | 10 | 6 | 0.045 | 0.081 | 0.453 | 0.04 |
| horse-colic | ARI | 0.5 | 50 | 2 | 0.142 | 0.105 | 0.695 | 0.01 |

Again, Silhouette prefers k=2 on satimage (two large, well separated groups), while ARI prefers k≈6–8, close to
the true number of classes.

## 4. CLARANS (own implementation, `src/clarans.py`)

### 4.1 Algorithm and implementation decisions
CLARANS (Ng & Han, 1994) is a k-medoids method that searches a graph whose nodes are sets of k medoids. Two nodes
are neighbours if they differ in one medoid. The algorithm:
1. Starts from a random node (k random points as medoids).
2. Picks a random neighbour (replaces a random medoid *i* by a random non-medoid *h*) and moves to it if the
   total cost decreases, resetting the counter of failed tries.
3. Declares a local minimum after `maxneighbor` consecutive failed tries.
4. Repeats from a new random node `numlocal` times and returns the best local minimum.

Unlike PAM, it does not evaluate all k(n−k) swaps at each step, and unlike CLARA, it does not restrict the
search to a sample, so every point can become a medoid.

**Dissimilarity.** Euclidean distance on the same preprocessed data as K-Means and K-Means++, so the three methods
are compared on the same representation. **Cost** = sum of the distances of each point to its closest medoid.
Medoids are real data points, so no "mean" is needed and any dissimilarity would be valid (for example Gower for
the mixed data). With Euclidean distance the method minimises the sum of distances rather than squared
distances, which makes it less sensitive to outliers than K-Means.

**Efficiency.** We store the n×k matrix of distances to the current medoids. Evaluating a neighbour only
recomputes one column (O(n·d)) and the row minima (O(n·k)), and the full n×n distance matrix that PAM needs is
never built. For satimage that matrix would take about 330 MB, while our implementation needs 6,435×k values.

**Validation.** On random subsets of horse-colic (30–40 points, k=2–4) we compared CLARANS (numlocal=2,
maxneighbor=250) with the exact optimum found by trying every combination of k medoids. CLARANS reached the
global optimum in 14 of 15 runs. In the remaining run (k=4) it stopped at a local minimum 1.4 % above the
optimum, which is expected for a randomised search.

### 4.2 Parameters
- k ∈ {2, 3, 4, 6, 8, 10} (same grid as K-Means).
- `numlocal` ∈ {1, 2}: the original paper recommends 2.
- `maxneighbor` ∈ {50, 250, 1000}: the paper recommends max(250, 1.25 % of k(n−k)), which gives 250 for
  horse-colic and between 250 and about 800 for satimage, depending on k.
- 5 seeds each, 180 runs per data set (`results/clarans.csv`).

Effect of the search parameters (mean over k):

| Data set | numlocal | maxneighbor | cost | cost std (seeds) | ARI | time (s) |
|---|---|---|---|---|---|---|
| satimage | 1 | 50 | 20,749 | 411 | 0.392 | 0.03 |
| satimage | 2 | 250 | 20,020 | 180 | 0.398 | 0.36 |
| satimage | 2 | 1000 | 19,774 | 86 | 0.399 | 1.41 |
| kr-vs-kp | 1 | 50 | 10,290 | 95 | 0.031 | 0.03 |
| kr-vs-kp | 2 | 1000 | 10,005 | 29 | 0.023 | 1.22 |
| horse-colic | 1 | 50 | 1,415 | 14 | 0.047 | 0.01 |
| horse-colic | 2 | 1000 | 1,384 | 1 | 0.077 | 0.13 |

- `maxneighbor` controls the trade-off between quality and time. Going from 50 to 1000 lowers the cost by 2–4 %
  and the std across seeds by 3–12×, and multiplies the time by about 20 (linear in the number of neighbours
  examined).
- `numlocal`=2 roughly doubles the time and helps less than a larger `maxneighbor`, because one long search
  already reaches a good local minimum on these data sets.
- The lower cost barely translates into better ARI on satimage (0.392 → 0.399). It helps more on horse-colic
  (0.047 → 0.077), where the 5 seeds give different partitions when `maxneighbor` is small.
- Final setting: numlocal=2, maxneighbor=1000. It is the most stable, and its cost (≤ 2.1 s per run) is
  acceptable for these data set sizes.

### 4.3 Choice of k — Figure `clarans_k_maxneighbor.png`
CLARANS with numlocal=2, maxneighbor=1000:

| Data set | k | ARI | Silhouette | Davies–Bouldin | F-measure | time (s) |
|---|---|---|---|---|---|---|
| satimage | 3 | 0.295 ± 0.004 | **0.437** ± 0.001 | **0.813** | 0.549 | 1.02 |
| satimage | 6 | **0.536** ± 0.004 | 0.348 ± 0.004 | 1.038 | **0.713** | 1.44 |
| kr-vs-kp | 2 | **0.033** ± 0.008 | **0.093** ± 0.003 | 2.982 | **0.589** | 0.83 |
| horse-colic | 2 | −0.018 ± 0.000 | **0.138** ± 0.000 | 2.676 | 0.606 | 0.09 |
| horse-colic | 4 | **0.159** ± 0.005 | 0.052 ± 0.002 | 3.530 | 0.614 | 0.12 |

- The cost always decreases with k (like the SSE in K-Means) and has no clear elbow, so it cannot be used alone
  to choose k.
- Internal metrics (Silhouette, Davies–Bouldin) choose k=3 on satimage and k=2 on the other two data sets.
  External metrics (ARI, F-measure) choose k=6 on satimage, **exactly the number of classes**, and k=4 on
  horse-colic. The internal metrics find the most compact geometric structure, which does not have to coincide
  with the class labels.
- On horse-colic, k=2 has the highest Silhouette but ARI≈0: the two compact groups found do not correspond to
  surgical/non-surgical lesions.

### 4.4 Comparison with K-Means and K-Means++
Same preprocessing and same k grid; mean ARI over 5 seeds (std in parentheses):

| Data set | k | K-Means | K-Means++ | CLARANS |
|---|---|---|---|---|
| satimage | 6 | 0.530 (0.001) | 0.506 (0.053) | **0.536** (0.004) |
| satimage | 3 | 0.294 (0.000) | 0.294 (0.000) | 0.295 (0.004) |
| kr-vs-kp | 2 | 0.002 (0.004) | 0.000 (0.000) | **0.033** (0.008) |
| horse-colic | 4 | 0.021 (0.014) | 0.044 (0.025) | **0.159** (0.005) |

- On satimage (numerical, compact classes) the three methods converge to almost the same partition.
- On the nominal and mixed data sets CLARANS clearly beats both K-Means versions on horse-colic (ARI 0.16 vs
  ≤ 0.06 for every k). A medoid is a real patient, while a mean of one-hot columns is a fractional vector that
  matches no possible patient, and minimising distances (not squared distances) reduces the influence of the
  atypical cases that are common in this data set.
- Cost: CLARANS is the slowest of the three (1.4 s vs 0.03 s for K-Means++ on satimage k=6), because each of the
  thousands of neighbours examined costs O(n·k). It is still far cheaper than PAM, which would evaluate
  k(n−k) ≈ 38,600 swaps per iteration on satimage with k=6.

### 4.5 Limitations
- The result is a local minimum that depends on the seed. Lowering this variance requires a larger
  `maxneighbor`, which costs time linearly.
- k must be given; it has to be chosen with an external procedure (here a k sweep plus internal/external metrics).
- With Euclidean distance on one-hot encoded data, nominal attributes are treated as 0/1 coordinates. A
  dedicated dissimilarity (Gower or simple matching) would represent them more faithfully, and CLARANS would
  support it without changing the algorithm.
- The time grows with n·k per neighbour evaluation. For very large data sets the per-swap cost becomes the
  bottleneck.

## 5. Cross-algorithm summary (best configuration by ARI)

| Data set | DBSCAN | BIRCH | K-Means | K-Means++ | CLARANS |
|---|---|---|---|---|---|
| satimage | 0.344 (Manhattan, eps=8, ms=10; 18 % noise) | **0.578** (thr=1, B=100, k=8) | 0.530 (k=6) | 0.506 (k=6) | 0.536 (k=6) |
| kr-vs-kp | 0.023 | **0.045** (thr=2, B=10, k=6) | 0.028 (k=4) | 0.030 (k=4) | 0.033 (k=2) |
| horse-colic | 0.122 (Manhattan, eps=8, ms=5; 69 % noise) | 0.142 (thr=0.5, B=50, k=2) | 0.055 (k=6) | 0.046 (k=10) | **0.159** (k=4) |

Full tables, with mean ± std of every metric and configuration, are in `results/summary.csv` and
`results/best_configs.csv`.

**Answers to the questions in the assignment (for these three algorithms):**

- *Which parameters have the greatest influence?*
  - DBSCAN: `eps`, then `min_samples`. The metric only matters on numerical data, and `algorithm` only changes
    the time.
  - BIRCH: `threshold`, also when a final k is used. `branching_factor` only affects time.
  - CLARANS: k determines the result. `maxneighbor` and `numlocal` mainly affect cost, stability and time.
- *Same parameters across data sets?* No for DBSCAN and BIRCH. Their parameters are distances, so they depend on
  the dimension and the encoding: eps≈1.5 (Euclidean) on satimage vs 2.5–3.8 on horse-colic, and the useful
  BIRCH threshold collapses kr-vs-kp at 4 but not satimage. CLARANS search parameters can be shared; only k
  changes.
- *Do the validation metrics agree?* No. Silhouette and Davies–Bouldin systematically prefer few, compact clusters
  (and, for DBSCAN, settings with a lot of noise). ARI and F-measure prefer the number of real classes. Purity is
  not reliable because it increases with k.
- *Agreement with the known classes?* Only satimage shows a clear relation between clusters and classes (ARI up
  to 0.58, F-measure 0.74). In kr-vs-kp the class (whether white can win) depends on interactions between board
  features, not on the overall similarity of positions, so no method exceeds ARI 0.05. In horse-colic, the
  surgical-lesion label is only weakly reflected in the clinical attributes. CLARANS and BIRCH capture it best.
- *Best algorithm by data type:*
  - Numerical (satimage): BIRCH (best ARI and fast), with CLARANS and K-Means close behind.
  - Mixed (horse-colic): CLARANS, the most robust thanks to medoids and L1-type cost.
  - Nominal (kr-vs-kp): no method finds the class structure. DBSCAN is the least suitable, because the discrete
    distances make `eps` a "number of differing attributes" threshold.
- *Execution time.* DBSCAN with `brute`/`auto` and BIRCH with a large threshold take hundredths of a second. The
  CLARANS time grows linearly with k and with `maxneighbor`, and it is the slowest of our partitional methods.
  Tree-based DBSCAN searches become slower than brute force with more than about 30 dimensions.

## Figures referenced
- `results/figures/k_distance_plots.png`: 5-NN distance per metric, used to choose `eps`.
- `results/figures/sensitivity_<dataset>.png`: DBSCAN `eps` vs clusters, noise ratio, Silhouette and ARI.
- `results/figures/birch_threshold_no_k.png`: BIRCH without global step, threshold vs clusters, ARI, Silhouette
  and time.
- `results/figures/birch_threshold_k.png`: BIRCH with a final k, threshold vs clusters, ARI, Silhouette and
  Davies–Bouldin.
- `results/figures/clarans_k_maxneighbor.png`: CLARANS k vs cost, ARI, Silhouette and time for each `maxneighbor`.

## References
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in
  large spatial databases with noise. *KDD-96*, 226–231.
- Zhang, T., Ramakrishnan, R., & Livny, M. (1996). BIRCH: An efficient data clustering method for very large
  databases. *ACM SIGMOD Record*, 25(2), 103–114.
- Ng, R. T., & Han, J. (1994). Efficient and effective clustering methods for spatial data mining. *VLDB '94*,
  144–155.
- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley.
