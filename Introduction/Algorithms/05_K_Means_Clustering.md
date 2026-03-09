# K-Means Clustering

## Table of Contents
1. [Overview](#overview)
2. [Core Theory](#core-theory)
3. [The K-Means Algorithm (Lloyd's Algorithm)](#the-k-means-algorithm-lloyds-algorithm)
4. [Objective Function](#objective-function)
5. [Convergence Properties](#convergence-properties)
6. [Choosing K](#choosing-k)
   - [Elbow Method](#elbow-method)
   - [Silhouette Score](#silhouette-score)
7. [Initialization — K-Means++](#initialization--k-means)
8. [Limitations and Failure Modes](#limitations-and-failure-modes)
9. [Variants](#variants)
10. [Gaussian Mixture Models (GMMs)](#gaussian-mixture-models-gmms)
11. [Strengths and Weaknesses](#strengths-and-weaknesses)
12. [Role in ML and LLMs](#role-in-ml-and-llms)

---

## Overview

K-Means is the most widely used unsupervised clustering algorithm. Its goal is to partition `m` data points into `K` clusters such that each point belongs to the cluster with the **nearest centroid**, minimizing the total within-cluster variance.

Unlike supervised learning, K-Means operates without labels — it discovers structure purely from the data's geometry. It is fast, scalable, and simple to implement, making it a staple preprocessing and analysis tool.

---

## Core Theory

Given a dataset `{x₁, ..., xₘ}` with `xᵢ ∈ ℝⁿ`, we want to find:
- `K` cluster centroids `{μ₁, ..., μₖ}`
- A cluster assignment function `c : xᵢ → {1, ..., K}`

Such that points are grouped by proximity in feature space.

The key assumption is that clusters are **spherical and roughly equally sized**. This is because K-Means minimizes Euclidean distance, which defines hyperspherical regions of influence around each centroid (Voronoi tessellation).

---

## The K-Means Algorithm (Lloyd's Algorithm)

The standard algorithm alternates between two steps until convergence:

```
Initialize: Choose K centroids μ₁, ..., μₖ (randomly or via K-Means++)

Repeat until convergence:

    Step 1 — Assignment:
        For each point xᵢ:
            c(i) = argmin_k ||xᵢ - μₖ||²

    Step 2 — Update:
        For each cluster k:
            μₖ = (1/|Cₖ|) * Σᵢ∈Cₖ xᵢ
            (recompute centroid as the mean of assigned points)
```

This is an instance of **Expectation-Maximization (EM)**:
- **E-step** (Assignment): fix centroids, assign points to nearest centroid
- **M-step** (Update): fix assignments, move centroids to cluster means

---

## Objective Function

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also called **inertia**:

```
J = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²
```

This is the sum of squared Euclidean distances from each point to its assigned centroid.

Each iteration of Lloyd's algorithm is **guaranteed to decrease or maintain** J:
- The assignment step minimizes J over assignments given fixed centroids
- The update step minimizes J over centroids given fixed assignments (the mean is the minimizer of squared distances)

---

## Convergence Properties

- **Convergence is guaranteed** — WCSS is non-increasing and bounded below by 0
- **Global optimum is NOT guaranteed** — K-Means converges to a local minimum
- The number of iterations is typically small in practice (10–100 iterations)
- Time complexity per iteration: O(m * K * n)

Because of local optima, K-Means is typically run **multiple times** with different initializations, and the best result (lowest WCSS) is kept.

---

## Choosing K

The number of clusters `K` is a hyperparameter that must be specified. Several methods help choose it:

### Elbow Method

Plot WCSS vs. K. The optimal K is where the rate of decrease sharply slows — the "elbow":

```
K:    1    2    3    4    5    6
WCSS: 500  300  150  130  125  123
                  ^
               Elbow at K=3
```

The "elbow" is often ambiguous in practice.

### Silhouette Score

For each point `i`, compute:
```
a(i) = mean distance from i to all other points in its cluster
b(i) = mean distance from i to all points in the nearest other cluster

s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

- `s(i) = 1`: point is well-matched to its cluster
- `s(i) = 0`: point is on the boundary between clusters
- `s(i) = -1`: point may be in the wrong cluster

Average `s(i)` over all points to get the overall Silhouette Score. Choose K that maximizes it.

Other methods: Gap statistic, Bayesian Information Criterion (BIC), domain knowledge.

---

## Initialization — K-Means++

Random initialization can lead to poor local optima. **K-Means++** provides a smarter initialization:

```
1. Choose first centroid μ₁ uniformly at random from the data
2. For each subsequent centroid μₖ:
   a. Compute D(xᵢ)² = min_{j<k} ||xᵢ - μⱼ||² for all points
   b. Choose μₖ = xᵢ with probability proportional to D(xᵢ)²
3. Run standard K-Means
```

This spreads initial centroids far apart, avoiding degenerate initializations. K-Means++ has an approximation guarantee: its expected WCSS is O(log K) times the optimal WCSS.

---

## Limitations and Failure Modes

| Problem | Description | Solution |
|---|---|---|
| Non-spherical clusters | K-Means can't cluster elongated or crescent-shaped clusters | DBSCAN, Spectral Clustering |
| Unequal cluster sizes | Large clusters are split, small ones are merged | GMMs |
| Unequal cluster densities | Denser clusters dominate | DBSCAN, GMMs |
| Sensitivity to outliers | Outliers pull centroids | K-Medoids, use median instead of mean |
| Must specify K | No automatic cluster discovery | DBSCAN (density-based), Hierarchical |
| Scale sensitivity | Features with large ranges dominate | Always normalize features first |

---

## Variants

| Variant | Key Difference |
|---|---|
| Mini-Batch K-Means | Updates centroids using mini-batches; much faster for large datasets |
| K-Medoids (PAM) | Centroids must be actual data points; more robust to outliers |
| Fuzzy C-Means | Soft assignments — each point has a degree of membership to each cluster |
| Bisecting K-Means | Hierarchical variant: repeatedly splits the largest cluster |
| DBSCAN | Density-based; finds arbitrary-shaped clusters; detects outliers |

---

## Gaussian Mixture Models (GMMs)

GMMs are the **probabilistic generalization** of K-Means. Instead of hard cluster assignments, each cluster is a Gaussian distribution:

```
p(x) = Σₖ πₖ * N(x | μₖ, Σₖ)
```

Where:
- `πₖ` is the mixing proportion (prior probability of cluster k)
- `μₖ` is the cluster mean
- `Σₖ` is the cluster covariance matrix

GMMs are trained via the **EM algorithm**:
- **E-step**: compute soft responsibilities `r(i,k) = P(cluster k | xᵢ)`
- **M-step**: update `πₖ`, `μₖ`, `Σₖ` using weighted statistics

K-Means is a special case of GMM where covariances are forced to be `σ²I` (spherical) and assignments are hard.

---

## Strengths and Weaknesses

**Strengths:**
- Very fast and scalable (O(m * K * n))
- Simple to implement and understand
- Works well when clusters are spherical and well-separated
- Useful for data compression and vector quantization

**Weaknesses:**
- Must specify K in advance
- Sensitive to initialization (mitigated by K-Means++)
- Only finds convex, spherical clusters
- Sensitive to feature scale
- Not robust to outliers

---

## Role in ML and LLMs

K-Means has direct and important applications in modern NLP and LLMs:

- **Vector Quantization (VQ)**: K-Means is used to create discrete codebooks in VQ-VAEs, which are fundamental to audio generation models (WaveNet, EnCodec) and image generation
- **Tokenization**: Byte-Pair Encoding (BPE) — the tokenization algorithm behind GPT models — shares conceptual overlap with iterative clustering of character sequences
- **Word/Sentence clustering**: K-Means applied to word embeddings or sentence embeddings reveals semantic clusters
- **Retrieval system analysis**: Clustering document embeddings to understand corpus structure
- **Probing**: Clustering LLM hidden states to analyze what information is encoded at different layers
- **Quantization for model compression**: K-Means based weight quantization reduces model size for deployment
