# Decision Trees and Ensemble Methods

## Table of Contents
1. [Overview](#overview)
2. [Decision Tree Core Theory](#decision-tree-core-theory)
3. [Splitting Criteria](#splitting-criteria)
   - [Information Gain / Entropy](#information-gain--entropy)
   - [Gini Impurity](#gini-impurity)
   - [Variance Reduction (Regression)](#variance-reduction-regression)
4. [Tree Construction Algorithm (CART)](#tree-construction-algorithm-cart)
5. [Overfitting and Pruning](#overfitting-and-pruning)
6. [Ensemble Methods Overview](#ensemble-methods-overview)
7. [Bagging and Random Forests](#bagging-and-random-forests)
8. [Boosting and Gradient Boosting](#boosting-and-gradient-boosting)
9. [XGBoost Key Innovations](#xgboost-key-innovations)
10. [Comparison Table](#comparison-table)
11. [Strengths and Weaknesses](#strengths-and-weaknesses)
12. [Role in ML and LLMs](#role-in-ml-and-llms)

---

## Overview

A **Decision Tree** is a non-parametric, hierarchical model that partitions the feature space into regions by recursively applying binary splitting rules. Each internal node tests a feature, each branch represents an outcome, and each leaf node holds a prediction.

**Ensemble Methods** combine many weak learners (usually trees) into a single strong predictor. The two dominant strategies are:
- **Bagging** (Bootstrap AGGregatING): trains models in parallel on bootstrapped datasets — e.g., Random Forest
- **Boosting**: trains models sequentially, each correcting the errors of the previous — e.g., Gradient Boosted Trees, XGBoost

---

## Decision Tree Core Theory

At its core, a decision tree answers a sequence of yes/no questions about features to arrive at a prediction.

Example structure:
```
           [Age > 30?]
          /            \
    [Income > 50k?]   Predict: No
      /         \
Predict: Yes   Predict: No
```

The key design question is: **how do we choose which feature and threshold to split on at each node?**

The answer is to choose the split that maximizes the **information gain** — i.e., the split that results in the most "pure" (homogeneous) child nodes.

---

## Splitting Criteria

### Information Gain / Entropy

**Entropy** measures the impurity of a node. For a node with `K` classes:

```
H(S) = -Σₖ pₖ * log₂(pₖ)
```

Where `pₖ` is the fraction of samples in class `k`.

- `H = 0`: perfectly pure node (all one class)
- `H = log₂(K)`: maximally impure node (equal mix of all classes)

**Information Gain** for a split on feature `A`:

```
IG(S, A) = H(S) - Σᵥ (|Sᵥ|/|S|) * H(Sᵥ)
```

We choose the feature and threshold that maximizes `IG`.

### Gini Impurity

An alternative to entropy, often faster to compute:

```
Gini(S) = 1 - Σₖ pₖ²
```

- `Gini = 0`: perfectly pure
- `Gini = 0.5`: maximally impure (binary case)

In practice, Gini and Entropy produce very similar trees. Gini is default in scikit-learn's `DecisionTreeClassifier`.

### Variance Reduction (Regression)

For regression trees, the splitting criterion is to minimize the weighted variance of child nodes:

```
Gain = Var(S) - Σᵥ (|Sᵥ|/|S|) * Var(Sᵥ)
```

Leaf predictions are the **mean** of all training samples in that leaf.

---

## Tree Construction Algorithm (CART)

CART (Classification and Regression Trees) is the standard algorithm:

```
function BuildTree(S, depth):
    if stopping_condition(S, depth):
        return LeafNode(predict(S))

    best_feature, best_threshold = argmax over all (f, t) of IG(S, f, t)

    S_left  = {x ∈ S | x[best_feature] <= best_threshold}
    S_right = {x ∈ S | x[best_feature] >  best_threshold}

    left_child  = BuildTree(S_left,  depth+1)
    right_child = BuildTree(S_right, depth+1)

    return InternalNode(best_feature, best_threshold, left_child, right_child)
```

Stopping conditions:
- Maximum depth reached
- Minimum samples per node
- Minimum information gain threshold
- Node is already pure

Time complexity: O(n * m * log(m)) per node, where `n` = features, `m` = samples.

---

## Overfitting and Pruning

A fully grown tree memorizes the training data (zero training error) but generalizes poorly. Two main strategies:

**Pre-pruning (Early Stopping):**
- `max_depth`: limit tree depth
- `min_samples_split`: minimum samples to allow a split
- `min_samples_leaf`: minimum samples in a leaf
- `min_impurity_decrease`: minimum gain to justify a split

**Post-pruning (Cost-Complexity Pruning):**
Prune subtrees by minimizing:
```
R_α(T) = R(T) + α * |T|
```
Where `R(T)` is the misclassification rate, `|T|` is the number of leaves, and `α` is a complexity parameter. As `α` increases, more pruning occurs.

---

## Ensemble Methods Overview

A single decision tree is a high-variance model — small changes in data can produce very different trees. Ensembles address this by combining many trees.

**Bias-Variance Decomposition:**
```
Total Error = Bias² + Variance + Irreducible Noise
```

- Bagging primarily reduces **variance** (averaging out overfitting)
- Boosting primarily reduces **bias** (correcting systematic errors)

---

## Bagging and Random Forests

**Bagging:**
1. Draw `B` bootstrapped datasets from the training data (sample with replacement)
2. Train a full decision tree on each
3. Aggregate predictions by majority vote (classification) or averaging (regression)

**Random Forests** extend bagging with **feature randomization**:
- At each split, only a random subset of `√n` features (classification) or `n/3` features (regression) are considered
- This decorrelates the trees, reducing ensemble variance further

Key hyperparameters:
| Parameter | Effect |
|---|---|
| `n_estimators` | More trees = lower variance, diminishing returns |
| `max_features` | Lower = more decorrelation but higher bias per tree |
| `max_depth` | Shallower = higher bias, lower variance |

**Out-of-Bag (OOB) Error:** Since each tree only sees ~63% of data (bootstrap), the remaining 37% (OOB samples) can be used as a free validation set without cross-validation.

---

## Boosting and Gradient Boosting

**AdaBoost (Adaptive Boosting):**
1. Assign equal weights to all training samples
2. Train a weak learner (shallow tree)
3. Increase weights of misclassified samples
4. Repeat, building `T` trees
5. Final prediction is a weighted vote

**Gradient Boosting:**
A more general framework where each tree is trained to predict the **negative gradient** (residuals) of the loss function:

```
Fₜ(x) = Fₜ₋₁(x) + α * hₜ(x)
```

Where `hₜ` is a tree fitted to the residuals `yᵢ - Fₜ₋₁(xᵢ)`.

For MSE loss, residuals = actual errors. For other losses, the "residuals" are the gradient of the loss w.r.t. the current prediction.

This makes gradient boosting **a gradient descent in function space**.

---

## XGBoost Key Innovations

XGBoost (eXtreme Gradient Boosting) adds several important improvements:

1. **Second-order approximation:** Uses both gradient `g` and Hessian `h` of the loss for better splits:
   ```
   Gain = 0.5 * [ G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L+G_R)²/(H_L+H_R+λ) ] - γ
   ```

2. **Regularization:** L1 (`α`) and L2 (`λ`) penalties on leaf weights are built into the objective

3. **Sparsity-aware split finding:** Handles missing values and sparse features efficiently

4. **Column block structure:** Sorts features once and reuses for faster split finding

5. **Shrinkage:** Multiplies each tree's output by a learning rate `η` to prevent overfitting

---

## Comparison Table

| Model | Training | Interpretability | Handles Non-Linearity | Overfitting Risk |
|---|---|---|---|---|
| Single Decision Tree | Fast | High | Yes | Very High |
| Random Forest | Parallel, Moderate | Low | Yes | Low |
| Gradient Boosting | Sequential, Slow | Low | Yes | Medium (with tuning) |
| XGBoost | Fast (optimized) | Low | Yes | Low (with regularization) |

---

## Strengths and Weaknesses

**Strengths:**
- No feature scaling required
- Handles mixed data types (categorical + numerical)
- Robust to outliers (especially Random Forest)
- Captures nonlinear relationships without feature engineering
- Feature importance scores are interpretable

**Weaknesses:**
- Decision boundaries are axis-aligned (piecewise constant)
- Extrapolation: tree models cannot predict beyond training data range
- Gradient boosting is sequential and harder to parallelize
- Requires careful hyperparameter tuning (especially boosting)

---

## Role in ML and LLMs

While tree-based models are not used inside LLMs themselves, they remain dominant in:
- **Tabular data tasks** where they frequently outperform neural networks
- **Feature importance analysis** for understanding what drives model predictions
- **Hybrid systems** where gradient boosted trees handle structured inputs fed into or alongside LLMs
- **Retrieval and ranking** in information retrieval pipelines that precede LLM generation

The concept of **boosting residuals** also parallels the **residual connections** in Transformer architectures — both are about iteratively refining an approximation.
