# Support Vector Machines (SVM)

## Table of Contents
1. [Overview](#overview)
2. [Core Theory — Maximum Margin Classifier](#core-theory--maximum-margin-classifier)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Hard Margin vs. Soft Margin](#hard-margin-vs-soft-margin)
5. [The Dual Problem and Support Vectors](#the-dual-problem-and-support-vectors)
6. [The Kernel Trick](#the-kernel-trick)
   - [Common Kernels](#common-kernels)
7. [SVM for Regression (SVR)](#svm-for-regression-svr)
8. [Multi-Class SVM](#multi-class-svm)
9. [Hyperparameters](#hyperparameters)
10. [Strengths and Weaknesses](#strengths-and-weaknesses)
11. [Role in ML and LLMs](#role-in-ml-and-llms)

---

## Overview

Support Vector Machines (SVMs) are supervised learning models for classification and regression. They are grounded in the principle of finding the **decision boundary that maximizes the margin** between classes.

SVMs were dominant in machine learning throughout the 1990s and 2000s before being surpassed by deep learning on large datasets. However, they remain state-of-the-art on small-to-medium datasets, particularly in high-dimensional spaces, and the **kernel trick** they introduced is a profound concept that influenced the entire field.

---

## Core Theory — Maximum Margin Classifier

Given a linearly separable binary classification problem with labels `y ∈ {-1, +1}`:

Any hyperplane can be written as:
```
wᵀx + b = 0
```

There are infinitely many hyperplanes that correctly separate the two classes. SVM finds the unique one with the **largest margin** — the maximum distance between the hyperplane and the nearest data points of each class.

The **margin** is defined as:
```
margin = 2 / ||w||
```

Maximizing the margin is equivalent to minimizing `||w||`, subject to the constraint that all points are correctly classified.

**Why maximum margin?**
- A larger margin provides better generalization to unseen data
- Intuitively, the boundary is farther from both classes, leaving more room for new points
- This is supported by **Statistical Learning Theory** (VC dimension / PAC learning)

---

## Mathematical Formulation

**Primal Optimization Problem (Hard Margin):**

```
minimize:    (1/2) * ||w||²

subject to:  yᵢ(wᵀxᵢ + b) ≥ 1,  for all i = 1,...,m
```

The constraint ensures every point is at least distance `1/||w||` from the boundary.

The points that lie exactly on the margin boundary — those where `yᵢ(wᵀxᵢ + b) = 1` — are called **support vectors**. They are the only training points that matter; removing all other points does not change the solution.

---

## Hard Margin vs. Soft Margin

**Hard Margin SVM** requires perfect linear separability. This is too restrictive for real data.

**Soft Margin SVM** introduces slack variables `ξᵢ ≥ 0` that allow points to violate the margin:

```
minimize:    (1/2) * ||w||² + C * Σᵢ ξᵢ

subject to:  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ,  for all i
             ξᵢ ≥ 0
```

The hyperparameter `C` controls the trade-off:
- **Large C**: Penalizes margin violations heavily → narrow margin, fewer misclassifications → more prone to overfitting
- **Small C**: Tolerates more violations → wide margin → more regularization

The soft margin SVM loss function can be rewritten using the **Hinge Loss**:
```
J(w) = (λ/2) * ||w||² + (1/m) * Σᵢ max(0, 1 - yᵢ(wᵀxᵢ + b))
```

---

## The Dual Problem and Support Vectors

Using **Lagrangian duality**, the primal problem can be rewritten as a dual:

```
maximize:   Σᵢ αᵢ - (1/2) * Σᵢ Σⱼ αᵢαⱼyᵢyⱼ (xᵢᵀxⱼ)

subject to: 0 ≤ αᵢ ≤ C,   Σᵢ αᵢyᵢ = 0
```

Key observations:
1. The data only appears as **dot products** `xᵢᵀxⱼ` — this enables the kernel trick
2. Most `αᵢ = 0` — only **support vectors** have `αᵢ > 0`
3. The solution `w = Σᵢ αᵢyᵢxᵢ` is a linear combination of support vectors only
4. Prediction: `sign(Σᵢ αᵢyᵢ(xᵢᵀx) + b)`

This sparsity is a key property: the model only "remembers" the most informative examples.

---

## The Kernel Trick

The dual formulation only uses dot products `⟨xᵢ, xⱼ⟩`. The **kernel trick** replaces these dot products with a **kernel function** `K(xᵢ, xⱼ)` that implicitly computes dot products in a **higher-dimensional feature space**:

```
K(xᵢ, xⱼ) = φ(xᵢ)ᵀ φ(xⱼ)
```

Where `φ` is a (potentially infinite-dimensional) feature map. We never need to compute `φ(x)` explicitly — only `K(xᵢ, xⱼ)`.

This allows SVM to find **nonlinear decision boundaries** in the original space while solving a **linear SVM** in the transformed space.

### Common Kernels

| Kernel | Formula | Use Case |
|---|---|---|
| Linear | `xᵢᵀxⱼ` | Linearly separable data, high-dimensional text |
| Polynomial | `(γ xᵢᵀxⱼ + r)^d` | Moderate nonlinearity |
| RBF (Gaussian) | `exp(-γ ||xᵢ - xⱼ||²)` | General nonlinear data; most widely used |
| Sigmoid | `tanh(γ xᵢᵀxⱼ + r)` | Resembles neural networks |

**RBF Kernel intuition:** It measures similarity based on Euclidean distance. It corresponds to an infinite-dimensional polynomial feature space — any smooth decision boundary can be represented given enough data.

A kernel must satisfy **Mercer's condition**: the kernel matrix `K` must be positive semi-definite.

---

## SVM for Regression (SVR)

Support Vector Regression uses an **ε-insensitive tube** around the predictions:

- Points inside the tube (error < ε) are not penalized
- Points outside the tube incur linear loss proportional to how far they are

```
minimize:   (1/2) * ||w||² + C * Σᵢ (ξᵢ + ξᵢ*)

subject to: yᵢ - (wᵀxᵢ + b) ≤ ε + ξᵢ
            (wᵀxᵢ + b) - yᵢ ≤ ε + ξᵢ*
```

---

## Multi-Class SVM

SVMs are inherently binary classifiers. Extensions to K classes:

| Strategy | Method |
|---|---|
| One-vs-One (OvO) | Train K*(K-1)/2 binary classifiers; vote |
| One-vs-Rest (OvR) | Train K binary classifiers; pick highest score |
| Multi-class SVM | Weston-Watkins or Crammer-Singer formulation |

---

## Hyperparameters

| Hyperparameter | Effect |
|---|---|
| `C` | Regularization strength (inverse); controls margin width vs. misclassification penalty |
| `kernel` | Defines the feature space (linear, rbf, poly, sigmoid) |
| `γ` (RBF/Poly/Sigmoid) | Kernel coefficient; higher γ = more complex decision boundary |
| `degree` (Polynomial) | Polynomial degree |
| `ε` (SVR) | Width of the insensitive tube |

---

## Strengths and Weaknesses

**Strengths:**
- Effective in high-dimensional spaces (e.g., text, genomics)
- Memory-efficient: only support vectors are stored
- Versatile via kernel choice
- Strong theoretical guarantees (maximum margin = good generalization)
- Works well with small datasets

**Weaknesses:**
- Slow training on large datasets: O(m²) to O(m³) complexity
- Does not produce probability estimates directly (requires Platt scaling)
- Sensitive to feature scaling — normalization is essential
- Kernel and hyperparameter selection requires careful cross-validation
- Black box when using nonlinear kernels

---

## Role in ML and LLMs

SVMs are rarely used inside modern LLM architectures, but their legacy is significant:

- **Kernel methods** inspired the study of **neural tangent kernels** (NTK) — which reveals that infinitely wide neural networks correspond to kernel machines
- The **hinge loss** is used in some contrastive learning objectives and ranking losses in retrieval systems
- **Support vector intuition** — the idea that only a few "critical" examples determine the boundary — conceptually parallels **attention mechanisms**, which focus on the most relevant tokens
- Pre-deep-learning NLP relied heavily on SVMs with bag-of-words and n-gram features for sentiment analysis, spam detection, and text classification
