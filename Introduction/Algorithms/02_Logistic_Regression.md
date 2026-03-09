# Logistic Regression

## Table of Contents
1. [Overview](#overview)
2. [Core Theory](#core-theory)
3. [The Sigmoid Function](#the-sigmoid-function)
4. [The Model](#the-model)
5. [Cost Function — Binary Cross-Entropy](#cost-function--binary-cross-entropy)
6. [Gradient Descent for Logistic Regression](#gradient-descent-for-logistic-regression)
7. [Multi-Class Extension — Softmax](#multi-class-extension--softmax)
8. [Decision Boundary](#decision-boundary)
9. [Regularization](#regularization)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Strengths and Weaknesses](#strengths-and-weaknesses)
12. [Role in ML and LLMs](#role-in-ml-and-llms)

---

## Overview

Logistic Regression is the standard algorithm for **binary classification**. Despite its name, it is a classification model — it predicts the **probability** that a given input belongs to a particular class.

It extends linear regression by passing the linear output through a nonlinear **sigmoid** function, squashing predictions into the range (0, 1) so they can be interpreted as probabilities.

Logistic regression is the simplest probabilistic classifier and is fundamental to understanding neural network output layers and loss functions.

---

## Core Theory

The generative story behind logistic regression is rooted in the **Bernoulli distribution**. We model:

```
P(y=1 | x; w) = σ(wᵀx)
P(y=0 | x; w) = 1 - σ(wᵀx)
```

Where `σ` is the sigmoid function. This can be written compactly as:

```
P(y | x; w) = σ(wᵀx)^y * (1 - σ(wᵀx))^(1-y)
```

The learning objective is to find weights `w` that **maximize the likelihood** of the observed labels — which is equivalent to minimizing **Binary Cross-Entropy** loss.

---

## The Sigmoid Function

The sigmoid (logistic) function maps any real number to (0, 1):

```
σ(z) = 1 / (1 + e^(-z))
```

Key properties:
- `σ(0) = 0.5`
- `σ(z) → 1` as `z → +∞`
- `σ(z) → 0` as `z → -∞`
- Derivative: `σ'(z) = σ(z) * (1 - σ(z))`

The derivative property is crucial — it allows efficient backpropagation. However, the sigmoid is prone to the **vanishing gradient problem** for very large or small inputs, where the gradient saturates near 0.

---

## The Model

```
z = w0 + w1*x1 + ... + wn*xn  (linear combination)
ŷ = σ(z)                       (predicted probability)
```

Classification rule:
```
predict class 1 if ŷ >= 0.5
predict class 0 if ŷ < 0.5
```

The 0.5 threshold corresponds to `z = 0`, which defines the decision boundary.

---

## Cost Function — Binary Cross-Entropy

The MSE loss used in linear regression is non-convex when combined with the sigmoid — it creates many local minima. Instead, we use **Binary Cross-Entropy (Log Loss)**:

```
J(w) = -(1/m) * Σ [ yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ) ]
```

Intuition:
- When `y=1`: loss = `-log(ŷ)` — penalizes heavily when ŷ is close to 0
- When `y=0`: loss = `-log(1-ŷ)` — penalizes heavily when ŷ is close to 1
- Perfect predictions yield zero loss

This function is **convex** when combined with the sigmoid, guaranteeing a global minimum.

---

## Gradient Descent for Logistic Regression

The gradient of binary cross-entropy with respect to the weights has a surprisingly clean form:

```
∂J/∂w = (1/m) * Xᵀ(ŷ - y)
```

This is structurally identical to the gradient for linear regression — the sigmoid and log cancel out elegantly. The update rule:

```
w := w - α * (1/m) * Xᵀ(ŷ - y)
```

---

## Multi-Class Extension — Softmax

For `K` classes, logistic regression generalizes to **Softmax Regression** (also called Multinomial Logistic Regression):

```
P(y=k | x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
```

The softmax function produces a probability distribution over all `K` classes:
- All outputs are in (0, 1)
- They sum to exactly 1

The corresponding loss is **Categorical Cross-Entropy**:

```
J(W) = -(1/m) * Σᵢ Σₖ yᵢₖ * log(P(y=k | xᵢ))
```

Where `yᵢₖ` is the one-hot encoded true label.

---

## Decision Boundary

The decision boundary is the set of points where `P(y=1|x) = 0.5`, which occurs when `z = wᵀx = 0`.

This means:
- For 2 features: the boundary is a **line**: `w0 + w1*x1 + w2*x2 = 0`
- For n features: the boundary is a **hyperplane**
- Logistic regression can only produce **linear decision boundaries**

To model nonlinear boundaries, features must be manually engineered (e.g., polynomial features), or a neural network must be used.

---

## Regularization

Identical to linear regression, L1 and L2 penalties can be applied:

| Type | Penalty | Effect |
|---|---|---|
| L2 (Ridge) | `λ * ||w||²` | Shrinks all weights; keeps all features |
| L1 (Lasso) | `λ * ||w||₁` | Drives some weights to zero; feature selection |
| Elastic Net | Combination of L1 + L2 | Best of both |

The regularization parameter `C = 1/λ` is common in libraries like scikit-learn — smaller `C` means stronger regularization.

---

## Evaluation Metrics

| Metric | Formula | Notes |
|---|---|---|
| Accuracy | `(TP + TN) / Total` | Misleading on imbalanced datasets |
| Precision | `TP / (TP + FP)` | Of predicted positives, how many are correct? |
| Recall | `TP / (TP + FN)` | Of actual positives, how many are found? |
| F1 Score | `2 * P*R / (P+R)` | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve | Threshold-independent measure of separability |
| Log Loss | Binary cross-entropy | Measures quality of probability estimates |

---

## Strengths and Weaknesses

**Strengths:**
- Outputs calibrated probabilities, not just class labels
- Fast to train and predict
- Highly interpretable — log-odds interpretation of weights
- Works well on linearly separable data

**Weaknesses:**
- Cannot model nonlinear decision boundaries
- Assumes features are independent (naive assumption)
- Sensitive to correlated features
- Underperforms on complex, high-dimensional data

---

## Role in ML and LLMs

Logistic regression and the softmax function are foundational to modern deep learning:

- The **output layer** of any classification neural network applies softmax + categorical cross-entropy
- The **attention mechanism** in Transformers applies softmax to compute attention weights
- **Language model heads** (predicting next tokens) are linear projections followed by softmax over the full vocabulary
- **Probing classifiers** for LLMs are typically logistic regression models fitted on frozen embeddings

The binary cross-entropy loss derived here is the same loss that trains classifiers in BERT-style models and discriminators in GANs. The softmax function is arguably the most important nonlinearity in modern LLMs.
