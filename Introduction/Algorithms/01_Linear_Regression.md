# Linear Regression

## Table of Contents
1. [Overview](#overview)
2. [Core Theory](#core-theory)
3. [The Model](#the-model)
4. [Cost Function](#cost-function)
5. [Gradient Descent Optimization](#gradient-descent-optimization)
6. [Analytical Solution (Normal Equation)](#analytical-solution-normal-equation)
7. [Assumptions](#assumptions)
8. [Regularization](#regularization)
   - [Ridge (L2)](#ridge-l2)
   - [Lasso (L1)](#lasso-l1)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Strengths and Weaknesses](#strengths-and-weaknesses)
11. [Role in ML and LLMs](#role-in-ml-and-llms)

---

## Overview

Linear Regression is one of the oldest and most fundamental supervised learning algorithms. Its goal is to model a continuous relationship between one or more input features and a continuous output variable by fitting a straight line (or hyperplane in higher dimensions) through the data.

Despite its simplicity, it forms the conceptual backbone for many advanced models — including neural networks, which can be viewed as stacked compositions of linear transformations followed by nonlinear activations.

---

## Core Theory

The central hypothesis is that the target variable `y` is a **linear combination** of the input features `x`, plus some noise:

```
y = w0 + w1*x1 + w2*x2 + ... + wn*xn + ε
```

Where:
- `w0` is the bias (intercept)
- `w1 ... wn` are the learned weights (coefficients)
- `ε` is irreducible Gaussian noise: ε ~ N(0, σ²)

In vector form:
```
y = wᵀx + ε
```

This noise assumption is critical — it is what motivates the use of **Mean Squared Error (MSE)** as the loss function (minimizing MSE is equivalent to Maximum Likelihood Estimation under Gaussian noise).

---

## The Model

Given a dataset of `m` training examples with `n` features:

- `X` is the design matrix of shape `(m, n+1)` — the +1 accounts for a bias column of ones
- `y` is the target vector of shape `(m, 1)`
- `w` is the weight vector of shape `(n+1, 1)`

Prediction:
```
ŷ = X · w
```

---

## Cost Function

The **Mean Squared Error (MSE)** cost function measures the average squared difference between predictions and true labels:

```
J(w) = (1/2m) * Σ (ŷᵢ - yᵢ)²
     = (1/2m) * ||Xw - y||²
```

The `1/2` factor is a convenience that simplifies the derivative. This cost function is:
- Convex (bowl-shaped) — guarantees a single global minimum
- Differentiable everywhere — enabling gradient-based optimization

---

## Gradient Descent Optimization

The gradient of the cost with respect to the weights is:

```
∂J/∂w = (1/m) * Xᵀ(Xw - y)
```

The weight update rule:
```
w := w - α * ∂J/∂w
```

Where `α` is the **learning rate**, a hyperparameter that controls step size.

### Variants
| Variant | Batch Size | Notes |
|---|---|---|
| Batch Gradient Descent | Full dataset | Stable but slow for large datasets |
| Stochastic Gradient Descent (SGD) | 1 sample | Noisy but fast; can escape local minima |
| Mini-Batch Gradient Descent | Small batch (e.g. 32–256) | Balance of stability and speed — used in practice |

---

## Analytical Solution (Normal Equation)

When the dataset is small enough, the optimal weights can be solved directly without iteration:

```
w* = (XᵀX)⁻¹ Xᵀy
```

This is derived by setting `∂J/∂w = 0` and solving. It is the **Ordinary Least Squares (OLS)** solution.

**Limitations:**
- Computing `(XᵀX)⁻¹` is O(n³) — prohibitive for large `n`
- `XᵀX` must be invertible (i.e., features cannot be linearly dependent)

---

## Assumptions

Linear Regression makes several key statistical assumptions:

| Assumption | Description |
|---|---|
| Linearity | The relationship between X and y is linear |
| Independence | Observations are independent of each other |
| Homoscedasticity | Variance of errors is constant across all values of X |
| Normality of errors | Residuals are normally distributed |
| No multicollinearity | Features are not highly correlated with each other |

Violations of these assumptions do not always break the model but can lead to poor generalization and unreliable confidence intervals.

---

## Regularization

Regularization adds a penalty term to the cost function to prevent overfitting by shrinking the weights toward zero.

### Ridge (L2)

Adds the squared magnitude of weights as a penalty:

```
J(w) = (1/2m) * ||Xw - y||² + λ * ||w||²
```

- Shrinks all weights but rarely sets them to exactly zero
- The closed-form solution becomes: `w* = (XᵀX + λI)⁻¹ Xᵀy`
- Useful when many small features are relevant

### Lasso (L1)

Adds the absolute magnitude of weights as a penalty:

```
J(w) = (1/2m) * ||Xw - y||² + λ * ||w||₁
```

- Tends to produce **sparse solutions** — drives irrelevant weights to exactly zero
- Performs implicit feature selection
- No closed-form solution; requires iterative methods

---

## Evaluation Metrics

| Metric | Formula | Notes |
|---|---|---|
| Mean Absolute Error (MAE) | `(1/m) Σ |ŷ - y|` | Robust to outliers |
| Mean Squared Error (MSE) | `(1/m) Σ (ŷ - y)²` | Penalizes large errors more |
| Root MSE (RMSE) | `√MSE` | Same unit as target |
| R² (Coefficient of Determination) | `1 - SS_res/SS_tot` | Fraction of variance explained; 1.0 is perfect |

---

## Strengths and Weaknesses

**Strengths:**
- Highly interpretable — each weight has a direct meaning
- Computationally very efficient
- Serves as a baseline for regression tasks
- No hyperparameters needed (without regularization)

**Weaknesses:**
- Cannot model nonlinear relationships (without feature engineering)
- Sensitive to outliers (MSE squares the errors)
- Assumes feature relationships are additive
- Breaks down when assumptions are violated

---

## Role in ML and LLMs

Linear regression is directly embedded in deep learning:
- Every fully-connected (dense) layer in a neural network is a linear transformation: `z = Wx + b`
- The output layer of a regression network is literally a linear regression
- The concept of a **linear probe** — fitting a linear classifier on top of frozen model embeddings — is standard practice for evaluating LLM representations
- Linear attention approximations in efficient transformers are inspired by the linear algebra of regression

Understanding linear regression is a prerequisite for understanding how gradients flow through any modern model.
