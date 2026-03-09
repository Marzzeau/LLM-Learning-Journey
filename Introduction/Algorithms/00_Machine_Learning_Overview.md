# Machine Learning Overview

## Table of Contents
1. [What is Machine Learning?](#what-is-machine-learning)
2. [Core Learning Paradigms](#core-learning-paradigms)
3. [The Learning Pipeline](#the-learning-pipeline)
4. [The Role of Math](#the-role-of-math)
   - [Linear Algebra](#linear-algebra)
   - [Calculus](#calculus)
   - [Probability and Statistics](#probability-and-statistics)
5. [Key Concepts](#key-concepts)
   - [The Model](#the-model)
   - [Loss Functions](#loss-functions)
   - [Optimization](#optimization)
   - [Generalization](#generalization)
6. [The Bias-Variance Tradeoff](#the-bias-variance-tradeoff)
7. [How ML Connects to LLMs](#how-ml-connects-to-llms)

---

## What is Machine Learning?

Machine Learning is a subfield of artificial intelligence in which systems **learn patterns from data** rather than being explicitly programmed with rules.

Instead of a developer writing `if X then Y`, a machine learning model infers the relationship between `X` and `Y` by observing many examples of both. The result is a mathematical function — parameterized by learned weights — that maps inputs to outputs.

Formally, following Tom Mitchell's definition:

> *A computer program is said to learn from experience E with respect to some task T and performance measure P, if its performance on T, as measured by P, improves with experience E.*

---

## Core Learning Paradigms

| Paradigm | What the model learns from | Examples |
|---|---|---|
| **Supervised Learning** | Labeled pairs `(input, target)` | Linear regression, neural networks, SVMs |
| **Unsupervised Learning** | Unlabeled data; finds structure | K-means clustering, autoencoders, PCA |
| **Reinforcement Learning** | Reward signals from an environment | Game-playing agents, robotics, RLHF in LLMs |
| **Self-Supervised Learning** | Labels derived from the data itself | Predicting masked words — the basis of LLM pretraining |

---

## The Learning Pipeline

Every ML system follows roughly the same pipeline:

```
Raw Data
   ↓
Preprocessing & Feature Engineering
   ↓
Model (parameterized function f(x; w))
   ↓
Loss Computation  →  how wrong are we?
   ↓
Optimization  →  adjust w to reduce loss
   ↓
Evaluation  →  does it generalize to unseen data?
```

Each step involves deliberate mathematical choices. The model architecture, the loss function, and the optimizer are all design decisions with mathematical consequences.

---

## The Role of Math

Machine learning is applied mathematics. Three fields underpin virtually every algorithm:

### Linear Algebra

Data is represented as vectors and matrices. Operations on that data — transformations, projections, decompositions — are all linear algebra.

```
Input vector:   x = [x1, x2, ..., xn]
Weight matrix:  W  (shape: output_dim × input_dim)
Transformation: z = Wx + b
```

Key concepts:
- **Dot product**: measures similarity and performs weighted sums
- **Matrix multiplication**: the core operation in every neural network layer
- **Eigenvalues/eigenvectors**: used in PCA and understanding model stability
- **Norms** (`||w||`): measure the size of vectors; used in regularization

### Calculus

Training a model means minimizing a loss function. This requires computing how the loss changes as each weight changes — that is, the **gradient**.

```
Gradient:   ∇J(w) = [∂J/∂w1, ∂J/∂w2, ..., ∂J/∂wn]
Update:     w := w - α * ∇J(w)
```

Key concepts:
- **Partial derivatives**: how much does `J` change if only `w_i` changes?
- **Chain rule**: the mathematical foundation of backpropagation in neural networks
- **Gradient descent**: iteratively moving weights in the direction that reduces loss

### Probability and Statistics

ML models learn from noisy, finite data and must generalize to new examples. Probability gives us the tools to reason about uncertainty.

```
Maximum Likelihood Estimation:  w* = argmax_w P(data | w)
Bayes' Theorem:                 P(w | data) ∝ P(data | w) * P(w)
```

Key concepts:
- **Probability distributions**: model assumptions about noise (e.g., Gaussian → MSE loss, Bernoulli → cross-entropy loss)
- **Expectation and variance**: describe what a model should output on average and how confident it should be
- **Maximum Likelihood Estimation (MLE)**: the principled framework from which most loss functions are derived

---

## Key Concepts

### The Model

A model is a parameterized function `f(x; w)` that maps an input `x` to a prediction `ŷ`:

```
ŷ = f(x; w)
```

The parameters `w` (weights and biases) are the only things that change during training. The architecture — the structure of `f` — is fixed by the designer.

### Loss Functions

The loss function `J(w)` quantifies how wrong the model's predictions are on the training data. The goal of training is to minimize `J`.

| Task | Common Loss | Formula |
|---|---|---|
| Regression | Mean Squared Error | `(1/m) Σ (ŷ - y)²` |
| Binary Classification | Binary Cross-Entropy | `-[y log(ŷ) + (1-y) log(1-ŷ)]` |
| Multi-class Classification | Categorical Cross-Entropy | `-Σ yᵢ log(ŷᵢ)` |
| Language Modeling | Negative Log-Likelihood | `-log P(next token | context)` |

Loss functions are not arbitrary — each one corresponds to a specific probabilistic assumption about how the data was generated.

### Optimization

Once the gradient `∇J(w)` is computed, weights are updated to reduce the loss:

```
w := w - α * ∇J(w)
```

Where `α` (the learning rate) controls how large each step is. Modern optimizers like Adam adapt the learning rate per parameter using estimates of the gradient's mean and variance:

```
m := β1 * m + (1 - β1) * ∇J       ← running mean of gradient
v := β2 * v + (1 - β2) * ∇J²      ← running variance of gradient
w := w - α * m / (√v + ε)
```

### Generalization

A model that memorizes training data without learning underlying patterns will fail on new inputs. This is **overfitting**.

Generalization is measured by performance on a held-out **test set** — data the model has never seen. The gap between training loss and test loss reveals how well the model has generalized.

Common techniques to improve generalization:
- **Regularization**: penalize large weights (L1, L2)
- **Dropout**: randomly zero out neurons during training
- **Early stopping**: halt training when validation loss stops improving
- **Data augmentation**: artificially expand the training set

---

## The Bias-Variance Tradeoff

Every model makes a tradeoff between two sources of error:

```
Total Error = Bias² + Variance + Irreducible Noise
```

| | High Bias | High Variance |
|---|---|---|
| **Also called** | Underfitting | Overfitting |
| **Symptom** | Poor training AND test performance | Good training, poor test performance |
| **Cause** | Model too simple | Model too complex |
| **Fix** | More model capacity, better features | Regularization, more data, simpler model |

Finding the right model complexity is one of the central challenges of applied ML.

---

## How ML Connects to LLMs

Large Language Models are machine learning models — the same principles apply at every scale:

- **Architecture**: a transformer is a deep stack of matrix multiplications and nonlinear activations
- **Loss function**: next-token prediction is cross-entropy loss over a vocabulary
- **Optimization**: Adam optimizer minimizes loss across billions of (token, next-token) pairs
- **Generalization**: pretraining on diverse text teaches generalizable structure; fine-tuning adapts it to a task
- **Regularization**: weight decay, dropout, and careful data curation all serve the same role as in small models

Understanding ML fundamentals is not optional background — it is the direct explanation for why LLMs behave the way they do.
