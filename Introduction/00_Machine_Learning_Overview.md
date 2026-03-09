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

Where:
- `x` — the input vector; each element `xᵢ` is one feature of a single data point
- `xn` — the value of the n-th feature (e.g., pixel intensity, word frequency)
- `W` — the weight matrix; each row defines a linear transformation for one output dimension
- `output_dim` — the number of values the layer produces (e.g., number of neurons)
- `input_dim` — the number of features in `x`
- `z` — the output of the linear transformation, before any activation function is applied
- `b` — the bias vector; one value per output dimension, shifts the output independently of `x`

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

Where:
- `∇J(w)` — the gradient of the loss with respect to all weights; a vector pointing in the direction of steepest increase in loss
- `J` — the loss function (e.g., MSE, cross-entropy); the value we are trying to minimize
- `w` — the full vector of model parameters (weights and biases) being optimized
- `∂J/∂wᵢ` — the partial derivative of `J` with respect to weight `wᵢ`; how much `J` changes if only `wᵢ` changes
- `α` — the learning rate; a small positive scalar (e.g., 0.001) controlling the size of each update step
- `:=` — assignment; the weight vector is replaced with its updated value after each step

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

Where:
- `w*` — the optimal weights; the values of `w` that maximize the likelihood of the observed data
- `argmax_w` — "the value of `w` that maximizes the following expression"
- `P(data | w)` — the likelihood; the probability of observing this dataset given a specific set of weights
- `P(w | data)` — the posterior; the probability that these weights are correct, after seeing the data
- `P(w)` — the prior; our belief about the weights before seeing any data (e.g., "weights should be small")
- `∝` — "proportional to"; the posterior equals the right-hand side up to a normalizing constant

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

Where:
- `ŷ` — the model's predicted output (pronounced "y-hat"); the estimate of the true target `y`
- `f` — the model function; the mathematical structure defined by the architecture (e.g., a neural network)
- `x` — the input to the model (a vector of features for one data point)
- `w` — the learned parameters (weights and biases); everything the model adjusts during training
- `;` — separates the input `x` from the parameters `w`; `x` is fed in at inference time, `w` is fixed after training

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

Where:
- `w` — the current weight vector being updated
- `α` — the learning rate; controls how large each step is (too large → unstable, too small → slow)
- `∇J(w)` — the gradient of the loss at the current weights; points toward steepest increase, so we subtract it

Modern optimizers like Adam adapt the learning rate per parameter using estimates of the gradient's mean and variance:

```
m := β1 * m + (1 - β1) * ∇J       ← running mean of gradient
v := β2 * v + (1 - β2) * ∇J²      ← running variance of gradient
w := w - α * m / (√v + ε)
```

Where:
- `m` — the first moment estimate; a running (exponential) average of past gradients, smoothing out noise
- `v` — the second moment estimate; a running average of past squared gradients, tracking gradient magnitude
- `β1` — decay rate for `m`; typically 0.9 (recent gradients matter more than old ones)
- `β2` — decay rate for `v`; typically 0.999
- `∇J` — the current gradient of the loss
- `∇J²` — element-wise square of the gradient
- `α` — the global learning rate (same role as in vanilla gradient descent)
- `√v` — square root of `v`; scales the step size down for parameters with large, consistent gradients
- `ε` — a tiny constant (e.g., 1e-8) added to prevent division by zero

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

Where:
- `Total Error` — the expected prediction error of the model on unseen data
- `Bias²` — squared bias; error from overly simplistic assumptions in the model (e.g., fitting a line to curved data)
- `Variance` — error from the model being too sensitive to fluctuations in the training data
- `Irreducible Noise` — error inherent in the data itself (measurement noise, missing features); cannot be eliminated by any model

| | High Bias | High Variance |
|---|---|---|
| **Also called** | Underfitting | Overfitting |
| **Symptom** | Poor training AND test performance | Good training, poor test performance |
| **Cause** | Model too simple | Model too complex |
| **Fix** | More model capacity, better features | Regularization, more data, simpler model |

Finding the right model complexity is one of the central challenges of applied ML.

---

## LLM Math in Practice

The following examples trace the actual arithmetic that runs inside a transformer at each stage, using small numbers to keep the calculations readable. Real LLMs use the same operations at larger scale.

---

### Step 1 — Token Embedding (Linear Algebra)

Every word (token) is converted to a vector by looking it up in a learned embedding matrix `E`.

```
Vocabulary size:   V = 5   (tokens: ["cat", "sat", "on", "the", "mat"])
Embedding dim:     d = 3

Embedding matrix E  (shape 5 × 3):
         d0     d1     d2
"cat"  [ 0.9,  0.1, -0.4 ]
"sat"  [ 0.2,  0.8,  0.3 ]
"on"   [-0.1,  0.5,  0.7 ]
"the"  [ 0.0, -0.3,  0.9 ]
"mat"  [ 0.8,  0.2, -0.5 ]

Input token: "cat"  →  one-hot vector:  x = [1, 0, 0, 0, 0]

Embedding lookup = x · E = [0.9, 0.1, -0.4]
```

This is just a row-selection — a dot product of a one-hot vector with `E`. The embedding vector `[0.9, 0.1, -0.4]` is the learned representation of "cat" that the rest of the network will process.

---

### Step 2 — Attention Score (Dot Product + Softmax)

Attention decides how much each token should "look at" every other token. For a single attention head:

```
Each token gets three vectors derived from its embedding:
  Q (query)  — what am I looking for?
  K (key)    — what do I contain?
  V (value)  — what do I contribute?

Sequence: ["cat", "sat"]
Q_cat = [1.0,  0.0]
K_cat = [1.0,  0.0]
K_sat = [0.5,  1.0]
V_cat = [0.8, -0.1]
V_sat = [0.2,  0.9]

Scaling factor: √d_k = √2 ≈ 1.41

Raw attention scores for "cat" attending to each token:
  score(cat→cat) = Q_cat · K_cat / √d_k = (1×1 + 0×0) / 1.41 = 0.71
  score(cat→sat) = Q_cat · K_sat / √d_k = (1×0.5 + 0×1) / 1.41 = 0.35

Softmax converts scores to weights that sum to 1:
  exp(0.71) ≈ 2.03,   exp(0.35) ≈ 1.42
  weights = [2.03, 1.42] / (2.03 + 1.42) = [0.59, 0.41]

Attention output for "cat" = 0.59 × V_cat + 0.41 × V_sat
  = 0.59×[0.8, -0.1] + 0.41×[0.2, 0.9]
  = [0.47, -0.06] + [0.08,  0.37]
  = [0.55, 0.31]
```

The token "cat" now carries a blend of its own information (59%) and "sat"'s information (41%). This is how context flows between tokens.

---

### Step 3 — Feed-Forward Layer (Matrix Multiply + Activation)

After attention, each token passes through a small feed-forward network independently:

```
Input:          h = [0.55, 0.31]       (attention output for "cat")
Weight matrix:  W = [[ 1.0, -0.5],
                     [ 0.3,  0.8],
                     [-0.7,  1.2]]     (shape: 3 × 2)
Bias:           b = [0.1, -0.2, 0.0]

Linear step:    z = W · h + b
  z[0] = 1.0×0.55 + (-0.5)×0.31 + 0.1 =  0.55 - 0.155 + 0.1 =  0.495
  z[1] = 0.3×0.55 +   0.8×0.31 - 0.2 =  0.165 + 0.248 - 0.2 =  0.213
  z[2] = (-0.7)×0.55 + 1.2×0.31 + 0.0 = -0.385 + 0.372       = -0.013

Activation (ReLU — zero out negatives):
  ReLU(z) = [0.495, 0.213, 0.0]
```

The network learns which features to amplify and which to suppress. Stacking many such layers is what gives transformers their representational power.

---

### Step 4 — Next-Token Prediction (Softmax over Vocabulary)

The final hidden state is projected to logits over the full vocabulary, then normalized:

```
Vocabulary: ["cat", "sat", "on", "the", "mat"]   (V = 5)
Final hidden state for last token: h = [0.5, 0.3, 0.0]

Output projection W_out  (shape: 5 × 3):
  "cat" row: [ 0.2,  0.1, -0.3]  →  logit = 0.5×0.2 + 0.3×0.1 + 0.0×(-0.3) = 0.13
  "sat" row: [ 0.4,  0.6,  0.1]  →  logit = 0.5×0.4 + 0.3×0.6 + 0.0×0.1    = 0.38
  "on"  row: [-0.1,  0.9,  0.5]  →  logit = 0.5×(-0.1)+ 0.3×0.9            = 0.22
  "the" row: [ 0.0, -0.2,  0.8]  →  logit = 0.5×0.0  + 0.3×(-0.2)          = -0.06
  "mat" row: [ 0.7,  0.3, -0.1]  →  logit = 0.5×0.7  + 0.3×0.3             = 0.44

Logits: [0.13, 0.38, 0.22, -0.06, 0.44]

Softmax probabilities:
  exp values: [1.14, 1.46, 1.25, 0.94, 1.55]   (sum ≈ 6.34)
  P("cat")  = 1.14 / 6.34 ≈ 0.18
  P("sat")  = 1.46 / 6.34 ≈ 0.23
  P("on")   = 1.25 / 6.34 ≈ 0.20
  P("the")  = 0.94 / 6.34 ≈ 0.15
  P("mat")  = 1.55 / 6.34 ≈ 0.24   ← highest probability
```

The model predicts "mat" as the most likely next token.

---

### Step 5 — Loss Computation (Cross-Entropy)

If the correct next token is "mat" (index 4), the loss for this one prediction is:

```
Cross-entropy loss = -log P(correct token)
                   = -log(0.24)
                   ≈ 1.43

Perfect prediction would give:  -log(1.0) = 0.0
Random guess (uniform) would give:  -log(0.2) ≈ 1.61
```

Training minimizes this loss by backpropagating gradients through every weight in Steps 1–4. After billions of such updates, the model's weights encode statistical patterns of language.

---

### Step 6 — Weight Update (Gradient Descent)

After backpropagation computes `∂J/∂w` for every weight, Adam updates each one:

```
Suppose one weight w = 0.40 has gradient ∂J/∂w = 0.12.

Adam state (after many steps, bias-corrected):
  m̂ = 0.11    (smoothed gradient estimate)
  v̂ = 0.015   (smoothed squared gradient estimate)

Learning rate:  α = 0.001,   ε = 1e-8

Update:
  w := w - α × m̂ / (√v̂ + ε)
     = 0.40 - 0.001 × 0.11 / (√0.015 + 1e-8)
     = 0.40 - 0.001 × 0.11 / 0.122
     = 0.40 - 0.0009
     = 0.3991
```

The weight moved by ~0.0009 — a tiny step in the direction that reduces loss. Repeat across every weight, across every token, across billions of training examples.

---

## How ML Connects to LLMs

Large Language Models are machine learning models — the same principles apply at every scale:

- **Architecture**: a transformer is a deep stack of matrix multiplications and nonlinear activations
- **Loss function**: next-token prediction is cross-entropy loss over a vocabulary
- **Optimization**: Adam optimizer minimizes loss across billions of (token, next-token) pairs
- **Generalization**: pretraining on diverse text teaches generalizable structure; fine-tuning adapts it to a task
- **Regularization**: weight decay, dropout, and careful data curation all serve the same role as in small models

Understanding ML fundamentals is not optional background — it is the direct explanation for why LLMs behave the way they do.
