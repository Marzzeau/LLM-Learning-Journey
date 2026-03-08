# Neural Network Theory
### Everything you need to complete the exercises

---

## Table of Contents

1. [The Neuron](#1-the-neuron)
2. [Layers and the Network](#2-layers-and-the-network)
3. [Activation Functions](#3-activation-functions) ← Exercise 1
4. [Loss Functions](#4-loss-functions) ← Exercise 2
5. [The Forward Pass](#5-the-forward-pass) ← Exercise 3 (part A)
6. [Backpropagation](#6-backpropagation) ← Exercise 3 (part B)
7. [Gradient Descent](#7-gradient-descent) ← Exercise 3 (part C)
8. [Weight Initialisation](#8-weight-initialisation) ← Exercise 3 (part D)
9. [The Training Loop](#9-the-training-loop) ← Exercise 4
10. [The XOR Problem](#10-the-xor-problem) ← Exercise 4

---

## 1. The Neuron

A biological neuron receives signals, adds them up, and fires if the total is
strong enough. An artificial neuron does the same thing with numbers.

```
Inputs          Weights         Neuron
──────          ───────         ──────
  x1 ──── w1 ──┐
  x2 ──── w2 ──┤──► ( Σ x·w + b ) ──► activation ──► output
  x3 ──── w3 ──┘
```

Mathematically, a single neuron computes:

```
z      =  x1·w1  +  x2·w2  +  x3·w3  +  b        (weighted sum + bias)
output =  activation(z)                             (non-linearity)
```

| Symbol | Name   | Role                                                     |
|--------|--------|----------------------------------------------------------|
| `x`    | Input  | The data fed into the neuron                             |
| `w`    | Weight | How much attention to pay to each input                  |
| `b`    | Bias   | A constant offset; lets the neuron fire without any input|
| `z`    | Pre-activation | Raw weighted sum before the activation function |
| `output` | Activation | The neuron's final signal                       |

**Why a bias?**
Without a bias, every decision boundary must pass through the origin (0, 0).
Adding `b` lets the boundary shift freely.

---

## 2. Layers and the Network

Real networks stack many neurons side-by-side in **layers**, and many layers
in sequence.

```
Input Layer   Hidden Layer 1   Hidden Layer 2   Output Layer
───────────   ──────────────   ──────────────   ────────────
   x1  ──────►  (neuron)                            (neuron) ──► ŷ
   x2  ──────►  (neuron) ──────►  (neuron) ──────►
   x3  ──────►  (neuron)          (neuron)
```

- **Input layer** — the raw features (no computation, just pass-through).
- **Hidden layers** — learn intermediate representations.
- **Output layer** — produces the final prediction `ŷ`.

### Matrix notation

Instead of writing one neuron at a time, we write all neurons in a layer
together as a matrix operation. If a layer has `n_in` inputs and `n_out`
neurons, and we process a batch of `B` samples:

```
X  shape: (B, n_in)     — batch of B input vectors
W  shape: (n_in, n_out) — one column per neuron
b  shape: (1, n_out)    — one bias per neuron (broadcast over B)

Z = X @ W + b           — shape: (B, n_out)
A = activation(Z)       — shape: (B, n_out)   element-wise
```

The `@` symbol denotes matrix multiplication. This single line replaces
computing every neuron individually — much faster on modern hardware.

---

## 3. Activation Functions

### Why do we need them?

Without activation functions, every layer is a linear transform, and any stack
of linear transforms collapses into a single linear transform. No matter how
many layers you add, the network can only learn straight-line (linear)
relationships.

```
Layer 1:  Z1 = X  @ W1 + b1
Layer 2:  Z2 = Z1 @ W2 + b2  =  X @ (W1·W2) + (b1·W2 + b2)
                                 └─────────────────────────┘
                                 This is just one linear layer!
```

Activation functions break linearity, letting the network learn curves,
boundaries, and complex patterns.

---

### Sigmoid  σ(x)

```
           1
σ(x) = ─────────
        1 + e^(−x)
```

| Property | Value |
|----------|-------|
| Output range | (0, 1) |
| When to use | Binary classification **output** layer |
| Derivative | `σ'(x) = σ(x) · (1 − σ(x))` |

**Shape:**

```
  1 │                          ┌──────────
    │                      ┌───┘
0.5 │ ─────────────────────┤
    │              ┌───────┘
  0 │──────────────┘
    └────────────────────────────────────►
                           0              x
```

The sigmoid squashes any real number into (0, 1), making it ideal for
outputting probabilities.

**Derivative** — why `σ(x) · (1 − σ(x))`?

Let `s = σ(x)`. By the quotient rule on `1/(1 + e^(-x))`:

```
σ'(x) = e^(−x) / (1 + e^(−x))²
      = [1/(1 + e^(−x))] · [e^(−x)/(1 + e^(−x))]
      = s · (1 − s)
```

The derivative peaks at 0.25 when `x = 0` and approaches 0 far from the
origin (the **vanishing gradient** problem — gradients shrink as they
pass through many sigmoid layers).

---

### ReLU  (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
```

| Property | Value |
|----------|-------|
| Output range | [0, ∞) |
| When to use | Hidden layers (default choice) |
| Derivative | `1 if x > 0 else 0` |

**Shape:**

```
  │          /
  │         /
  │        /
  │       /
  │      /
  │     /
  │────/───────────────►
  0                     x
```

**Why ReLU is popular:**
- No vanishing gradient for positive inputs — the gradient is exactly 1.
- Fast to compute (just a comparison).
- Sparsity: many neurons output 0, which can be efficient.

**Dead neuron problem:** If a neuron's input is always negative, its output
is always 0 and its gradient is always 0. It can never recover. Good weight
initialisation (see §8) reduces this risk.

**Derivative:**

```
ReLU'(x) = 1   if x > 0
            0   if x ≤ 0
```

In NumPy: `(x > 0).astype(float)` — cast the boolean array to 1.0 / 0.0.

---

### Softmax

```
softmax(x)_i = e^(x_i) / Σ_j e^(x_j)
```

| Property | Value |
|----------|-------|
| Output range | (0, 1) for each element; all elements sum to 1 |
| When to use | Multi-class classification output layer |

Softmax turns a vector of raw scores (called **logits**) into a probability
distribution. The largest score gets the highest probability.

**Numerical stability trick:** Subtract the maximum before exponentiating.

```
softmax(x)_i = e^(x_i − max(x)) / Σ_j e^(x_j − max(x))
```

Mathematically identical (the constant cancels), but prevents `e^(large number)` overflow.

---

## 4. Loss Functions

A **loss function** (also called *cost function* or *objective function*)
measures how wrong the network's predictions are. Training is the process
of minimising this number.

Two parts are always needed:
1. The **loss value** — a scalar measuring total error.
2. The **derivative** — which direction to push the predictions to reduce the loss.

---

### Mean Squared Error (MSE)

Used for **regression** — predicting continuous values.

```
MSE(ŷ, y) = (1/n) · Σᵢ (ŷᵢ − yᵢ)²
```

- Squaring makes all errors positive and penalises large errors more.
- The `1/n` average lets the loss be independent of batch size.

**Derivative with respect to ŷ:**

```
∂MSE/∂ŷᵢ = (2/n) · (ŷᵢ − yᵢ)
```

Interpretation: if `ŷᵢ > yᵢ`, the gradient is positive → push `ŷᵢ` down.
If `ŷᵢ < yᵢ`, the gradient is negative → push `ŷᵢ` up.

---

### Binary Cross-Entropy (BCE)

Used for **binary classification** — predicting probability of class 1.

```
BCE(ŷ, y) = −(1/n) · Σᵢ [ yᵢ·log(ŷᵢ) + (1 − yᵢ)·log(1 − ŷᵢ) ]
```

When `y = 1`: only `log(ŷ)` matters — penalise predicting low probability.
When `y = 0`: only `log(1 − ŷ)` matters — penalise predicting high probability.

`log` is the natural logarithm. Key values:
```
log(1.0) =  0.0        ← perfect prediction, zero loss
log(0.5) = −0.693      ← uncertain prediction
log(0.0) = −∞          ← catastrophically wrong (we clip to avoid this)
```

**Why cross-entropy instead of MSE for classification?**
MSE applied to sigmoid outputs has nearly-zero gradients when the prediction is
confidently wrong (sigmoid saturates). Cross-entropy does not have this problem
— it penalises confident mistakes very harshly.

**Derivative with respect to ŷ:**

```
∂BCE/∂ŷᵢ = (ŷᵢ − yᵢ) / (n · ŷᵢ · (1 − ŷᵢ))
```

Note: when BCE is paired with a sigmoid output layer, these two derivatives
cancel beautifully, giving simply `(ŷ − y)/n` as the combined gradient. This
is why the BCE + sigmoid combination is the standard for binary classification.

---

## 5. The Forward Pass

The forward pass is the computation that produces a prediction from an input.
Data flows **left to right** through every layer in sequence.

### One layer

```
Input X          Linear step          Activation
(B, n_in) ──►   Z = X @ W + b   ──►  A = f(Z)   ──► (B, n_out)
```

### Full network (L layers)

```
A₀ = X                                    (raw input)
Z₁ = A₀ @ W₁ + b₁  →  A₁ = f₁(Z₁)
Z₂ = A₁ @ W₂ + b₂  →  A₂ = f₂(Z₂)
...
Zₗ = Aₗ₋₁ @ Wₗ + bₗ →  Aₗ = fₗ(Zₗ)      (final output ŷ)
```

Each layer transforms the data into a new representation, hopefully one that
makes the final prediction easier.

### What to cache

During the forward pass, save `X` (the input) and `Z` (pre-activation) for
each layer. **You will need them during backpropagation.**

```python
self.input  = x          # save input
self.z      = x @ W + b  # save pre-activation
self.output = f(self.z)  # compute and return activation
```

---

## 6. Backpropagation

Backpropagation answers: **by how much should each weight change to reduce
the loss?**

It is an efficient application of the **chain rule** from calculus, working
backwards from the loss through every layer.

### The Chain Rule (one variable)

If `y = f(g(x))`, then:

```
dy/dx = (dy/dg) · (dg/dx)
```

Read: "the rate of change of `y` with respect to `x` equals the rate of
change of `y` with respect to `g`, times the rate of change of `g` with
respect to `x`."

### The Chain Rule through a layer

A layer computes `A = f(Z)` where `Z = X @ W + b`.
The loss `L` depends on `A`, which depends on `Z`, which depends on `W`, `b`, `X`.

Given `∂L/∂A` (gradient arriving from the next layer), we compute:

```
∂L/∂Z = ∂L/∂A  ·  f'(Z)          element-wise multiply by activation derivative

∂L/∂W = Xᵀ @ ∂L/∂Z               how much did each weight contribute?

∂L/∂b = Σ ∂L/∂Z   (sum over batch) how much did each bias contribute?

∂L/∂X = ∂L/∂Z @ Wᵀ               gradient to pass to the PREVIOUS layer
```

### Shape guide

```
X     : (B, n_in)          input batch
W     : (n_in, n_out)      weights
b     : (1, n_out)         biases
Z     : (B, n_out)         pre-activation
A     : (B, n_out)         post-activation

∂L/∂A : (B, n_out)        arriving gradient (same shape as A)
∂L/∂Z : (B, n_out)        element-wise product (same shape as Z)
∂L/∂W : (n_in, n_out)     Xᵀ @ ∂L/∂Z  — same shape as W
∂L/∂b : (1, n_out)        sum over B   — same shape as b
∂L/∂X : (B, n_in)         ∂L/∂Z @ Wᵀ — same shape as X
```

All shapes must be consistent — a mismatch means something is wrong.

### Why `Xᵀ @ ∂L/∂Z` for the weight gradient?

Imagine a single sample and a single neuron: `z = x·w`.
Then `∂z/∂w = x`, so `∂L/∂w = (∂L/∂z) · x`.

For a full batch and many neurons, this generalises to the matrix product
`Xᵀ @ ∂L/∂Z`, which sums the outer products over every sample in the batch.

### Why `∂L/∂Z @ Wᵀ` for the input gradient?

We need this to pass the gradient back to the **previous layer**.
`z = x @ W`, so by the matrix chain rule, `∂z/∂x = Wᵀ`, giving
`∂L/∂x = ∂L/∂z @ Wᵀ`.

### Worked example (2-layer network)

```
Forward:
  Z1 = X  @ W1 + b1   →   A1 = relu(Z1)
  Z2 = A1 @ W2 + b2   →   A2 = sigmoid(Z2)   ← prediction ŷ
  L  = BCE(A2, y)

Backward (right to left):
  ∂L/∂A2 = BCE_derivative(A2, y)

  Layer 2:
    ∂L/∂Z2 = ∂L/∂A2 · sigmoid'(Z2)
    ∂L/∂W2 = A1ᵀ @ ∂L/∂Z2
    ∂L/∂b2 = Σ ∂L/∂Z2
    ∂L/∂A1 = ∂L/∂Z2 @ W2ᵀ        ← pass to layer 1

  Layer 1:
    ∂L/∂Z1 = ∂L/∂A1 · relu'(Z1)
    ∂L/∂W1 = Xᵀ @ ∂L/∂Z1
    ∂L/∂b1 = Σ ∂L/∂Z1
    (∂L/∂X — not needed, X is the data)
```

Each layer only needs to know the gradient arriving at its output. It
computes the gradient at its input and passes it back. This is why each
`DenseLayer.backward()` takes `d_output` and returns `d_input`.

---

## 7. Gradient Descent

Once we know `∂L/∂W` and `∂L/∂b` for every layer, we update the weights
to reduce the loss.

### The update rule

```
W ← W − α · ∂L/∂W
b ← b − α · ∂L/∂b
```

`α` (alpha) is the **learning rate** — a small positive number (e.g. 0.01 to 0.5).

### Intuition

Imagine standing on a hilly landscape where height = loss. You want to reach
the lowest valley. The gradient points uphill. Moving *opposite* to the gradient
means moving downhill — decreasing the loss.

```
Loss
  │    *
  │  *   *                   ← start here, gradient is negative (slope down-right)
  │        *
  │          *   *   *       ← gradient is positive (slope up-right)
  │              minimum
  └─────────────────────────► Weight value
```

### The learning rate

| Too small | Converges very slowly; may take millions of steps |
|-----------|--------------------------------------------------|
| Too large | Overshoots the minimum; loss bounces or diverges |
| Just right | Steady decrease towards minimum                 |

A typical starting point: `0.01` for large networks, `0.1–0.5` for small ones.

### Batch gradient descent (what we implement)

We compute the gradient on the **entire dataset** at once and take one step.
Simple and stable for small datasets like XOR.

---

## 8. Weight Initialisation

### Why not initialise to zero?

If all weights are zero, every neuron computes the same value, every gradient
is the same, and every update is the same. All neurons remain identical forever
— the network never learns anything useful. This is called the
**symmetry problem**.

### Why not use large random values?

Large initial weights push activations into the **saturated regions** of sigmoid
(where the function is nearly flat). Flat region → tiny gradients → very slow
learning (vanishing gradient).

### Xavier Initialisation (for sigmoid / tanh)

```
scale = sqrt(2 / (fan_in + fan_out))
W = random_normal(shape) * scale
```

`fan_in` = number of inputs to the layer, `fan_out` = number of neurons.

Keeps the variance of activations roughly constant across layers for
sigmoid/tanh, which are centred around 0.

### He Initialisation (for ReLU)

```
scale = sqrt(2 / fan_in)
W = random_normal(shape) * scale
```

ReLU kills half its inputs (the negative ones), so we need larger weights
to compensate. He initialisation accounts for this by using `2/fan_in`
instead of `2/(fan_in + fan_out)`.

**This is what we use in the exercises for hidden ReLU layers.**

Using the wrong initialisation can cause the network to never learn (dead
neurons with ReLU, or near-zero gradients with sigmoid).

---

## 9. The Training Loop

One **epoch** = one pass over the entire dataset.

```
repeat for N epochs:
  1. Forward pass  → compute prediction ŷ
  2. Loss          → measure how wrong ŷ is
  3. Backward pass → compute gradients for every W and b
  4. Update        → W ← W − lr · dW,   b ← b − lr · db
```

In code:

```python
for epoch in range(epochs):
    y_pred = model.forward(X)                   # step 1
    loss   = loss_fn(y_pred, y)                 # step 2
    d_loss = loss_fn_derivative(y_pred, y)      # step 3a (starting gradient)
    model.backward(d_loss)                      # step 3b (backprop)
    model.update_weights(learning_rate)         # step 4
```

### What you expect to see

```
Epoch     0  |  loss: 0.6931      ← network is guessing ~50/50
Epoch   500  |  loss: 0.3245
Epoch  1000  |  loss: 0.1023
Epoch  2000  |  loss: 0.0134
Epoch  5000  |  loss: 0.0008      ← near-perfect predictions
```

The loss should **steadily decrease**. If it:
- Stays flat → learning rate too small, bad init, or bug in backprop.
- Explodes to `nan` → learning rate too large.
- Decreases then flatlines → possibly stuck in a local minimum; try a
  different learning rate or more neurons.

---

## 10. The XOR Problem

XOR stands for "exclusive or":

```
x1  x2  | y
---------+---
 0   0  | 0      (same → 0)
 0   1  | 1      (different → 1)
 1   0  | 1      (different → 1)
 1   1  | 0      (same → 0)
```

### Why a single neuron cannot solve XOR

A single neuron (one layer, no hidden layer) can only draw a **straight line**
to separate classes. XOR is not **linearly separable** — there is no straight
line that puts (0,0) and (1,1) on one side and (0,1) and (1,0) on the other.

```
x2
1 │  O       X
  │
0 │  X       O
  └───────────── x1
     0       1

O = class 0,  X = class 1
No straight line separates them.
```

### Why a hidden layer solves it

A hidden layer with even 2 neurons can learn two boundaries, and the output
neuron combines them:

```
Neuron 1 learns: x1 OR x2       (fires when either input is 1)
Neuron 2 learns: x1 AND x2      (fires when both inputs are 1)
Output  learns: Neuron1 AND NOT Neuron2
```

More neurons → more complex boundaries the network can represent.

### Universal Approximation Theorem

A neural network with **one hidden layer of sufficient width** can approximate
any continuous function on a compact domain to arbitrary precision. This is why
adding a hidden layer unlocks what a single-layer network cannot learn.

---

## Summary: What Each Exercise Tests

| Exercise | Concept |
|----------|---------|
| 1 — Activations | Non-linearity; sigmoid squashes to (0,1); ReLU is fast and simple; softmax produces a probability distribution |
| 2 — Loss functions | Measuring error; MSE for regression; BCE for classification; derivatives tell gradient descent which way to move |
| 3 — Neural network | Forward pass (layer by layer); backpropagation (chain rule, reverse order); gradient descent weight update |
| 4 — Training XOR | The full loop; non-linear problems require hidden layers; convergence behaviour |

---

## Quick Reference: Formulas

```
Sigmoid          σ(x)     = 1 / (1 + e^(−x))
Sigmoid deriv    σ'(x)    = σ(x) · (1 − σ(x))

ReLU             f(x)     = max(0, x)
ReLU deriv       f'(x)    = 1 if x > 0, else 0

Softmax          s(x)_i   = e^(x_i − max) / Σ e^(x_j − max)

MSE              L        = (1/n) · Σ (ŷ − y)²
MSE deriv        ∂L/∂ŷ   = (2/n) · (ŷ − y)

BCE              L        = −(1/n) · Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]
BCE deriv        ∂L/∂ŷ   = (ŷ − y) / (n · ŷ · (1 − ŷ))

Forward pass     Z        = X @ W + b
                 A        = activation(Z)

Backprop         ∂L/∂Z   = ∂L/∂A · activation'(Z)    ← element-wise
                 ∂L/∂W   = Xᵀ @ ∂L/∂Z
                 ∂L/∂b   = Σ ∂L/∂Z                    ← sum over batch
                 ∂L/∂X   = ∂L/∂Z @ Wᵀ

Weight update    W        ← W − lr · ∂L/∂W
                 b        ← b − lr · ∂L/∂b

He init (ReLU)   scale    = sqrt(2 / fan_in)
Xavier (sigmoid) scale    = sqrt(2 / (fan_in + fan_out))
```
