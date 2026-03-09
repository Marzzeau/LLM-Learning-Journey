# Neural Networks and Backpropagation

## Table of Contents
1. [Overview](#overview)
2. [Biological Inspiration](#biological-inspiration)
3. [The Artificial Neuron](#the-artificial-neuron)
4. [Activation Functions](#activation-functions)
5. [Network Architecture — MLP](#network-architecture--mlp)
6. [Forward Pass](#forward-pass)
7. [Loss Functions](#loss-functions)
8. [Backpropagation](#backpropagation)
   - [Chain Rule](#chain-rule)
   - [The Backprop Algorithm](#the-backprop-algorithm)
   - [Computational Graphs](#computational-graphs)
9. [Gradient Descent and Optimizers](#gradient-descent-and-optimizers)
10. [The Vanishing and Exploding Gradient Problem](#the-vanishing-and-exploding-gradient-problem)
11. [Regularization Techniques](#regularization-techniques)
12. [Universal Approximation Theorem](#universal-approximation-theorem)
13. [Role in ML and LLMs](#role-in-ml-and-llms)

---

## Overview

Artificial Neural Networks (ANNs) are computational models loosely inspired by the structure of biological brains. A **feedforward neural network** (or Multi-Layer Perceptron, MLP) is a sequence of parameterized linear transformations interleaved with nonlinear **activation functions**.

Neural networks are the foundation of all modern deep learning, including CNNs, RNNs, and the Transformer architecture that powers LLMs. Understanding their mechanics — especially **backpropagation** — is essential for understanding how any deep model learns.

---

## Biological Inspiration

The neuron is the basic computational unit of the brain:
- Receives input signals through **dendrites**
- Integrates these signals in the **cell body (soma)**
- Fires an output signal through the **axon** if the combined signal exceeds a threshold
- Connects to other neurons through **synapses** (with variable strength = weights)

The artificial neuron captures this with:
- **Inputs**: feature values or outputs from previous layer
- **Weights**: learnable synaptic strengths
- **Bias**: learnable threshold
- **Activation**: nonlinear firing function

---

## The Artificial Neuron

A single neuron computes:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b
a = f(z)
```

Where:
- `z` is the **pre-activation** (weighted sum)
- `b` is the **bias**
- `f(·)` is the **activation function**
- `a` is the **activation** (output)

Without the activation function, a neural network is just a linear transformation regardless of depth — multiple linear layers compose to a single linear layer.

---

## Activation Functions

Activation functions introduce the **nonlinearity** that allows neural networks to approximate complex functions.

| Function | Formula | Range | Properties |
|---|---|---|---|
| Sigmoid | `1/(1+e^{-z})` | (0, 1) | Saturates; vanishing gradient |
| Tanh | `(e^z - e^{-z})/(e^z + e^{-z})` | (-1, 1) | Zero-centered; still saturates |
| ReLU | `max(0, z)` | [0, ∞) | Most popular; no saturation for z>0; "dying ReLU" |
| Leaky ReLU | `max(0.01z, z)` | (-∞, ∞) | Fixes dying ReLU |
| GELU | `z * Φ(z)` | (-∞, ∞) | Smooth; used in BERT, GPT |
| Swish | `z * σ(z)` | (-∞, ∞) | Self-gated; used in some large models |
| Softmax | `e^{zₖ}/Σe^{zⱼ}` | (0,1), sums to 1 | Output layer for classification |

**ReLU** (Rectified Linear Unit) revolutionized deep learning due to:
- No saturation for positive values (gradient = 1)
- Computationally trivial
- Empirically much faster training than sigmoid/tanh

**GELU** is preferred in modern transformers because it is smooth (differentiable everywhere) and approximates a stochastic regularizer.

---

## Network Architecture — MLP

A **Multi-Layer Perceptron** has:
- **Input layer**: passes raw features (no computation)
- **Hidden layers**: one or more layers with learned weights + activations
- **Output layer**: produces final predictions (with task-appropriate activation)

```
Input → [Linear → Activation] × L hidden layers → Linear → Output
```

For a network with layers L=1 to L:
```
a⁰ = x                    (input)
z^l = W^l * a^{l-1} + b^l (pre-activation)
a^l = f(z^l)               (activation)
ŷ = a^L                    (output)
```

**Width** = number of neurons per layer
**Depth** = number of layers

Deep networks learn **hierarchical representations**: early layers detect simple patterns (edges), later layers combine these into complex abstractions (faces, words).

---

## Forward Pass

The forward pass propagates input `x` through the network to produce a prediction `ŷ`:

```python
# Pseudocode for a 2-hidden-layer MLP
a0 = x                          # Input
z1 = W1 @ a0 + b1               # Layer 1 pre-activation
a1 = relu(z1)                   # Layer 1 activation

z2 = W2 @ a1 + b2               # Layer 2 pre-activation
a2 = relu(z2)                   # Layer 2 activation

z3 = W3 @ a2 + b3               # Output pre-activation
ŷ  = softmax(z3)                # Output (for classification)
```

All intermediate values `{z^l, a^l}` are cached — they are needed for backpropagation.

---

## Loss Functions

The loss function measures how wrong the model's prediction is:

| Task | Loss Function | Formula |
|---|---|---|
| Binary Classification | Binary Cross-Entropy | `-[y log(ŷ) + (1-y) log(1-ŷ)]` |
| Multi-Class Classification | Categorical Cross-Entropy | `-Σₖ yₖ log(ŷₖ)` |
| Regression | Mean Squared Error | `(ŷ - y)²` |
| Regression (robust) | Mean Absolute Error | `|ŷ - y|` |

The goal of training is to minimize the average loss over all training examples.

---

## Backpropagation

Backpropagation is the algorithm for computing **gradients of the loss with respect to all parameters** in the network. It is an efficient application of the **chain rule** of calculus.

### Chain Rule

For composed functions `f(g(x))`:
```
d/dx f(g(x)) = f'(g(x)) * g'(x)
```

For a chain of operations `L → a^L → a^{L-1} → ... → W^l`:
```
∂L/∂W^l = ∂L/∂a^L * ∂a^L/∂a^{L-1} * ... * ∂a^{l+1}/∂W^l
```

Each term is a local gradient that is easy to compute.

### The Backprop Algorithm

Starting from the output layer and working backward:

```
1. Compute loss gradient: δ^L = ∂L/∂z^L

2. For each layer l = L, L-1, ..., 1:

   Gradients w.r.t. parameters:
       ∂L/∂W^l = δ^l * (a^{l-1})ᵀ
       ∂L/∂b^l = δ^l

   Backpropagate gradient:
       δ^{l-1} = (W^l)ᵀ * δ^l ⊙ f'(z^{l-1})

   Where ⊙ is element-wise multiplication
```

The "error signal" `δ^l` measures how much the loss would change with a small perturbation to `z^l`.

### Computational Graphs

Modern frameworks (PyTorch, TensorFlow) represent computations as **directed acyclic graphs (DAGs)**:
- Each node is an operation
- Each edge carries a tensor
- **Forward pass**: evaluate the graph from inputs to outputs
- **Backward pass (autograd)**: traverse the graph in reverse, accumulating gradients using the chain rule

This is called **automatic differentiation (autograd)** and allows gradients to be computed for any differentiable program.

---

## Gradient Descent and Optimizers

| Optimizer | Update Rule | Key Feature |
|---|---|---|
| SGD | `w -= α * ∂L/∂w` | Simple; requires careful LR tuning |
| SGD + Momentum | `v = βv + ∂L/∂w; w -= αv` | Accumulates velocity; escapes local minima |
| AdaGrad | Adapts LR per-parameter | Good for sparse features; LR decays to 0 |
| RMSProp | Divides by running average of squared gradients | Fixes AdaGrad LR decay |
| Adam | Combines momentum + RMSProp | **Most widely used** in deep learning |

**Adam** update:
```
m_t = β₁ * m_{t-1} + (1-β₁) * g_t       (1st moment estimate)
v_t = β₂ * v_{t-1} + (1-β₂) * g_t²      (2nd moment estimate)
m̂_t = m_t / (1 - β₁ᵗ)                   (bias-corrected)
v̂_t = v_t / (1 - β₂ᵗ)                   (bias-corrected)
w_t = w_{t-1} - α * m̂_t / (√v̂_t + ε)
```

Default parameters: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`.

---

## The Vanishing and Exploding Gradient Problem

During backpropagation, gradients are multiplied together as they propagate through layers. In deep networks:

**Vanishing Gradients:**
- If `|∂a/∂z| < 1` at each layer (e.g., sigmoid), gradients shrink exponentially
- Early layers receive near-zero gradients → fail to learn
- Sigmoid/tanh activations are prone to this

**Exploding Gradients:**
- If `|∂a/∂z| > 1` at each layer, gradients grow exponentially
- Causes numerical instability and divergence

**Solutions:**
| Problem | Solution |
|---|---|
| Vanishing | ReLU activation, Residual connections, Layer normalization |
| Exploding | Gradient clipping, Weight initialization, Batch normalization |
| Both | Careful weight initialization (He, Xavier/Glorot) |

**He initialization** for ReLU:
```
W ~ N(0, √(2/fan_in))
```

**Xavier initialization** for sigmoid/tanh:
```
W ~ N(0, √(2/(fan_in + fan_out)))
```

---

## Regularization Techniques

| Technique | Description |
|---|---|
| L2 Weight Decay | Add `λ||W||²` to loss; penalizes large weights |
| Dropout | Randomly zero out `p` fraction of activations during training; ensemble effect |
| Batch Normalization | Normalize activations within a mini-batch; stabilizes training, acts as regularizer |
| Layer Normalization | Normalizes across features (not batch); preferred in Transformers |
| Early Stopping | Stop training when validation loss stops improving |
| Data Augmentation | Expand training data via transformations |

---

## Universal Approximation Theorem

**Theorem:** A feedforward network with a single hidden layer containing a sufficient number of neurons and a non-polynomial activation function can approximate any continuous function on a compact domain to arbitrary precision.

This establishes that neural networks are **universal function approximators**. However:
- It does not say how many neurons are needed (may be exponential)
- It does not address generalization
- Depth provides exponential advantages over width for certain function classes

---

## Role in ML and LLMs

Backpropagation through MLPs is the engine of all modern deep learning:

- Every component of a Transformer (attention, MLP layers, layer norm) is trained via backprop
- **GPT**, **BERT**, and all large language models are deep neural networks trained with Adam and cross-entropy loss
- The **residual connections** in Transformers directly address the vanishing gradient problem
- **Layer normalization** in LLMs is a direct application of normalization theory
- The **two-stage pre-training → fine-tuning** paradigm relies on gradient flow from task-specific loss back through frozen or adapted parameters
