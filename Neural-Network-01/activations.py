"""
EXERCISE 1 — Activation Functions
==================================
Activation functions introduce non-linearity into the network,
allowing it to learn complex patterns that a plain matrix multiply cannot.

Work through each blank in order. Run the file to check your answers:

    python activations.py

Blanks are marked with:
    None  # [BLANK] — replace None with the correct expression
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SIGMOID
# Maps any real number → (0, 1). Used in binary-classification output layers.
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x):
    """
    Formula: σ(x) = 1 / (1 + e^(−x))

    Tip: use np.exp() for e^x
    """
    # ┌─ BLANK 1 ──────────────────────────────────────────────────────────────
    # │  Return: 1 / (1 + np.exp(-x))
    return None  # [BLANK]
    # └────────────────────────────────────────────────────────────────────────


def sigmoid_derivative(x):
    """
    The derivative of sigmoid, needed during backpropagation.

    Formula: σ'(x) = σ(x) · (1 − σ(x))

    Tip: call sigmoid(x) and multiply.
    """
    # ┌─ BLANK 2 ──────────────────────────────────────────────────────────────
    # │  Step a: s = sigmoid(x)
    # │  Step b: return s * (1 - s)
    s = None       # [BLANK] — compute sigmoid(x)
    return None    # [BLANK] — return s * (1 - s)
    # └────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# RELU  (Rectified Linear Unit)
# The most widely used hidden-layer activation. Fast and avoids vanishing grads.
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    """
    Formula: ReLU(x) = max(0, x)

    Tip: np.maximum(a, b) returns the element-wise maximum of two arrays.
    """
    # ┌─ BLANK 3 ──────────────────────────────────────────────────────────────
    # │  Return: np.maximum(0, x)
    return None  # [BLANK]
    # └────────────────────────────────────────────────────────────────────────


def relu_derivative(x):
    """
    Formula: ReLU'(x) = 1 if x > 0, else 0

    Tip: a boolean array cast to float gives 1.0/0.0.
         (x > 0).astype(float)
    """
    # ┌─ BLANK 4 ──────────────────────────────────────────────────────────────
    # │  Return: (x > 0).astype(float)
    return None  # [BLANK]
    # └────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SOFTMAX
# Converts a vector of raw scores into probabilities that sum to 1.
# Used in multi-class classification output layers.
# ─────────────────────────────────────────────────────────────────────────────

def softmax(x):
    """
    Formula: softmax(x)_i = e^(x_i) / Σ_j e^(x_j)

    Numerical stability trick: subtract max(x) before exponentiating.
    This doesn't change the result mathematically:
        e^(x_i - C) / Σ e^(x_j - C)  =  e^(x_i) / Σ e^(x_j)
    but prevents overflow when x_i is very large.
    """
    # ┌─ BLANK 5 ──────────────────────────────────────────────────────────────
    # │  Step a: shifted = x - np.max(x, axis=-1, keepdims=True)
    # │  Step b: exps    = np.exp(shifted)
    # │  Step c: return    exps / np.sum(exps, axis=-1, keepdims=True)
    shifted = None  # [BLANK] — subtract max for stability
    exps    = None  # [BLANK] — compute e^shifted
    return   None  # [BLANK] — divide each element by the sum
    # └────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — run `python activations.py` to check your answers
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("EXERCISE 1 — Activation Functions")
    print("=" * 55)

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # ── sigmoid ──────────────────────────────────────────────
    result = sigmoid(x)
    expect = np.array([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
    print(f"\nsigmoid({x})")
    print(f"  got:      {result}")
    print(f"  expected: {expect}")

    # ── sigmoid_derivative ───────────────────────────────────
    result = sigmoid_derivative(x)
    expect = np.array([0.1050, 0.1966, 0.2500, 0.1966, 0.1050])
    print(f"\nsigmoid_derivative({x})")
    print(f"  got:      {result}")
    print(f"  expected: {expect}")

    # ── relu ─────────────────────────────────────────────────
    result = relu(x)
    expect = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    print(f"\nrelu({x})")
    print(f"  got:      {result}")
    print(f"  expected: {expect}")

    # ── relu_derivative ──────────────────────────────────────
    result = relu_derivative(x)
    expect = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    print(f"\nrelu_derivative({x})")
    print(f"  got:      {result}")
    print(f"  expected: {expect}")

    # ── softmax ──────────────────────────────────────────────
    scores = np.array([2.0, 1.0, 0.1])
    result = softmax(scores)
    print(f"\nsoftmax({scores})")
    print(f"  got:      {result}")
    sm_sum = None if result is None else f"{result.sum():.4f}"
    print(f"  sums to:  {sm_sum}  (should be 1.0000)")
    print(f"  expected: [0.6590, 0.2424, 0.0986]")
