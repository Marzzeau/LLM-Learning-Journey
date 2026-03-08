"""
EXERCISE 2 — Loss Functions
============================
A loss function measures how wrong the network's predictions are.
We minimise it during training. Lower loss = better predictions.

Two loss functions to implement:
  • Mean Squared Error (MSE)       — for regression
  • Binary Cross-Entropy (BCE)     — for binary classification

You also need their *derivatives*, which tell gradient descent which
direction to push the predictions to reduce the loss.

Run:  python loss_functions.py
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MEAN SQUARED ERROR  (regression tasks)
# ─────────────────────────────────────────────────────────────────────────────

def mse(y_pred, y_true):
    """
    Average squared difference between predictions and targets.

    Formula: MSE = (1/n) · Σ (ŷ − y)²

    Args:
        y_pred : numpy array shape (n,) or (n, 1) — network output
        y_true : numpy array shape (n,) or (n, 1) — ground-truth values
    Returns:
        Scalar loss value.
    """
    # ┌─ BLANK 1 ──────────────────────────────────────────────────────────────
    # │  Step a: squared_errors = (y_pred - y_true) ** 2
    # │  Step b: return np.mean(squared_errors)
    squared_errors = None  # [BLANK]
    return          None  # [BLANK]
    # └────────────────────────────────────────────────────────────────────────


def mse_derivative(y_pred, y_true):
    """
    Gradient of MSE with respect to y_pred.

    Formula: dL/dŷ = (2/n) · (ŷ − y)

    This tells us: if ŷ > y, push ŷ down (positive gradient).
                   if ŷ < y, push ŷ up  (negative gradient).

    Args:
        y_pred : numpy array shape (n,) or (n, 1)
        y_true : numpy array shape (n,) or (n, 1)
    Returns:
        numpy array — same shape as inputs.
    """
    n = len(y_true)
    # ┌─ BLANK 2 ──────────────────────────────────────────────────────────────
    # │  Return: (2 / n) * (y_pred - y_true)
    return None  # [BLANK]
    # └────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# BINARY CROSS-ENTROPY  (binary classification tasks)
# ─────────────────────────────────────────────────────────────────────────────

def binary_cross_entropy(y_pred, y_true):
    """
    How surprised we are by the predicted probabilities, given the true labels.
    Penalises confident wrong answers very heavily (log goes to -∞ near 0).

    Formula: BCE = −mean( y·log(ŷ) + (1−y)·log(1−ŷ) )

    Args:
        y_pred : numpy array, values in (0, 1) — predicted probabilities
        y_true : numpy array, values in {0, 1} — true binary labels
    Returns:
        Scalar loss value.
    """
    # Clamp to avoid log(0) = -inf
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)

    # ┌─ BLANK 3 ──────────────────────────────────────────────────────────────
    # │  Return: -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return None  # [BLANK]
    # └────────────────────────────────────────────────────────────────────────


def binary_cross_entropy_derivative(y_pred, y_true):
    """
    Gradient of BCE with respect to y_pred.

    Formula: dL/dŷ = (ŷ − y) / (n · ŷ · (1 − ŷ))

    Args:
        y_pred : numpy array, values in (0, 1)
        y_true : numpy array, values in {0, 1}
    Returns:
        numpy array — same shape as inputs.
    """
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    n = len(y_true)

    # ┌─ BLANK 4 ──────────────────────────────────────────────────────────────
    # │  Return: (y_pred - y_true) / (n * y_pred * (1 - y_pred))
    return None  # [BLANK]
    # └────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("EXERCISE 2 — Loss Functions")
    print("=" * 55)

    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

    # ── MSE ──────────────────────────────────────────────────
    loss = mse(y_pred, y_true)
    grad = mse_derivative(y_pred, y_true)
    print(f"\nMSE loss:      {loss}  (expected 0.0250)")
    print(f"MSE gradient:  {grad}")
    print(f"  expected:    [-0.05  0.05 -0.10  0.10]")

    # ── BCE ──────────────────────────────────────────────────
    loss = binary_cross_entropy(y_pred, y_true)
    grad = binary_cross_entropy_derivative(y_pred, y_true)
    print(f"\nBCE loss:      {loss}  (expected ~0.1643)")
    print(f"BCE gradient:  {grad}")
    print(f"  expected:    [-0.2778  0.2778 -0.4167  0.4167] (roughly)")
