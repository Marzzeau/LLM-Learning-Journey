"""
SOLUTION 2 — Loss Functions
(Only open this after attempting loss_functions.py yourself!)
"""

import numpy as np


def mse(y_pred, y_true):
    squared_errors = (y_pred - y_true) ** 2
    return np.mean(squared_errors)


def mse_derivative(y_pred, y_true):
    n = len(y_true)
    return (2 / n) * (y_pred - y_true)


def binary_cross_entropy(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_derivative(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    n = len(y_true)
    return (y_pred - y_true) / (n * y_pred * (1 - y_pred))


if __name__ == "__main__":
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    print(f"MSE loss:     {mse(y_pred, y_true):.4f}  (expected 0.0250)")
    print(f"MSE grad:     {mse_derivative(y_pred, y_true)}")
    print(f"BCE loss:     {binary_cross_entropy(y_pred, y_true):.4f}  (expected ~0.1643)")
    print(f"BCE grad:     {binary_cross_entropy_derivative(y_pred, y_true).round(4)}")
