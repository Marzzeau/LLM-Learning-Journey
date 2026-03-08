"""
SOLUTION 1 — Activation Functions
(Only open this after attempting activations.py yourself!)
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps    = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


if __name__ == "__main__":
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("sigmoid:            ", sigmoid(x).round(4))
    print("sigmoid_derivative: ", sigmoid_derivative(x).round(4))
    print("relu:               ", relu(x))
    print("relu_derivative:    ", relu_derivative(x))
    scores = np.array([2.0, 1.0, 0.1])
    sm = softmax(scores)
    print(f"softmax:             {sm.round(4)}  (sum={sm.sum():.4f})")
