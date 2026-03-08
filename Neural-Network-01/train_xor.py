"""
EXERCISE 4 — Training: The XOR Problem
========================================
XOR is a classic test for neural networks because it is *not* linearly
separable — a single layer cannot solve it, but a hidden layer can.

Truth table:
    x1  x2  | y
   ---------+---
    0   0   | 0
    0   1   | 1
    1   0   | 1
    1   1   | 0

Your tasks:
  1. Define the model architecture
  2. Complete the training loop  (forward → loss → backward → update)
  3. Watch the loss fall to near 0 and predictions snap to correct values

Run:  python train_xor.py
"""

import numpy as np
from neural_network   import NeuralNetwork, DenseLayer
from loss_functions   import binary_cross_entropy, binary_cross_entropy_derivative

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

y = np.array([[0], [1], [1], [0]], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

# ┌─ BLANK 1 ──────────────────────────────────────────────────────────────────
# │  Define a NeuralNetwork that can solve XOR.
# │  Minimum architecture: 2 inputs → hidden (relu) → 1 output (sigmoid)
# │  A hidden layer with 4 neurons is enough.
# │
# │  model = NeuralNetwork([
# │      DenseLayer(2, 4, activation='relu'),
# │      DenseLayer(4, 1, activation='sigmoid'),
# │  ])
model = None  # [BLANK] — replace None with a NeuralNetwork(...)
# └────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

learning_rate = 0.5
epochs        = 5000

print("Training on XOR...\n")

for epoch in range(epochs):

    # ┌─ BLANK 2 ──────────────────────────────────────────────────────────────
    # │
    # │  Step 1 — Forward pass: get predictions
    # │  y_pred = model.forward(X)
    y_pred = None  # [BLANK]

    # │  Step 2 — Compute the loss (how wrong are we?)
    # │  loss = binary_cross_entropy(y_pred, y)
    loss = None    # [BLANK]

    # │  Step 3 — Compute loss gradient (the starting point for backprop)
    # │  d_loss = binary_cross_entropy_derivative(y_pred, y)
    d_loss = None  # [BLANK]

    # │  Step 4 — Backward pass: propagate gradients through the network
    # │  model.backward(d_loss)
    # [BLANK — one line]

    # │  Step 5 — Update weights using gradient descent
    # │  model.update_weights(learning_rate)
    # [BLANK — one line]
    # └────────────────────────────────────────────────────────────────────────

    if epoch % 500 == 0:
        print(f"  Epoch {epoch:5d}  |  loss: {loss}")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

print("\nFinal predictions:")
print(f"  {'x1':>3} {'x2':>3}  |  {'raw':>6}  predicted  true")
print(f"  {'-'*40}")

y_pred = model.forward(X) if model is not None else [None] * 4

for i in range(len(X)):
    raw       = y_pred[i, 0] if y_pred[i] is not None else None
    predicted = 1 if (raw is not None and raw > 0.5) else 0
    true      = int(y[i, 0])
    mark      = "OK" if predicted == true else "WRONG"
    print(f"  {int(X[i,0]):>3} {int(X[i,1]):>3}  |  "
          f"{raw if raw is None else f'{raw:.4f}':>6}  "
          f"{predicted:>9}  {true:>4}  {mark}")

print("\nAll 4 should be [OK] if your network converged!")
