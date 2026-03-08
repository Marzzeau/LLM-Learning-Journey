"""
SOLUTION 4 — Training: The XOR Problem
(Only open this after attempting train_xor.py yourself!)
"""

import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_network import NeuralNetwork, DenseLayer
from loss_functions import binary_cross_entropy, binary_cross_entropy_derivative

np.random.seed(42)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

model = NeuralNetwork([
    DenseLayer(2, 4, activation='relu'),
    DenseLayer(4, 1, activation='sigmoid'),
])

learning_rate = 0.5
epochs        = 5000

print("Training on XOR...\n")
for epoch in range(epochs):
    y_pred = model.forward(X)
    loss   = binary_cross_entropy(y_pred, y)
    d_loss = binary_cross_entropy_derivative(y_pred, y)
    model.backward(d_loss)
    model.update_weights(learning_rate)
    if epoch % 500 == 0:
        print(f"  Epoch {epoch:5d}  |  loss: {loss:.4f}")

print("\nFinal predictions:")
print(f"  {'x1':>3} {'x2':>3}  |  {'raw':>6}  predicted  true")
print(f"  {'-'*40}")
y_pred = model.forward(X)
for i in range(len(X)):
    raw       = y_pred[i, 0]
    predicted = 1 if raw > 0.5 else 0
    true      = int(y[i, 0])
    mark      = "OK" if predicted == true else "WRONG"
    print(f"  {int(X[i,0]):>3} {int(X[i,1]):>3}  |  {raw:.4f}  {predicted:>9}  {true:>4}  {mark}")
