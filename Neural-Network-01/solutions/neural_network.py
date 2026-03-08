"""
SOLUTION 3 — The Neural Network
(Only open this after attempting neural_network.py yourself!)
"""

import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activations import sigmoid, sigmoid_derivative, relu, relu_derivative


class DenseLayer:

    def __init__(self, input_size, output_size, activation='relu'):
        self.activation_name = activation
        # He init for ReLU, Xavier for sigmoid/linear
        if activation == 'relu':
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(2.0 / (input_size + output_size))
        self.W   = np.random.randn(input_size, output_size) * scale
        self.b   = np.zeros((1, output_size))
        self.input  = None
        self.z      = None
        self.output = None
        self.dW = None
        self.db = None

    def _activate(self, z):
        if self.activation_name == 'relu':
            return relu(z)
        elif self.activation_name == 'sigmoid':
            return sigmoid(z)
        return z

    def _activate_derivative(self, z):
        if self.activation_name == 'relu':
            return relu_derivative(z)
        elif self.activation_name == 'sigmoid':
            return sigmoid_derivative(z)
        return np.ones_like(z)

    def forward(self, x):
        self.input  = x
        self.z      = x @ self.W + self.b
        self.output = self._activate(self.z)
        return self.output

    def backward(self, d_output):
        d_z     = d_output * self._activate_derivative(self.z)
        self.dW = self.input.T @ d_z
        self.db = np.sum(d_z, axis=0, keepdims=True)
        d_input = d_z @ self.W.T
        return d_input


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_loss):
        grad = d_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db


if __name__ == "__main__":
    np.random.seed(42)
    nn = NeuralNetwork([
        DenseLayer(2, 4, activation='relu'),
        DenseLayer(4, 1, activation='sigmoid'),
    ])
    x = np.array([[1.0, 2.0]])
    out = nn.forward(x)
    print(f"output: {out}  shape: {out.shape}")
    nn.backward(np.ones((1, 1)))
    print(f"dW[0] shape: {nn.layers[0].dW.shape}")
    print(f"dW[1] shape: {nn.layers[1].dW.shape}")
