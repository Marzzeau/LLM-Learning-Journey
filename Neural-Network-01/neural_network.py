"""
EXERCISE 3 — The Neural Network
=================================
Now we assemble the network from individual layers.

Architecture: Multi-Layer Perceptron (MLP)

    Input → [Dense Layer → Activation] × N layers → Output

Key concepts:
  Forward pass  — data flows input → output, producing a prediction.
  Backward pass — gradients flow output → input (backpropagation).
  Weight update — adjust W and b to reduce the loss (gradient descent).

Classes to complete:
  DenseLayer       — one fully-connected layer
  NeuralNetwork    — stacks DenseLayer objects into a complete model

Run:  python neural_network.py
"""

import numpy as np
from activations import sigmoid, sigmoid_derivative, relu, relu_derivative


# ─────────────────────────────────────────────────────────────────────────────
# DENSE LAYER
# ─────────────────────────────────────────────────────────────────────────────

class DenseLayer:
    """
    A single fully-connected (dense) layer.

    Every neuron in this layer is connected to every neuron in the previous
    layer. The computation is:

        z      = x @ W + b      (linear transformation)
        output = activation(z)  (non-linearity)

    Parameters
    ----------
    input_size  : int   — number of features coming in
    output_size : int   — number of neurons in this layer
    activation  : str   — 'relu', 'sigmoid', or 'linear'
    """

    def __init__(self, input_size, output_size, activation='relu'):
        self.activation_name = activation

        # ┌─ BLANK 1 — Weight initialisation ──────────────────────────────────
        # │
        # │  Use He initialisation for ReLU, Xavier for sigmoid/linear.
        # │
        # │    ReLU   → scale = sqrt(2 / input_size)          (He)
        # │    others → scale = sqrt(2 / (input_size + output_size)) (Xavier)
        # │
        # │  self.W shape: (input_size, output_size)
        # │  Hint: np.random.randn(input_size, output_size) * scale
        # │
        # │  Biases: start at zero.
        # │  self.b shape: (1, output_size)
        # │  Hint: np.zeros((1, output_size))
        if activation == 'relu':
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(2.0 / (input_size + output_size))
        self.W   = None  # [BLANK] initialise weights
        self.b   = None  # [BLANK] initialise biases
        # └────────────────────────────────────────────────────────────────────

        # Cache filled during forward / backward (do NOT modify these):
        self.input  = None   # the x we received  — needed in backward
        self.z      = None   # pre-activation  z = x @ W + b
        self.output = None   # post-activation a = activation(z)

        # Gradients filled during backward:
        self.dW = None
        self.db = None

    # ── helpers (already complete) ───────────────────────────────────────────

    def _activate(self, z):
        if self.activation_name == 'relu':
            return relu(z)
        elif self.activation_name == 'sigmoid':
            return sigmoid(z)
        else:                        # 'linear' — no activation
            return z

    def _activate_derivative(self, z):
        if self.activation_name == 'relu':
            return relu_derivative(z)
        elif self.activation_name == 'sigmoid':
            return sigmoid_derivative(z)
        else:
            return np.ones_like(z)

    # ── forward pass ─────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Compute this layer's output from input x.

        Steps:
          1. Save x  (needed later in backward)
          2. z      = x @ W + b          (linear step)
          3. output = activation(z)      (non-linear step)
          4. Return output

        Args:
            x : numpy array shape (batch_size, input_size)
        Returns:
            numpy array shape (batch_size, output_size)
        """
        self.input = x   # already done — save the input

        # ┌─ BLANK 2 ──────────────────────────────────────────────────────────
        # │  Compute the pre-activation z = x @ self.W + self.b
        # │  Hint: matrix multiply with @, then add self.b (broadcasts over batch)
        self.z = None  # [BLANK]

        # │  Apply the activation function
        # │  Hint: self._activate(self.z)
        self.output = None  # [BLANK]
        # └────────────────────────────────────────────────────────────────────

        return self.output

    # ── backward pass ────────────────────────────────────────────────────────

    def backward(self, d_output):
        """
        Compute gradients and propagate them to the previous layer.

        The chain rule gives us (shapes noted in brackets):

          d_z     = d_output  *  activation'(z)    [batch, out]  element-wise
          dW      = input.T   @  d_z               [in, out]
          db      = sum(d_z, axis=0, keepdims=True) [1, out]
          d_input = d_z       @  W.T               [batch, in]

        We store dW and db so update_weights() can use them later.
        We return d_input so it can be passed to the previous layer.

        Args:
            d_output : numpy array shape (batch_size, output_size)
                       — gradient of the loss w.r.t. this layer's OUTPUT
        Returns:
            d_input  : numpy array shape (batch_size, input_size)
                       — gradient of the loss w.r.t. this layer's INPUT
        """
        # ┌─ BLANK 3 ──────────────────────────────────────────────────────────
        # │
        # │  Step 1: apply activation derivative (chain rule through activation)
        # │  d_z = d_output * self._activate_derivative(self.z)
        d_z = None  # [BLANK]

        # │  Step 2: gradient w.r.t. weights
        # │  self.dW = self.input.T @ d_z
        self.dW = None  # [BLANK]

        # │  Step 3: gradient w.r.t. biases  (sum over the batch dimension)
        # │  self.db = np.sum(d_z, axis=0, keepdims=True)
        self.db = None  # [BLANK]

        # │  Step 4: gradient to send to the PREVIOUS layer
        # │  d_input = d_z @ self.W.T
        d_input = None  # [BLANK]
        # └────────────────────────────────────────────────────────────────────

        return d_input


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK  (chains layers together)
# ─────────────────────────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    A multi-layer perceptron built from DenseLayer objects.

    Usage example:
        nn = NeuralNetwork([
            DenseLayer(2, 4, activation='relu'),
            DenseLayer(4, 1, activation='sigmoid'),
        ])
        predictions = nn.forward(X)
        nn.backward(grad_of_loss)
        nn.update_weights(learning_rate=0.1)
    """

    def __init__(self, layers):
        """
        Args:
            layers : list of DenseLayer — ordered from input to output.
        """
        self.layers = layers

    def forward(self, x):
        """
        Pass x through every layer in sequence.

        Args:
            x : numpy array shape (batch_size, n_features)
        Returns:
            Final output after all layers.
        """
        # ┌─ BLANK 4 ──────────────────────────────────────────────────────────
        # │  Loop through self.layers in order.
        # │  Pass x through each layer and update x with the result.
        # │
        # │  for layer in self.layers:
        # │      x = layer.forward(x)
        # │  return x
        pass  # [BLANK] — replace `pass` with the loop and return
        # └────────────────────────────────────────────────────────────────────

    def backward(self, d_loss):
        """
        Backpropagate the loss gradient through every layer in REVERSE order.

        Args:
            d_loss : numpy array — gradient of the loss w.r.t. the final output.
        """
        # ┌─ BLANK 5 ──────────────────────────────────────────────────────────
        # │  Loop through self.layers in REVERSE order.
        # │  Each layer's backward() takes the gradient and returns a new one.
        # │
        # │  grad = d_loss
        # │  for layer in reversed(self.layers):
        # │      grad = layer.backward(grad)
        pass  # [BLANK] — replace `pass` with the loop
        # └────────────────────────────────────────────────────────────────────

    def update_weights(self, learning_rate):
        """
        Apply gradient descent to every layer's weights and biases.

        Gradient descent rule:
            W ← W − lr · dW
            b ← b − lr · db

        Args:
            learning_rate : float — step size (e.g. 0.01)
        """
        # ┌─ BLANK 6 ──────────────────────────────────────────────────────────
        # │  for layer in self.layers:
        # │      layer.W -= learning_rate * layer.dW
        # │      layer.b -= learning_rate * layer.db
        pass  # [BLANK] — replace `pass` with the loop
        # └────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 55)
    print("EXERCISE 3 — Neural Network")
    print("=" * 55)

    nn = NeuralNetwork([
        DenseLayer(2, 4, activation='relu'),
        DenseLayer(4, 1, activation='sigmoid'),
    ])

    x = np.array([[1.0, 2.0]])

    print("\nForward pass:")
    out = nn.forward(x)
    print(f"  output : {out}")
    print(f"  shape  : {None if out is None else out.shape}  (expected (1, 1))")
    print(f"  range  : (0, 1)  — should look like a probability")

    print("\nBackward pass:")
    d_loss = np.ones((1, 1))
    nn.backward(d_loss)
    dw0 = nn.layers[0].dW
    dw1 = nn.layers[1].dW
    print(f"  layers[0].dW shape: {None if dw0 is None else dw0.shape}  (expected (2, 4))")
    print(f"  layers[1].dW shape: {None if dw1 is None else dw1.shape}  (expected (4, 1))")

    print("\nIf shapes look right, move on to train_xor.py!")
