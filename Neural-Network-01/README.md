# Neural Network from Scratch — Learning Exercise

Build a working neural network using only NumPy, step by step.
Each file has blanks marked `None  # [BLANK]` or `pass  # [BLANK]`.
Fill them in, run the file, and check your output against the expected values.

---

## Setup

```bash
pip install numpy
```

---

## Work through the files in order

### 1. `activations.py` — Activation Functions
Implement: `sigmoid`, `sigmoid_derivative`, `relu`, `relu_derivative`, `softmax`

```bash
python activations.py
```

These are the non-linear functions applied after each layer.
Without them, a deep network collapses into a single matrix multiply.

---

### 2. `loss_functions.py` — Loss Functions
Implement: `mse`, `mse_derivative`, `binary_cross_entropy`, `binary_cross_entropy_derivative`

```bash
python loss_functions.py
```

The loss tells the network *how wrong* it is.
The derivative tells gradient descent *which direction* to nudge the weights.

---

### 3. `neural_network.py` — The Network
Implement:
- `DenseLayer.__init__`    — weight & bias initialisation
- `DenseLayer.forward`     — linear transform + activation
- `DenseLayer.backward`    — backpropagation through one layer
- `NeuralNetwork.forward`  — chain layers input → output
- `NeuralNetwork.backward` — chain gradients output → input
- `NeuralNetwork.update_weights` — gradient descent step

```bash
python neural_network.py
```

---

### 4. `train_xor.py` — Put It All Together
Define the model, then complete the training loop.

```bash
python train_xor.py
```

XOR is the "hello world" of neural networks.
After ~5000 epochs all four predictions should round to the correct label.

---

## Key concepts at a glance

```
Forward pass
────────────
  x  →  z = x @ W + b  →  a = activation(z)  →  ...  →  ŷ  →  loss

Backward pass (chain rule)
──────────────────────────
  d_z      = d_a  *  activation'(z)       ← multiply by activation derivative
  dW       = x.T  @  d_z                  ← how much did each weight contribute?
  db       = sum(d_z,  axis=0)            ← same for biases
  d_x      = d_z  @  W.T                  ← pass gradient to previous layer

Gradient descent
────────────────
  W ← W − lr · dW
  b ← b − lr · db
```

---

## Stuck? Check the solutions

Each file has a complete solution in `solutions/`:

| Exercise          | Solution                      |
|-------------------|-------------------------------|
| `activations.py`  | `solutions/activations.py`    |
| `loss_functions.py` | `solutions/loss_functions.py` |
| `neural_network.py` | `solutions/neural_network.py` |
| `train_xor.py`    | `solutions/train_xor.py`      |

Run a solution from the `solutions/` directory:

```bash
cd solutions
python train_xor.py
```

---

## Expected training output (exercise 4)

```
Training on XOR...

  Epoch     0  |  loss: 0.6931
  Epoch   500  |  loss: 0.6203
  Epoch  1000  |  loss: 0.3891
  ...
  Epoch  4500  |  loss: 0.0071

Final predictions:
   x1  x2  |    raw  predicted  true
  ----------------------------------------
    0   0  |  0.0027          0     0  OK
    0   1  |  0.9981          1     1  OK
    1   0  |  0.9983          1     1  OK
    1   1  |  0.0029          0     0  OK
```
