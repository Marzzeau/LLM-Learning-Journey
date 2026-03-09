# Convolutional Neural Networks (CNNs)

## Table of Contents
1. [Overview](#overview)
2. [Core Theory — Convolution Operation](#core-theory--convolution-operation)
3. [Key Components](#key-components)
   - [Convolutional Layer](#convolutional-layer)
   - [Activation Function](#activation-function)
   - [Pooling Layer](#pooling-layer)
   - [Fully Connected Layer](#fully-connected-layer)
4. [Feature Hierarchy](#feature-hierarchy)
5. [Receptive Field](#receptive-field)
6. [Important CNN Architectures](#important-cnn-architectures)
7. [Training CNNs](#training-cnns)
8. [Transfer Learning with CNNs](#transfer-learning-with-cnns)
9. [1D Convolutions in NLP](#1d-convolutions-in-nlp)
10. [Strengths and Weaknesses](#strengths-and-weaknesses)
11. [Role in ML and LLMs](#role-in-ml-and-llms)

---

## Overview

Convolutional Neural Networks (CNNs) are specialized neural networks designed to process **grid-structured data** — most commonly images (2D grids of pixels), but also audio (1D) and video (3D).

The key innovation is the **convolutional layer**, which applies learned filters across the entire input — sharing weights spatially. This exploits two fundamental properties of natural data:

1. **Local connectivity**: nearby pixels are more correlated than distant ones
2. **Translation equivariance**: a feature detector useful in one part of an image is useful everywhere

CNNs dominated computer vision from 2012 (AlexNet) onward and their convolutional operations also appeared in early NLP models before Transformers took over.

---

## Core Theory — Convolution Operation

A **convolution** slides a small filter (kernel) across the input and computes a dot product at each position:

For a 1D signal `x` and filter `w` of size `k`:
```
(x * w)[t] = Σᵢ x[t + i] * w[i]   for i = 0,...,k-1
```

For a 2D image `X` (height H, width W) and filter `W` (size `k × k`):
```
(X * F)[i, j] = Σₘ Σₙ X[i+m, j+n] * F[m, n]
```

The output is called a **feature map** or **activation map**.

**Strict mathematical convolution** involves flipping the filter; in deep learning, this operation is technically **cross-correlation** (no flip), but is universally called "convolution."

---

## Key Components

### Convolutional Layer

Parameters:
- `filters` (K): Number of learned filters — produces K feature maps
- `kernel_size` (k): Spatial size of each filter (e.g., 3×3, 5×5)
- `stride` (s): How many pixels the filter moves at each step (default: 1)
- `padding` (p): Zeros added around the border to control output size

Output size formula:
```
output_size = floor((input_size - kernel_size + 2 * padding) / stride) + 1
```

**Parameter sharing:** Each filter has `k × k × C_in` weights, shared across all spatial locations. This is what makes CNNs so parameter-efficient compared to fully connected layers.

For an image of size `H × W × C` with `K` filters of size `k × k`:
- MLP (fully connected) would need: `H * W * C * K` weights per output pixel
- CNN needs: `k * k * C * K` weights total (shared across all positions)

### Activation Function

After each convolution, a nonlinearity is applied element-wise — almost universally **ReLU**:
```
a[i,j,k] = max(0, z[i,j,k])
```

### Pooling Layer

Pooling reduces the spatial dimensions of feature maps, achieving:
- Computational reduction
- Translation invariance
- Increasing receptive field

**Max Pooling:** Takes the maximum value in each local window
```
output[i,j] = max over (i*s:i*s+k, j*s:j*s+k) of input
```

**Average Pooling:** Takes the mean value — smoother, used in later architectures

**Global Average Pooling (GAP):** Takes mean across entire spatial dimensions — produces one value per channel. Used in modern architectures instead of fully connected layers to reduce parameters.

### Fully Connected Layer

After convolutional and pooling layers extract features, one or more FC layers perform classification. The final layer uses softmax for multi-class problems.

A typical CNN architecture:
```
Input Image → [Conv → ReLU → Pool] × N → Flatten → [FC → ReLU] × M → Softmax → Output
```

---

## Feature Hierarchy

CNNs learn a **hierarchical feature representation** through depth:

```
Layer 1:  Edges, corners, color gradients (Gabor-like filters)
Layer 2:  Textures, simple patterns (combinations of edges)
Layer 3:  Object parts (wheels, eyes, windows)
Layer 4:  Objects (faces, cars, dogs)
Layer 5+: Abstract semantic features
```

This is not hand-designed — it emerges automatically from training on labeled data. Visualizations of learned filters confirm this hierarchy.

---

## Receptive Field

The **receptive field** of a neuron is the region of the input that influences its activation.

With stride=1, padding=0, kernel size `k`:
- Layer 1: receptive field = `k`
- Layer 2: receptive field = `k + (k-1)` = `2k - 1`
- Layer L: receptive field = `L * (k-1) + 1`

Deep networks gradually increase receptive field, allowing neurons in later layers to "see" global context.

**Dilated (Atrous) Convolutions** insert gaps between kernel elements, exponentially increasing receptive field without more parameters:
```
Dilation rate d: (x * w)[t] = Σᵢ x[t + d*i] * w[i]
```

---

## Important CNN Architectures

| Architecture | Year | Key Innovation |
|---|---|---|
| LeNet-5 | 1998 | First practical CNN; handwriting recognition |
| AlexNet | 2012 | Deep CNN on GPU; ReLU; dropout; ImageNet breakthrough |
| VGGNet | 2014 | Very deep (16-19 layers) with 3×3 filters only |
| GoogLeNet/Inception | 2014 | Inception modules (parallel multi-scale convolutions) |
| ResNet | 2015 | Residual connections; 152 layers; solved vanishing gradients |
| DenseNet | 2016 | Each layer connected to all subsequent layers |
| EfficientNet | 2019 | Compound scaling of width/depth/resolution |
| ConvNeXt | 2022 | Modernized ResNet matching Transformer performance |

**ResNet Residual Connection:**
```
output = F(x, {Wᵢ}) + x
```
The identity shortcut allows gradients to flow directly through layers, enabling extremely deep networks (100+ layers).

---

## Training CNNs

CNNs are trained with standard **backpropagation + gradient descent** (typically Adam or SGD + momentum).

**Data Augmentation** is critical for CNNs to generalize:
- Random crops and flips
- Color jitter, brightness, contrast
- Rotation, scaling
- Mixup, CutMix (label-mixing augmentations)

**Batch Normalization** (introduced in 2015) normalizes each feature map across the batch:
```
x̂ = (x - μ_B) / √(σ²_B + ε)
y = γ * x̂ + β
```
Dramatically stabilizes training, allows higher learning rates, reduces need for dropout.

**Learning Rate Scheduling:**
- Step decay: reduce LR by factor at fixed intervals
- Cosine annealing: smoothly decay to zero
- Warm-up: start with small LR, increase to target, then decay

---

## Transfer Learning with CNNs

Pre-trained CNNs (trained on ImageNet with 1.2M images) learn universal visual features. **Transfer learning** reuses these:

**Fine-tuning strategies:**
1. **Feature extraction**: Freeze all layers, replace final classifier, train only the new head
2. **Partial fine-tuning**: Freeze early layers (generic features), fine-tune later layers (task-specific)
3. **Full fine-tuning**: Update all weights with a small learning rate

Rule of thumb:
- Small target dataset → Feature extraction
- Large target dataset → Full fine-tuning

---

## 1D Convolutions in NLP

CNNs were applied to NLP before Transformers via 1D convolutions over word embeddings:

```
Input: sentence of T words → embedding matrix of shape (T, d)
Filter: shape (k, d) — a k-gram detector
Output: feature map of shape (T-k+1, 1)
```

Multiple filters of different sizes (2, 3, 4, 5) capture different n-gram features. Global max pooling over time extracts the most salient feature per filter.

This **TextCNN** architecture is fast, simple, and surprisingly effective for text classification. It is still used in applications requiring low latency.

---

## Strengths and Weaknesses

**Strengths:**
- Dramatically fewer parameters than FC networks for images (weight sharing)
- Built-in translation equivariance
- Hierarchical feature learning is well-suited to visual data
- Excellent on structured grid data (images, audio spectrograms)

**Weaknesses:**
- Not naturally suited to variable-length, non-grid data
- Cannot easily model **long-range dependencies** (limited receptive field in shallow nets)
- Lacks global context — Transformers can attend to any position
- Requires a lot of data to train from scratch

---

## Role in ML and LLMs

CNNs are foundational to the visual half of multimodal LLMs:

- **CLIP** (OpenAI) uses a CNN or Vision Transformer to encode images alongside a text Transformer
- **GPT-4V**, **LLaVA**, **Gemini** use visual encoders (CNN or ViT) whose outputs are projected into the LLM's token space
- **Vision Transformer (ViT)**: splits images into patches and processes them as a sequence — essentially treating image patches as "tokens" for a Transformer
- CNN-style **1D convolutional layers** still appear in some LLM positional encoding and tokenizer architectures
- **Dilated causal convolutions** are used in Wavenet-style audio generation, which powers modern text-to-speech systems used alongside LLMs
