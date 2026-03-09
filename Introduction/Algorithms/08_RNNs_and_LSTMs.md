# Recurrent Neural Networks (RNNs) and LSTMs

## Table of Contents
1. [Overview](#overview)
2. [Core Theory — Sequential Processing](#core-theory--sequential-processing)
3. [Vanilla RNN Architecture](#vanilla-rnn-architecture)
4. [Backpropagation Through Time (BPTT)](#backpropagation-through-time-bptt)
5. [The Vanishing Gradient Problem in RNNs](#the-vanishing-gradient-problem-in-rnns)
6. [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
   - [Cell State](#cell-state)
   - [Gates](#gates)
   - [LSTM Equations](#lstm-equations)
7. [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
8. [Bidirectional RNNs](#bidirectional-rnns)
9. [Encoder-Decoder (Seq2Seq) Architecture](#encoder-decoder-seq2seq-architecture)
10. [Attention Mechanism (Pre-Transformer)](#attention-mechanism-pre-transformer)
11. [Strengths and Weaknesses](#strengths-and-weaknesses)
12. [Role in ML and LLMs](#role-in-ml-and-llms)

---

## Overview

Recurrent Neural Networks (RNNs) are a class of neural networks designed for **sequential data** — data where order matters and where each element depends on previous context. This includes:
- Natural language (text, speech)
- Time series (financial data, sensor readings)
- Video (sequences of frames)
- Music generation

Unlike feedforward networks, RNNs have **recurrent connections** — the output at each time step feeds back as input to the next step. This allows them to maintain a **hidden state** that acts as memory across the sequence.

RNNs and their successor LSTMs were the dominant approach for NLP from 2013 to 2018, before being supplanted by Transformers.

---

## Core Theory — Sequential Processing

A standard neural network maps `x → ŷ` with no notion of sequence. RNNs process sequences by sharing the same parameters across all time steps:

For a sequence `(x₁, x₂, ..., xₜ)`:
- At each step `t`, the RNN takes current input `xₜ` and previous hidden state `h_{t-1}`
- Produces a new hidden state `hₜ` (updated memory)
- Optionally produces an output `yₜ`

This is analogous to running the same neural network at each time step, where the hidden state carries information forward.

**Parameter sharing across time** is what makes RNNs efficient — the same weights `W` are used at every position, just as convolution shares weights across space.

---

## Vanilla RNN Architecture

```
hₜ = tanh(Wₓₓ · xₜ + Wₕₕ · h_{t-1} + bₕ)
yₜ = Wᵧₕ · hₜ + bᵧ
```

Where:
- `xₜ ∈ ℝᵈ`: input at time step t
- `h_{t-1} ∈ ℝʰ`: previous hidden state (initialized to zeros: `h₀ = 0`)
- `hₜ ∈ ℝʰ`: current hidden state
- `yₜ ∈ ℝᵒ`: output at time step t
- `Wₓₓ ∈ ℝ^{h×d}`, `Wₕₕ ∈ ℝ^{h×h}`, `Wᵧₕ ∈ ℝ^{o×h}`: weight matrices

The total parameter count is `h*d + h*h + o*h` — far fewer than an equivalent feedforward network.

**Common RNN configurations:**
| Task | Input-Output Pattern | Example |
|---|---|---|
| Many-to-One | Sequence → Single | Sentiment analysis |
| One-to-Many | Single → Sequence | Image captioning |
| Many-to-Many (same length) | Sequence → Sequence | POS tagging |
| Many-to-Many (different length) | Sequence → Sequence | Machine translation (Seq2Seq) |

---

## Backpropagation Through Time (BPTT)

RNNs are trained by "unrolling" the computation graph through time and applying standard backpropagation:

```
Loss L = Σₜ Lₜ(yₜ, ŷₜ)

∂L/∂Wₕₕ = Σₜ ∂Lₜ/∂hₜ * ∂hₜ/∂Wₕₕ

∂Lₜ/∂h_k = ∂Lₜ/∂hₜ * Πⱼ₌ₖ₊₁ᵗ ∂hⱼ/∂h_{j-1}
```

The chain of gradients from time `t` back to time `k` involves a product of `t-k` Jacobians of the form `∂hⱼ/∂h_{j-1} = diag(tanh'(zⱼ)) * Wₕₕ`.

For long sequences, this product either:
- **Vanishes** (if `||Wₕₕ|| < 1`) — gradients become zero
- **Explodes** (if `||Wₕₕ|| > 1`) — gradients diverge

**Truncated BPTT**: A practical approximation that only backpropagates through the last `k` time steps (e.g., k=20), limiting both computational cost and gradient issues.

---

## The Vanishing Gradient Problem in RNNs

The vanishing gradient is **catastrophic for RNNs** because it means the model cannot learn **long-range dependencies**.

If a token at position `t=100` needs to influence the gradient of a token at position `t=1`, the gradient must pass through 99 multiplication steps. With the tanh activation (`|tanh'| ≤ 1`), this product collapses to zero.

In practice, vanilla RNNs can only reliably learn dependencies spanning ~10-20 steps. This severely limits their performance on tasks requiring long-range context (which is most NLP tasks).

The solution: **LSTM** and **GRU**.

---

## Long Short-Term Memory (LSTM)

LSTMs (Hochreiter & Schmidhuber, 1997) solve the vanishing gradient problem through a fundamentally different memory mechanism: the **cell state** — an information highway that runs straight through time with minimal transformation.

### Cell State

The cell state `Cₜ ∈ ℝʰ` is a vector of memory that can carry information across hundreds of time steps. Crucially, information is added or removed via **multiplicative gates** (not tanh squashing). The gradient of the cell state is the product of gate values — which can be kept close to 1, allowing gradients to flow.

### Gates

LSTMs have three learnable gates, each producing values in (0, 1) via sigmoid:

| Gate | Symbol | Role |
|---|---|---|
| Forget Gate | `fₜ` | What fraction of previous cell state to forget |
| Input Gate | `iₜ` | What new information to write to cell state |
| Output Gate | `oₜ` | What part of cell state to expose as hidden state |

Each gate takes both `xₜ` (current input) and `h_{t-1}` (previous hidden state) as inputs.

### LSTM Equations

```
fₜ = σ(Wf · [h_{t-1}, xₜ] + bf)          (Forget gate)
iₜ = σ(Wi · [h_{t-1}, xₜ] + bi)          (Input gate)
C̃ₜ = tanh(Wc · [h_{t-1}, xₜ] + bc)      (Candidate values)
Cₜ = fₜ ⊙ C_{t-1} + iₜ ⊙ C̃ₜ            (Cell state update)
oₜ = σ(Wo · [h_{t-1}, xₜ] + bo)          (Output gate)
hₜ = oₜ ⊙ tanh(Cₜ)                       (Hidden state)
```

Where `⊙` is element-wise multiplication (Hadamard product).

**Gradient flow through cell state:**
```
∂Cₜ/∂C_{t-1} = fₜ
```

The gradient of the cell state only passes through the forget gate value `fₜ`. If `fₜ ≈ 1` (don't forget), the gradient flows unchanged — no vanishing!

This is analogous to the **residual connections** in Transformers (though discovered 18 years earlier).

---

## Gated Recurrent Unit (GRU)

GRUs (Cho et al., 2014) are a simplified LSTM with only two gates and no separate cell state:

```
zₜ = σ(Wz · [h_{t-1}, xₜ])     (Update gate: controls how much to update)
rₜ = σ(Wr · [h_{t-1}, xₜ])     (Reset gate: controls how much past to use)
h̃ₜ = tanh(W · [rₜ ⊙ h_{t-1}, xₜ])   (Candidate hidden state)
hₜ = (1 - zₜ) ⊙ h_{t-1} + zₜ ⊙ h̃ₜ  (New hidden state)
```

GRUs have:
- Fewer parameters than LSTM
- Similar performance in practice
- Faster training due to simpler computation

The **update gate** `zₜ` acts like both the forget and input gates combined. When `zₜ = 0`, the hidden state is unchanged (perfect memory). When `zₜ = 1`, the hidden state is completely replaced.

---

## Bidirectional RNNs

A standard RNN only processes left-to-right. **Bidirectional RNNs** (BiRNNs) run two separate RNNs:
- Forward: processes sequence left-to-right
- Backward: processes sequence right-to-left

The hidden states are concatenated at each time step:
```
h̄ₜ = [h→ₜ; h←ₜ]
```

BiRNNs allow each position to have context from **both past and future** — critical for tasks like Named Entity Recognition, POS tagging, and BERT-style language understanding.

---

## Encoder-Decoder (Seq2Seq) Architecture

For variable-length input → variable-length output (translation, summarization):

```
Encoder: Processes input sequence x₁,...,xₜ
         Final hidden state c = h_T (context vector)

Decoder: Generates output sequence y₁,...,yₜ'
         Conditioned on context vector c
         At each step: hₜ' = f(h_{t'-1}, y_{t'-1}, c)
```

The encoder "compresses" the entire input sequence into a single fixed-length context vector `c`. This is a fundamental bottleneck — for long sequences, the context vector can't carry all information. **Attention** was introduced to solve this.

---

## Attention Mechanism (Pre-Transformer)

Bahdanau et al. (2015) introduced attention to allow the decoder to look at the **entire encoder sequence**, not just the final hidden state:

```
eₜₜ' = score(h_{t'}, h̄ₜ)         (Alignment score)
αₜₜ' = softmax(eₜₜ')             (Attention weights)
cₜ' = Σₜ αₜₜ' * h̄ₜ             (Context vector for step t')
```

Where the score function can be:
- Dot product: `h_{t'}ᵀ h̄ₜ`
- Additive (Bahdanau): `vᵀ tanh(W₁h̄ₜ + W₂h_{t'})`

This **soft alignment** mechanism allows the decoder to attend to different source positions at each decoding step — revolutionizing machine translation and directly inspiring the Transformer.

---

## Strengths and Weaknesses

**Strengths:**
- Naturally handles variable-length sequences
- Parameter sharing across time (efficient)
- LSTM/GRU can handle long-range dependencies much better than vanilla RNN
- Bidirectional variants provide full context

**Weaknesses:**
- **Sequential computation**: cannot parallelize across time steps — slow training
- Still struggles with very long sequences (thousands of tokens)
- Fixed hidden state dimension limits memory capacity
- Even LSTMs have ~5× fewer parameters than Transformers at comparable performance
- Transformers generally outperform LSTMs given sufficient data

---

## Role in ML and LLMs

RNNs and LSTMs are the direct predecessors of modern LLMs:

- **The attention mechanism** invented for Seq2Seq RNNs became the foundation of the Transformer
- The **encoder-decoder paradigm** (introduced for RNNs) is retained in T5, BART, and translation models
- **ELMo** (2018), the first contextual word embedding model, used bidirectional LSTMs — the direct precursor to BERT
- LSTMs are still used in **streaming speech recognition** and **on-device NLP** where computational efficiency matters
- Many **time-series forecasting** models (financial, weather) still use LSTMs effectively
- The concept of a **hidden state as memory** directly inspired the **KV cache** in Transformers and **state-space models** like Mamba, which revisit recurrent processing for efficiency
