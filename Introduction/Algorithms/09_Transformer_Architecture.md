# The Transformer Architecture

## Table of Contents
1. [Overview](#overview)
2. [Motivation — Limitations of RNNs](#motivation--limitations-of-rnns)
3. [High-Level Architecture](#high-level-architecture)
4. [Input Representation](#input-representation)
   - [Tokenization](#tokenization)
   - [Token Embeddings](#token-embeddings)
   - [Positional Encoding](#positional-encoding)
5. [Scaled Dot-Product Attention](#scaled-dot-product-attention)
6. [Multi-Head Attention](#multi-head-attention)
7. [Feed-Forward Network](#feed-forward-network)
8. [Layer Normalization and Residual Connections](#layer-normalization-and-residual-connections)
9. [Encoder vs Decoder vs Encoder-Decoder](#encoder-vs-decoder-vs-encoder-decoder)
10. [Masked Self-Attention (Causal Masking)](#masked-self-attention-causal-masking)
11. [Cross-Attention](#cross-attention)
12. [Scaling Laws](#scaling-laws)
13. [Key Variants and Descendants](#key-variants-and-descendants)
14. [Computational Complexity](#computational-complexity)
15. [Role as the Foundation of LLMs](#role-as-the-foundation-of-llms)

---

## Overview

The **Transformer** (Vaswani et al., "Attention Is All You Need", 2017) is the neural network architecture that underlies virtually all modern large language models. It completely replaced RNNs for NLP by being:

1. **Fully parallelizable** — no sequential dependence across time steps
2. **Capable of modeling arbitrary-range dependencies** — any token can attend to any other token in O(1) steps
3. **Highly scalable** — performance improves predictably with more compute, data, and parameters

The Transformer introduced the **self-attention mechanism** as the primary computational primitive, replacing recurrence and convolution entirely.

---

## Motivation — Limitations of RNNs

RNNs had three critical limitations that the Transformer addressed:

| Limitation | RNN | Transformer |
|---|---|---|
| Parallelism | Sequential; step `t` depends on `t-1` | Fully parallel; all positions computed simultaneously |
| Long-range dependencies | Gradient vanishes over long sequences | Constant path length between any two positions |
| Representation bottleneck | Single hidden state must compress all context | All positions attend to all others directly |

---

## High-Level Architecture

The original Transformer is an **encoder-decoder** model:

```
Input Sequence
     ↓
[Token Embedding + Positional Encoding]
     ↓
┌─────────────────────────────────┐
│         ENCODER                 │  × N layers
│  Multi-Head Self-Attention      │
│  Add & Norm                     │
│  Feed-Forward Network           │
│  Add & Norm                     │
└─────────────────────────────────┘
     ↓ (encoder output)
┌─────────────────────────────────┐
│         DECODER                 │  × N layers
│  Masked Multi-Head Self-Attention│
│  Add & Norm                     │
│  Multi-Head Cross-Attention     │
│  Add & Norm                     │
│  Feed-Forward Network           │
│  Add & Norm                     │
└─────────────────────────────────┘
     ↓
[Linear + Softmax]
     ↓
Output Probabilities
```

---

## Input Representation

### Tokenization

Text is converted to integer tokens via a vocabulary. The dominant algorithm is **Byte-Pair Encoding (BPE)**:
1. Start with character-level vocabulary
2. Iteratively merge the most frequent adjacent pair
3. Repeat until vocabulary size is reached (e.g., 50,000 for GPT-2)

BPE balances:
- Coverage (unknown words are decomposed into subword units)
- Efficiency (common words are single tokens)
- Compression (text represented compactly)

### Token Embeddings

Each integer token ID is mapped to a dense vector via a learnable **embedding matrix** `E ∈ ℝ^{V×d}`:

```
xₜ = E[token_id_t]   ∈ ℝᵈ
```

The embedding dimension `d` (model dimension / `d_model`) is a critical hyperparameter (typical values: 512, 768, 1024, 4096).

### Positional Encoding

Since self-attention is **permutation-equivariant** (no built-in notion of order), positional information must be explicitly injected.

**Sinusoidal Positional Encoding** (original Transformer):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Properties: fixed (not learned), generalizes to sequence lengths unseen during training.

**Learned Positional Embeddings** (BERT, GPT): positional embeddings are trainable parameters, one per position.

**Rotary Position Embedding (RoPE)** (LLaMA, GPT-NeoX): encodes relative positions by rotating query and key vectors. Enables length generalization and is now standard in most LLMs.

Final input to the model:
```
input = TokenEmbedding(x) + PositionalEncoding(pos)
```

---

## Scaled Dot-Product Attention

The core operation of the Transformer. Given input sequence `X ∈ ℝ^{T×d}`:

Three linear projections:
```
Q = X · Wᵠ    (Queries)    ∈ ℝ^{T×dₖ}
K = X · Wᴷ    (Keys)       ∈ ℝ^{T×dₖ}
V = X · Wᵛ    (Values)     ∈ ℝ^{T×dᵥ}
```

Attention computation:
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

Step by step:
1. `QKᵀ ∈ ℝ^{T×T}`: compute dot product similarity between every query-key pair
2. `/ √dₖ`: scale to prevent softmax saturation (dot products grow with dimension)
3. `softmax(...)`: normalize to get attention weights (probability distribution over positions)
4. `· V`: weighted sum of value vectors — the output for each position

**Intuition:**
- **Query**: "What am I looking for?"
- **Key**: "What do I contain?"
- **Value**: "What do I pass on if attended to?"

The attention matrix `A = softmax(QKᵀ/√dₖ)` is `T×T` — every position attends to every other. Position `i` sees content from position `j` weighted by `A[i,j]`.

---

## Multi-Head Attention

A single attention head captures one type of relationship. **Multi-Head Attention** runs `H` attention heads in parallel, each with different projections:

```
headₕ = Attention(Q·Wₕᵠ, K·Wₕᴷ, V·Wₕᵛ)
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) · Wᵒ
```

Each head uses dimension `dₖ = d_model / H`, so total computation is the same as one large head.

Benefits:
- Different heads learn to attend to different types of relationships (syntactic, semantic, positional, coreference)
- Interpretability studies show specialization: some heads track subject-verb agreement, others track anaphora

Typical values: `H = 8` (original Transformer), `H = 12` (BERT-base), `H = 32` (LLaMA-7B), `H = 96` (GPT-3)

---

## Feed-Forward Network

Each Transformer layer contains a **position-wise Feed-Forward Network (FFN)** applied identically to each position:

```
FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂
```

Or with GELU activation:
```
FFN(x) = GELU(x·W₁ + b₁) · W₂ + b₂
```

Key dimensions:
- Input/output: `d_model` (e.g., 768)
- Inner dimension: `4 * d_model` (e.g., 3072) — the standard ratio is 4×

The FFN applies the same transformation to every token independently. It accounts for ~2/3 of total parameters in large models.

**Mechanistic interpretation**: FFN layers act as **key-value stores** that retrieve factual knowledge. When the model "knows" a fact, it's largely stored in the FFN weights (as demonstrated by factual editing research).

---

## Layer Normalization and Residual Connections

Every sub-layer (attention, FFN) uses:
1. **Residual connection**: adds the input directly to the output
2. **Layer normalization**: normalizes across features

**Pre-norm** (modern standard, used in GPT-3, LLaMA):
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Post-norm** (original Transformer):
```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

Pre-norm is more stable for very deep networks and large learning rates.

**Layer Normalization:**
```
LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
```
Where `μ`, `σ` are mean and std computed over the feature dimension (not batch), and `γ`, `β` are learned scale and shift.

---

## Encoder vs Decoder vs Encoder-Decoder

| Architecture | Self-Attention | Pre-training | Example Models |
|---|---|---|---|
| Encoder-only | Bidirectional (all positions see all) | Masked Language Model (MLM) | BERT, RoBERTa, DistilBERT |
| Decoder-only | Causal/masked (left-to-right only) | Causal Language Model (next token) | GPT-2, GPT-3, LLaMA, Claude |
| Encoder-Decoder | Bidirectional encoder + Causal decoder | Span corruption, seq2seq | T5, BART, mT5 |

**Decoder-only** models dominate modern LLMs because:
- Next-token prediction is a natural, scalable objective
- The entire training data can be used without special masking procedures
- Scales better to very large models

---

## Masked Self-Attention (Causal Masking)

In decoder-only models, the model should only attend to **past and current tokens** — not future tokens (otherwise it would cheat during training and generation would be impossible).

This is enforced by adding a mask to the attention scores before softmax:
```
Masked Attention:
    scores[i,j] = -∞  if j > i   (mask future positions)
    scores[i,j] = QᵢKⱼᵀ/√dₖ   if j ≤ i
```

After softmax, masked positions have weight ≈ 0. This is called **causal** or **autoregressive** attention.

---

## Cross-Attention

In encoder-decoder models, the decoder attends to **encoder outputs** via cross-attention:

```
Q comes from the decoder hidden state
K, V come from the encoder output
```

This allows the decoder to "look at" the encoded input at each generation step — the same mechanism as Bahdanau attention but implemented with dot-product attention.

---

## Scaling Laws

Kaplan et al. (2020) discovered that LLM performance (loss) follows predictable **power laws** with:
- Number of parameters `N`
- Dataset size `D`
- Compute budget `C`

```
L(N) ≈ (Nc/N)^αN       Performance scales with parameters
L(D) ≈ (Dc/D)^αD       Performance scales with data
L(C) ≈ (Cc/C)^αC       Performance scales with compute
```

The Chinchilla scaling law (Hoffmann et al., 2022) refines this:
```
Optimal: N_tokens ≈ 20 × N_params
```

For a 7B parameter model, train on ~140B tokens. This is why LLaMA, Mistral, and similar models train smaller models on vastly more data.

---

## Key Variants and Descendants

| Model | Year | Key Innovation |
|---|---|---|
| BERT | 2018 | Encoder-only; Masked LM + NSP pre-training |
| GPT-2 | 2019 | Decoder-only; scaled language model |
| T5 | 2020 | Unified text-to-text framework |
| GPT-3 | 2020 | 175B params; few-shot learning |
| LLaMA | 2023 | Efficient open-source; RoPE, SwiGLU, RMSNorm |
| Mistral | 2023 | Grouped Query Attention, Sliding Window Attention |
| Mixtral | 2023 | Mixture of Experts (MoE) Transformer |
| Claude | 2023+ | Constitutional AI, RLHF, long context |

---

## Computational Complexity

Self-attention is O(T²·d) in time and space — quadratic in sequence length T:

| Operation | Complexity |
|---|---|
| Self-Attention | O(T² · d) |
| FFN | O(T · d²) |

For long contexts (T > 4096), the quadratic attention becomes a bottleneck. Solutions:
- **Sparse Attention**: only attend to nearby tokens + global tokens (Longformer, BigBird)
- **Flash Attention**: reorders computation to reduce memory I/O (not fewer FLOPs, but much faster in practice)
- **Grouped Query Attention (GQA)**: share K,V across multiple query heads — reduces memory for KV cache
- **Linear Attention**: approximates softmax attention in O(T·d²) (Performer, RWKV)

---

## Role as the Foundation of LLMs

The Transformer is not just used in LLMs — it **is** the LLM:

- Every major LLM (GPT, Claude, Gemini, LLaMA, Mistral) is a stack of Transformer decoder layers
- **Pre-training** = autoregressive next-token prediction loss, optimized by Adam
- **Fine-tuning / RLHF / RLAIF** = update Transformer weights based on human preferences
- **In-context learning** (few-shot prompting) emerges from attention patterns in the Transformer
- The **KV cache** exploited during inference is literally the cached Keys and Values from earlier attention computations
- **Chain-of-thought reasoning** emerges because the Transformer can route intermediate computations through its residual stream

Understanding the Transformer is the single most important algorithmic concept for understanding modern LLMs.
