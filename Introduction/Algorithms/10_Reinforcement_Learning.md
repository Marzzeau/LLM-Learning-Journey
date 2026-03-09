# Reinforcement Learning

## Table of Contents
1. [Overview](#overview)
2. [Core Framework — The MDP](#core-framework--the-mdp)
3. [Key Concepts](#key-concepts)
   - [Return and Discounting](#return-and-discounting)
   - [Value Functions](#value-functions)
   - [The Bellman Equation](#the-bellman-equation)
   - [The Exploration-Exploitation Dilemma](#the-exploration-exploitation-dilemma)
4. [Q-Learning](#q-learning)
5. [Deep Q-Network (DQN)](#deep-q-network-dqn)
6. [Policy Gradient Methods](#policy-gradient-methods)
   - [REINFORCE Algorithm](#reinforce-algorithm)
   - [Actor-Critic Methods](#actor-critic-methods)
7. [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
8. [Model-Based vs. Model-Free RL](#model-based-vs-model-free-rl)
9. [Reinforcement Learning from Human Feedback (RLHF)](#reinforcement-learning-from-human-feedback-rlhf)
10. [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
11. [Strengths and Weaknesses](#strengths-and-weaknesses)
12. [Role in ML and LLMs](#role-in-ml-and-llms)

---

## Overview

Reinforcement Learning (RL) is a paradigm in which an **agent** learns to make decisions by interacting with an **environment**, receiving **rewards** for actions, and discovering the optimal **policy** — the strategy for choosing actions that maximizes cumulative reward.

Unlike supervised learning (learn from labeled examples) or unsupervised learning (learn structure from data), RL learns from **trial and error** with **delayed rewards**. The agent has no teacher — only a reward signal that may come long after the actions that caused it.

RL powers:
- Game-playing AI (AlphaGo, AlphaZero, OpenAI Five)
- Robotics and control systems
- Recommendation systems
- Most critically: **alignment of LLMs with human preferences (RLHF)**

---

## Core Framework — The MDP

The standard mathematical formalism for RL is the **Markov Decision Process (MDP)**:

```
MDP = (S, A, P, R, γ)
```

| Symbol | Name | Description |
|---|---|---|
| `S` | State space | All possible states of the environment |
| `A` | Action space | All possible actions the agent can take |
| `P(s'|s,a)` | Transition model | Probability of reaching state `s'` from `s` after action `a` |
| `R(s,a,s')` | Reward function | Immediate reward signal |
| `γ ∈ [0,1)` | Discount factor | How much to value future rewards |

**The Markov Property:** The future is conditionally independent of the past given the present state:
```
P(sₜ₊₁ | s₁,...,sₜ, a₁,...,aₜ) = P(sₜ₊₁ | sₜ, aₜ)
```

This assumption means the current state contains all information needed for optimal decision-making.

**Policy:** A policy `π` maps states to actions:
- Deterministic: `a = π(s)`
- Stochastic: `a ~ π(a|s)` — probability distribution over actions

---

## Key Concepts

### Return and Discounting

The **return** `Gₜ` is the total discounted reward from time step `t`:

```
Gₜ = rₜ + γ*r_{t+1} + γ²*r_{t+2} + ... = Σₖ₌₀^∞ γᵏ * r_{t+k}
```

The discount factor `γ`:
- `γ = 0`: only immediate reward matters (myopic)
- `γ = 1`: all future rewards equal (infinite horizon; requires careful handling)
- `γ = 0.99`: typical value — future rewards matter but exponentially less

Discounting ensures mathematical convergence and reflects the reality that near-term rewards are more certain.

### Value Functions

Value functions estimate "how good" it is to be in a state (or take an action in a state):

**State Value Function:**
```
Vπ(s) = E_π[Gₜ | Sₜ = s]
```
Expected return starting from state `s`, following policy `π`.

**Action-Value Function (Q-function):**
```
Qπ(s, a) = E_π[Gₜ | Sₜ = s, Aₜ = a]
```
Expected return starting from state `s`, taking action `a`, then following policy `π`.

The **optimal policy** greedily selects the action that maximizes Q:
```
π*(s) = argmax_a Q*(s, a)
```

### The Bellman Equation

The Bellman equation expresses value as a recursive relationship:

```
Vπ(s) = Σₐ π(a|s) * Σₛ' P(s'|s,a) * [R(s,a,s') + γ * Vπ(s')]
```

For the optimal value function:
```
V*(s) = max_a Σₛ' P(s'|s,a) * [R(s,a,s') + γ * V*(s')]
Q*(s,a) = Σₛ' P(s'|s,a) * [R(s,a,s') + γ * max_a' Q*(s',a')]
```

This bootstrapping property — defining the value of a state in terms of the values of successor states — is the foundation of all RL algorithms.

### The Exploration-Exploitation Dilemma

The agent must balance:
- **Exploitation**: choosing actions known to yield high reward (greedy)
- **Exploration**: trying new actions to discover potentially better rewards

Strategies:
- **ε-greedy**: with probability `ε` choose a random action, else choose the best known
- **Softmax/Boltzmann**: choose action `a` with probability `∝ exp(Q(s,a)/τ)`, temperature `τ` controls randomness
- **Upper Confidence Bound (UCB)**: prefer actions that are both high-reward and under-explored
- **Thompson Sampling**: sample from the posterior distribution of reward estimates

---

## Q-Learning

Q-Learning (Watkins, 1989) is a **model-free**, **off-policy** algorithm that directly estimates `Q*(s,a)` via temporal difference updates.

**Temporal Difference (TD) Error:**
```
δₜ = rₜ + γ * max_a' Q(sₜ₊₁, a') - Q(sₜ, aₜ)
```

**Q-Learning Update:**
```
Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α * δₜ
```

Where `α` is the learning rate.

The key insight: the **target** `rₜ + γ * max_a' Q(sₜ₊₁, a')` is a bootstrapped estimate of the true Q-value. By repeatedly applying this update, Q converges to Q* (guaranteed for tabular MDPs with appropriate learning rates and sufficient exploration).

**On-policy vs Off-policy:**
- **On-policy** (SARSA): update uses `a'` sampled from current policy `π`
- **Off-policy** (Q-Learning): update uses `max_a'` regardless of what policy actually chose

---

## Deep Q-Network (DQN)

Q-Learning with a table works only for small, discrete state spaces. **DQN** (Mnih et al., 2015) replaces the table with a deep neural network `Q(s,a;θ)`:

```
L(θ) = E[(r + γ * max_a' Q(s',a';θ⁻) - Q(s,a;θ))²]
```

Two critical innovations:
1. **Experience Replay Buffer**: store transitions `(s,a,r,s')` in a buffer; sample random mini-batches for training. Breaks temporal correlations and improves data efficiency.

2. **Target Network**: use a separate, slowly-updated copy `θ⁻` for computing TD targets. Prevents instability from constantly-moving targets.

DQN achieved human-level performance on 49 Atari games from raw pixels — a landmark in deep RL.

---

## Policy Gradient Methods

Instead of learning a value function and deriving a policy, **policy gradient** methods directly optimize a parameterized policy `π(a|s;θ)`.

**Objective:**
```
J(θ) = E_π[Σₜ rₜ] = E_π[Gₜ]
```

**Policy Gradient Theorem:**
```
∇_θ J(θ) = E_π[Gₜ * ∇_θ log π(aₜ|sₜ;θ)]
```

Intuition: increase the probability of actions that led to high returns; decrease the probability of actions that led to low returns.

### REINFORCE Algorithm

```
1. Collect full trajectory τ = (s₀,a₀,r₀,...,sₜ,aₜ,rₜ) under current π
2. Compute returns Gₜ for each step
3. Update: θ ← θ + α * Σₜ Gₜ * ∇_θ log π(aₜ|sₜ;θ)
```

Problem: very high variance (return Gₜ depends on entire trajectory). Solution: subtract a **baseline** `b(s)` (typically V(s)):

```
∇_θ J(θ) = E_π[(Gₜ - b(sₜ)) * ∇_θ log π(aₜ|sₜ;θ)]
```

`Gₜ - V(sₜ)` is called the **advantage** — how much better action `aₜ` was than average.

### Actor-Critic Methods

Combine policy gradient (actor) with value function estimation (critic):

```
Actor:  π(a|s;θ) — learns which actions to take
Critic: V(s;w)   — estimates how good states are

TD error: δₜ = rₜ + γ*V(sₜ₊₁;w) - V(sₜ;w)   ≈ Advantage

Actor update:  θ ← θ + α_θ * δₜ * ∇_θ log π(aₜ|sₜ;θ)
Critic update: w ← w + α_w * δₜ * ∇_w V(sₜ;w)
```

The critic reduces variance by providing a baseline and enabling online updates (no need to wait for episode end).

---

## Proximal Policy Optimization (PPO)

PPO (Schulman et al., 2017) is the most widely used policy gradient algorithm and the core of RLHF for LLMs.

**Problem with naive policy gradient**: large updates can collapse performance — moving too far from the current policy in one step leads to instability.

**Solution**: clip the policy ratio to restrict how much the policy can change in one update:

```
ratio = π(a|s;θ) / π(a|s;θ_old)   (new policy / old policy)

L_CLIP(θ) = E_t[min(rₜ(θ) * Âₜ, clip(rₜ(θ), 1-ε, 1+ε) * Âₜ)]
```

Where `Âₜ` is the estimated advantage and `ε` (typically 0.2) limits the ratio.

**Intuition:**
- If advantage is positive (action was good): increase probability, but not more than `1+ε` times
- If advantage is negative (action was bad): decrease probability, but not less than `1-ε` times

PPO is:
- Simple to implement
- Robust and stable
- Sample-efficient
- Works across a wide range of tasks

---

## Model-Based vs. Model-Free RL

| Approach | Description | Pros | Cons | Examples |
|---|---|---|---|---|
| Model-Free | Learn value/policy directly from experience | Simple; works without environment model | Sample-inefficient | Q-Learning, PPO, SAC |
| Model-Based | Learn a world model `P(s'|s,a)`, plan using it | Sample-efficient; allows planning | Hard to learn accurate models | AlphaZero, MuZero, Dreamer |

**AlphaZero**: uses Monte Carlo Tree Search (MCTS) with a learned value network and policy network. No hand-crafted heuristics — learns superhuman chess, shogi, and Go from self-play.

---

## Reinforcement Learning from Human Feedback (RLHF)

RLHF is the primary technique for aligning LLMs with human values and making them helpful, harmless, and honest.

**Three-stage process:**

### Stage 1: Supervised Fine-Tuning (SFT)
Fine-tune the pre-trained LLM on high-quality human-written demonstrations:
```
LLM_SFT = fine-tune(LLM_base, human_demonstrations)
```

### Stage 2: Reward Model Training
Collect human preference data: show pairs of model outputs `(y₁, y₂)` and ask which is better.

Train a **reward model** `R(x, y)` (a LLM with a scalar output head) to predict human preferences:
```
L_RM = -E[(log σ(R(x, y_w) - R(x, y_l)))]
```

Where `y_w` is the preferred ("win") output and `y_l` is the less preferred ("lose") output.

### Stage 3: RL Fine-Tuning with PPO
Optimize the LLM policy to maximize the reward model's score, with a KL penalty to prevent drift from the SFT model:

```
objective = E_{x~D, y~π}[R(x,y)] - β * KL[π(y|x) || π_SFT(y|x)]
```

The KL penalty prevents reward hacking — the LLM gaming the reward model with outputs that score high but are actually bad.

---

## Direct Preference Optimization (DPO)

DPO (Rafailov et al., 2023) is an elegant alternative to RLHF that **eliminates the need for a separate reward model and RL training**.

It shows that the optimal policy for the RLHF objective has a closed form:
```
π*(y|x) ∝ π_SFT(y|x) * exp(R(x,y)/β)
```

Which can be rearranged so the reward model is implicitly defined by the policy ratio:
```
R(x,y) = β * log(π*(y|x)/π_SFT(y|x)) + const
```

Substituting into the preference loss:
```
L_DPO = -E[(log σ(β * log π_θ(y_w|x)/π_SFT(y_w|x) - β * log π_θ(y_l|x)/π_SFT(y_l|x)))]
```

This is just a classification loss — no RL needed! DPO:
- Is simpler to implement
- Is more stable to train
- Often matches or exceeds PPO performance
- Is now widely used (Zephyr, Tulu 2, Llama 3 instruct)

---

## Strengths and Weaknesses

**Strengths:**
- Can optimize for complex, non-differentiable objectives (human preferences, game scores)
- Learns from interaction — does not require labeled data
- Can discover superhuman strategies through self-play

**Weaknesses:**
- **Sample inefficiency**: typically needs millions of interactions
- **Reward hacking**: agents find unintended ways to maximize reward
- **Credit assignment**: hard to identify which actions caused a delayed reward
- **Stability**: training can be highly unstable
- **Sparse rewards**: many real tasks provide rewards rarely or only at the end

---

## Role in ML and LLMs

RL is critical to the final mile of LLM training:

- **RLHF / PPO**: Used to train ChatGPT, Claude, Gemini — makes models actually helpful
- **DPO**: The simpler RLHF replacement dominating open-source fine-tuning
- **RLAIF**: RL from AI Feedback — uses another LLM as the reward model (Constitutional AI / Claude)
- **DeepSeek-R1**: Used RL with sparse rewards (correctness) to train chain-of-thought reasoning, achieving o1-level performance
- **AlphaCode**: RL-based code generation with unit test rewards
- **Self-play RL**: Used in safety research to find adversarial prompts (red-teaming)
- **Reward modeling** remains an open research challenge — a key bottleneck in alignment is specifying what humans actually want
