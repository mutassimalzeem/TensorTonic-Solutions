# TensorTonic Solutions Roadmap

A structured learning roadmap for solving TensorTonic problems and documenting them in this repository.

> This roadmap is designed to match the repository structure:
>
> - `00-math-and-numpy-basics`
> - `01-core-ml`
> - `02-optimization`
> - `03-losses-and-activations`
> - `04-neural-networks`
> - `05-sequence-models`
> - `06-attention-and-transformers`
> - `07-advanced`

---

## Purpose of This Roadmap

The goal of this repository is not just to solve problems, but to build strong intuition step by step.

Each problem should help improve one or more of these skills:

- mathematical intuition
- NumPy and tensor manipulation
- machine learning fundamentals
- optimization understanding
- neural network building blocks
- sequence modeling
- attention and transformers
- implementation confidence
- technical documentation for GitHub

This roadmap follows a beginner-to-advanced progression based on publicly visible TensorTonic examples and modules, including problems such as **Sigmoid (NumPy)**, **Softmax Function**, **Cross Entropy Loss**, **Linear Regression**, **Gradient Descent**, **Adam Optimizer**, **Batch Normalization**, **RNN forward pass**, and transformer-related modules such as **Tokenization**, **Embeddings**, **Positional Encoding**, **Scaled Dot-Product Attention**, **Multi-Head Attention**, **Layer Normalization**, **Feed Forward**, and **Output Projection**. citeturn0search0

---

## Recommended Solve Order

Follow this order strictly for the best learning curve:

```text
00-math-and-numpy-basics
→ 01-core-ml
→ 02-optimization
→ 03-losses-and-activations
→ 04-neural-networks
→ 05-sequence-models
→ 06-attention-and-transformers
→ 07-advanced
```

Why this order works:

1. First, learn array operations and vectorized thinking.
2. Then, understand basic machine learning models.
3. After that, learn how models are optimized.
4. Then connect activations and losses.
5. Then build neural network intuition.
6. Then move to sequence data.
7. Then learn attention and transformers.
8. Finally, solve mixed and advanced implementation problems.

---

# 00 — Math and NumPy Basics

**Folder:** `00-math-and-numpy-basics/`

## Goal

Build the foundation for all later ML and deep learning work.

## What to learn here

- array creation
- shapes and dimensions
- broadcasting
- elementwise operations
- reductions like sum, mean, max
- vectorized mathematical functions
- writing clean NumPy-based solutions

## Problem roadmap

- [ ] TT-001 — Sigmoid (NumPy)
- [ ] Softmax Function
- [ ] Array shape manipulation problems
- [ ] Broadcasting practice problems
- [ ] Elementwise math problems
- [ ] Reduction operation problems
- [ ] Normalization-style problems

## Focus points

- understand input and output shapes clearly
- avoid unnecessary loops
- write vectorized solutions where possible
- explain every mathematical step in simple language

## Repo outcome

After finishing this section, it should be easy to read and write NumPy-based tensor operations.

---

# 01 — Core ML

**Folder:** `01-core-ml/`

## Goal

Understand the basic machine learning workflow from raw input to prediction.

## What to learn here

- how linear models work
- how parameters affect outputs
- how error is measured
- how gradients help update parameters
- difference between regression and classification

## Problem roadmap

- [ ] Linear Regression
- [ ] Prediction function implementation
- [ ] Cost / error computation
- [ ] Gradient Descent
- [ ] Logistic Regression
- [ ] Threshold-based prediction
- [ ] Data preprocessing mini problems
- [ ] Evaluation metric basics

## Focus points

- understand the model equation clearly
- connect prediction, error, and parameter updates
- explain the intuition before writing the implementation

## Repo outcome

This section should make classical ML feel less like formulas and more like a full pipeline.

---

# 02 — Optimization

**Folder:** `02-optimization/`

## Goal

Understand how models learn.

## What to learn here

- parameter update logic
- effect of learning rate
- why optimization can be unstable
- why some optimizers converge faster than others

## Problem roadmap

- [ ] Gradient Descent
- [ ] Momentum
- [ ] RMSProp
- [ ] Adam Optimizer
- [ ] Learning-rate comparison tasks
- [ ] Convergence intuition tasks
- [ ] Optimization notes and visual explanations

## Focus points

- know what is being updated and why
- connect gradient direction to loss reduction
- understand the role of moving averages in Adam-like optimizers

## Repo outcome

This section should make training dynamics much easier to understand later in neural networks.

---

# 03 — Losses and Activations

**Folder:** `03-losses-and-activations/`

## Goal

Understand how outputs become probabilities, and how models know whether they are wrong.

## What to learn here

- activation functions
- binary vs multiclass outputs
- error measurement in regression and classification
- why some activations and losses are paired together

## Problem roadmap

- [ ] Sigmoid
- [ ] Softmax Function
- [ ] Mean Squared Error
- [ ] Binary Cross Entropy
- [ ] Cross Entropy Loss
- [ ] Activation-loss pairing notes
- [ ] Numerical stability notes

## Focus points

- know when to use sigmoid and when to use softmax
- understand why MSE is common in regression
- understand why cross entropy is common in classification
- explain loss values as feedback signals

## Repo outcome

This section should build strong intuition for output layers and learning objectives.

---

# 04 — Neural Networks

**Folder:** `04-neural-networks/`

## Goal

Move from single equations to learnable layered systems.

## What to learn here

- neurons and weighted sums
- dense layers
- forward pass logic
- backpropagation intuition
- normalization inside networks
- how deeper models learn richer patterns

## Problem roadmap

- [ ] Single neuron forward pass
- [ ] Dense / fully connected layer
- [ ] Backprop basics
- [ ] Multi-layer forward pass
- [ ] Weight initialization basics
- [ ] Batch Normalization
- [ ] Small MLP implementation

## Focus points

- understand each layer as a transformation
- track tensor shapes through every step
- explain how gradients move backward
- connect normalization with stable training

## Repo outcome

This section should help build the mindset needed for implementing full neural networks from scratch.

---

# 05 — Sequence Models

**Folder:** `05-sequence-models/`

## Goal

Understand how models process ordered information.

## What to learn here

- sequence input formatting
- hidden states
- recurrence
- time-step-based updates
- why sequence modeling is different from normal feedforward models

## Problem roadmap

- [ ] Sequence input formatting
- [ ] RNN forward pass
- [ ] Hidden state update tasks
- [ ] Multi-step sequence processing
- [ ] Sequence output generation
- [ ] RNN limitations notes
- [ ] Optional LSTM / GRU foundation notes

## Focus points

- understand why word order matters
- know how the hidden state carries information
- explain recurrence with small manual examples
- understand why RNNs struggle with long-range dependencies

## Repo outcome

This section should create a bridge between standard neural networks and transformer models.

---

# 06 — Attention and Transformers

**Folder:** `06-attention-and-transformers/`

## Goal

Learn the full pipeline from text tokens to transformer outputs.

## What to learn here

- tokenization
- embeddings
- positional encoding
- attention mechanism
- multi-head attention
- normalization and feed-forward layers
- assembling a transformer block

## Problem roadmap

- [ ] Tokenization
- [ ] Embeddings
- [ ] Positional Encoding
- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Attention
- [ ] Layer Normalization
- [ ] Feed Forward
- [ ] Output Projection
- [ ] Mini transformer block assembly

## Focus points

- understand one word → one id → one vector
- understand why positional information is needed
- understand attention scores and weighted sums
- understand how multiple heads capture different relationships
- track tensor shapes at every step

## Repo outcome

This section should make the transformer architecture feel implementable instead of mysterious.

TensorTonic’s public research track around **Attention Is All You Need** highlights modules such as **Tokenization**, **Embeddings**, **Positional Encoding**, **Multi-Head Attention**, **Layer Normalization**, **Feed Forward**, and **Output Projection**, which align directly with this section structure. citeturn0search0

---

# 07 — Advanced

**Folder:** `07-advanced/`

## Goal

Solve harder problems that combine multiple ideas at once.

## What to learn here

- integration of multiple modules
- implementation maturity
- debugging harder models
- research-paper-inspired builds
- architecture-level understanding

## Problem roadmap

- [ ] Full training loop tasks
- [ ] Hard optimization/debugging problems
- [ ] Transformer assembly tasks
- [ ] Research-style implementation tasks
- [ ] Mixed hard problems from multiple categories
- [ ] Comparative notes between architectures
- [ ] Personal deep-dive experiments

## Focus points

- solve without depending too much on high-level libraries
- explain tradeoffs clearly
- document mistakes and lessons learned
- focus on implementation depth, not only correctness

## Repo outcome

This section should represent the transition from learner to confident builder.

---

# Priority Problem List

These are the most important publicly visible TensorTonic topics to prioritize first:

- [ ] TT-001 — Sigmoid (NumPy)
- [ ] Softmax Function
- [ ] Cross Entropy Loss
- [ ] Linear Regression
- [ ] Gradient Descent
- [ ] Adam Optimizer
- [ ] Batch Normalization
- [ ] RNN forward pass
- [ ] Tokenization
- [ ] Embeddings
- [ ] Positional Encoding
- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Attention
- [ ] Layer Normalization
- [ ] Feed Forward
- [ ] Output Projection

These topics are visible across TensorTonic’s public homepage and research pages. citeturn0search0

---

# How to Place Future Problems into the Repo

Whenever a new TensorTonic problem is solved, place it using this rule:

## Put the problem in `00-math-and-numpy-basics` if it is about

- NumPy
- arrays
- broadcasting
- shapes
- elementwise math
- basic tensor operations

## Put the problem in `01-core-ml` if it is about

- regression
- basic classification
- preprocessing
- metrics
- core model equations

## Put the problem in `02-optimization` if it is about

- gradient updates
- optimizers
- learning rates
- convergence
- training dynamics

## Put the problem in `03-losses-and-activations` if it is about

- activation functions
- probability outputs
- regression/classification loss functions
- numerical stability in output/loss calculations

## Put the problem in `04-neural-networks` if it is about

- neurons
- dense layers
- backpropagation
- normalization
- MLP-style networks

## Put the problem in `05-sequence-models` if it is about

- RNNs
- LSTMs
- GRUs
- hidden states
- ordered sequence processing

## Put the problem in `06-attention-and-transformers` if it is about

- tokenization
- embeddings
- attention
- transformer blocks
- positional encoding
- encoder/decoder ideas

## Put the problem in `07-advanced` if it is about

- full architecture assembly
- paper implementation
- combined hard concepts
- difficult debugging or integration

---

# Recommended Documentation Flow for Each Problem

For every solved problem, keep the same file structure:

```text
TT-xxx-problem-name/
├── README.md
├── solution.py
├── notes.md
├── test_cases.md
└── explanation.md
```

## Suggested writing flow

### `README.md`
Include:
- problem summary
- category
- difficulty
- key concepts
- approach overview
- final learning takeaway

### `solution.py`
Include:
- clean implementation
- comments only where useful
- readable variable names
- no unnecessary complexity

### `notes.md`
Include:
- formulas
- intuition
- edge cases
- mistakes made
- improvements for later

### `test_cases.md`
Include:
- sample inputs
- expected outputs
- edge case tests
- shape verification where needed

### `explanation.md`
Include:
- step-by-step reasoning
- why the approach works
- time complexity if relevant
- shape flow if relevant

---

# Suggested Milestones

## Milestone 1 — Foundations
Complete:
- `00-math-and-numpy-basics`
- basic parts of `01-core-ml`
- basic parts of `03-losses-and-activations`

Outcome:
- comfortable with vectorized ML math

## Milestone 2 — Learning Mechanics
Complete:
- remaining `01-core-ml`
- all of `02-optimization`
- initial `04-neural-networks`

Outcome:
- clear understanding of how models train

## Milestone 3 — Deep Learning Core
Complete:
- remaining `04-neural-networks`
- all of `05-sequence-models`

Outcome:
- can explain feedforward and recurrent processing

## Milestone 4 — Transformer Readiness
Complete:
- all of `06-attention-and-transformers`

Outcome:
- can build transformer components with confidence

## Milestone 5 — Advanced Builder Stage
Complete:
- `07-advanced`
- personal experiments and harder reconstructions

Outcome:
- able to implement and document advanced deep learning systems clearly

---

# Final Notes

This roadmap is intentionally structured as a learning path, not just a random list of problems.

The main idea is:

- start small
- stay consistent
- document every solution properly
- focus on understanding, not just passing the problem
- turn the repository into proof of growth

As TensorTonic expands its catalog, new problems can be inserted into the same folder system without breaking the structure of the repository. TensorTonic publicly describes its platform as a collection of **200+ ML problems**, so treating this roadmap as a stable category-based solve path is safer than assuming a fixed public master list of all titles. citeturn0search0
