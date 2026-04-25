<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=220&color=0:0f172a,30:0b3b66,65:0ea5e9,100:22c55e&text=TensorTonic%20Solutions&fontColor=ffffff&fontAlignY=38&desc=Building%20ML%20intuition%20from%20scratch%20with%20code%2C%20notes%2C%20and%20experiments&descAlignY=58&animation=fadeIn" alt="TensorTonic Solutions banner" />

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=2800&pause=900&color=38BDF8&center=true&vCenter=true&width=900&lines=From+math+basics+to+ML+building+blocks;TensorTonic+problems+solved+with+clean+Python;Each+exercise+includes+docs%2C+notes%2C+and+test+cases)](https://github.com/)

<p>
  <img src="https://img.shields.io/badge/Focus-Machine%20Learning%20Foundations-0f172a?style=for-the-badge&logo=tensorflow&logoColor=white" alt="ML Foundations" />
  <img src="https://img.shields.io/badge/Language-Python-1d4ed8?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python" />
  <img src="https://img.shields.io/badge/Style-Learn%20by%20Building-059669?style=for-the-badge&logo=bookstack&logoColor=white" alt="Learn by Building" />
</p>

</div>

## Overview

This repository collects my **TensorTonic** solutions for core machine learning and NumPy-based exercises.

The goal is not just to store answers. It is to build a compact learning archive where each problem includes:

- a working `solution.py`
- a problem-focused `README.md`
- a short `explanation.md`
- quick-reference notes
- test case examples for revision

If you are learning ML fundamentals from first principles, this repo is designed to feel more like a **study lab** than a plain answer dump.

---

## Learning Flow

<p align="center">
  <img src="https://progress-bar.xyz/100?title=math%20%26%20numpy%20basics&width=700&color=38bdf8" alt="Math and NumPy basics progress" />
</p>

```text
Math Basics -> NumPy Operations -> Optimization -> Sequence Models -> Transformer Foundations
```

Current solved sequence in this repo:

1. `TT-001` Sigmoid
2. `TT-002` Mean Squared Error
3. `TT-003` Softmax
4. `TT-004` Cross Entropy
5. `TT-005` Gradient Descent
6. `TT-006` Adam
7. `TT-007` Matrix Transposition
8. `TT-008` Dot Product
9. `TT-009` Linear Regression Closed Form
10. `TT-010` Batch Normalization Forward
11. `TT-011` RNN Step Forward
12. `TT-012` Euclidean Distance
13. `TT-013` Tokenization
14. `TT-014` Embedding

---

## Repository Map

```text
Tensortonic Solutions/
|
+-- 00-math-and-numpy-basics/
|   |
|   +-- TT-001-sigmoid-numpy/
|   +-- TT-002-MSE-numpy/
|   +-- TT-003-Softmax/
|   +-- TT-004-Cross-Entropy/
|   +-- TT-005-Gradien-Descent/
|   +-- TT-006-Adam/
|   +-- TT-007-Matrix_Transposition/
|   +-- TT-008-Dot-Product/
|   +-- TT-009-Linear-Regression-Closed-Form/
|   +-- TT-010-Batch-Normalization/
|   +-- TT-011-RNN-Step-Forward/
|   +-- TT-012-Euclidean-Distance/
|   +-- TT-013-Tokenization/
|   +-- TT-014-Embedding/
|
+-- templates/
+-- roadmap.md
```

---

## Solved Problems

| ID | Problem | Concepts | Folder |
|---|---|---|---|
| `TT-001` | Sigmoid | activation function, NumPy basics | `00-math-and-numpy-basics/TT-001-sigmoid-numpy` |
| `TT-002` | MSE | regression loss, averaging, arrays | `00-math-and-numpy-basics/TT-002-MSE-numpy` |
| `TT-003` | Softmax | normalization, stability, exponentials | `00-math-and-numpy-basics/TT-003-Softmax` |
| `TT-004` | Cross Entropy | classification loss, log probabilities | `00-math-and-numpy-basics/TT-004-Cross-Entropy` |
| `TT-005` | Gradient Descent | derivatives, updates, optimization | `00-math-and-numpy-basics/TT-005-Gradien-Descent` |
| `TT-006` | Adam | momentum, adaptive learning rates, bias correction | `00-math-and-numpy-basics/TT-006-Adam` |
| `TT-007` | Matrix Transposition | matrix operations, axes, NumPy | `00-math-and-numpy-basics/TT-007-Matrix_Transposition` |
| `TT-008` | Dot Product | linear algebra, vector multiplication | `00-math-and-numpy-basics/TT-008-Dot-Product` |
| `TT-009` | Linear Regression Closed Form | normal equation, regression, matrix inversion | `00-math-and-numpy-basics/TT-009-Linear-Regression-Closed-Form` |
| `TT-010` | Batch Normalization Forward | normalization, feature scaling, neural nets | `00-math-and-numpy-basics/TT-010-Batch-Normalization` |
| `TT-011` | RNN Step Forward | recurrent networks, hidden state update | `00-math-and-numpy-basics/TT-011-RNN-Step-Forward` |
| `TT-012` | Euclidean Distance | vector distance, L2 norm | `00-math-and-numpy-basics/TT-012-Euclidean-Distance` |
| `TT-013` | Tokenization | NLP basics, vocabulary mapping, encoding | `00-math-and-numpy-basics/TT-013-Tokenization` |
| `TT-014` | Embedding | token vectors, PyTorch, transformer basics | `00-math-and-numpy-basics/TT-014-Embedding` |

---

## What Makes This Repo Useful

<table>
  <tr>
    <td width="33%">
      <h3>Readable Solutions</h3>
      <p>Problems are solved with focused Python implementations instead of bloated notebooks or framework-heavy code.</p>
    </td>
    <td width="33%">
      <h3>Revision Friendly</h3>
      <p>Each folder is structured so a concept can be reviewed quickly before interviews, classes, or practice sessions.</p>
    </td>
    <td width="33%">
      <h3>Concept First</h3>
      <p>The repository emphasizes intuition: what the formula means, why the code works, and where the result is used in ML.</p>
    </td>
  </tr>
</table>

---

## Example Folder Structure

Inside a typical problem directory, you will usually find:

```text
TT-xxx-problem-name/
|
+-- solution.py
+-- README.md
+-- explanation.md
+-- note.md / notes.md
+-- test_cases.md
```

This makes each exercise self-contained and easy to revisit later.

---

## Tech Stack

```python
{
    "language": "Python",
    "numerical_computing": ["NumPy", "PyTorch"],
    "focus": [
        "Machine Learning fundamentals",
        "Mathematical intuition",
        "Optimization basics",
        "Vectorized computation",
        "Sequence modeling foundations",
        "Transformer building blocks"
    ]
}
```

---

## Why TensorTonic

TensorTonic is a great environment for practicing the mechanics behind machine learning systems.

Instead of hiding everything behind high-level libraries, these exercises push you to implement core ideas yourself:

- activation functions
- loss functions
- probability transformations
- optimization steps
- array-based reasoning
- sequence modeling basics
- transformer input representations

That is where a lot of real intuition gets built.

---

## Roadmap

The broader learning direction for this repo lives in [`roadmap.md`](./roadmap.md).

The intention is to keep expanding this repository in a clean progression:

- stronger math and NumPy foundations
- classical ML building blocks
- deeper optimization concepts
- neural network components
- more complete end-to-end implementations

---

## Quick Start

```bash
git clone <your-repo-url>
cd "Tensortonic Solutions"
```

Then open any exercise folder and run or inspect `solution.py`.

---

## Final Note

This project is part practice set, part knowledge base, and part progress tracker.

If you are also learning machine learning from scratch, I hope this repo gives you a clean and motivating example of how to document your journey while improving your implementation skills.

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&height=120&color=0:0ea5e9,100:22c55e&text=Keep%20Building%20The%20Math&fontColor=ffffff&fontSize=28&animation=twinkling" alt="Footer banner" />
</div>
