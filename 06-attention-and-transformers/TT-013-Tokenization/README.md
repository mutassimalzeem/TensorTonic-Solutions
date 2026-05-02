# TT-013 — Tokenization

## Problem Link
TensorTonic: Tokenization

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement a simple **word-level tokenizer** that builds a vocabulary from input texts and supports encoding and decoding between words and integer token IDs.

The tokenizer lowercases text, splits on whitespace, reserves special tokens, and maps unknown words to `<UNK>`.

---

## Official Problem Statement
Create a basic tokenizer class with vocabulary building, encoding, and decoding. Unknown words should map to `<UNK>`, and unknown IDs should decode back to `<UNK>`.

---

## Constraints
- Input texts are processed with whitespace splitting
- Token matching is case-insensitive using `.lower()`
- Vocabulary includes special tokens first: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- Unknown words map to the ID of `<UNK>`
- Unknown IDs decode to the string `<UNK>`
- `encode` requires the vocabulary to be built first

---

## Example

### Input
```python
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(["hello world"])

encoded = tokenizer.encode("Hello machine world")
```

### Output
```python
[4, 1, 5]
```

### Decoded
```python
"hello <UNK> world"
```

---

## Solution Overview

The implementation in `solution.py`:

- collects unique lowercase words from all training texts with `build_vocab(texts)`
- prepends the special tokens before the sorted vocabulary words
- creates both `word_to_id` and `id_to_word` mappings
- stores the integer ID of `<UNK>` in `self.unk_id`
- uses `self.word_to_id.get(token, self.unk_id)` during encoding
- uses `self.id_to_word.get(index, self.unk_token)` during decoding

This approach keeps the tokenizer deterministic and easy to inspect.

---

## Usage

```python
from solution import SimpleTokenizer

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(["hello world"])

encoded = tokenizer.encode("Hello machine world")
print(encoded)

decoded = tokenizer.decode(encoded)
print(decoded)
```
