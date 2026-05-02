# Notes for TT-013 — Tokenization

- The tokenizer is word-based, not character-based or subword-based.
- In `solution.py`, all text is normalized with `.lower()` before tokenization.
- Token splitting is done with Python's default `.split()`, so it uses whitespace boundaries.
- The vocabulary always begins with the special tokens `<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>`.
- The remaining vocabulary words are sorted, which makes token IDs deterministic.
- `encode` cannot be called before `build_vocab`; otherwise it raises `ValueError("Vocab not built!")`.
- Unknown words are mapped to the integer ID of `<UNK>`.
- Unknown IDs are decoded back to the string `<UNK>`.
- Punctuation is not removed separately, so punctuation attached to a word becomes part of that token.
