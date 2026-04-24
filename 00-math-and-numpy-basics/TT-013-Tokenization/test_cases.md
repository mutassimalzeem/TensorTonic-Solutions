# Test Cases for TT-013 — Tokenization

## 1. Basic vocabulary build and encode
- Input:
  ```python
  tokenizer = SimpleTokenizer()
  tokenizer.build_vocab(["hello world"])
  tokenizer.encode("hello world")
  ```
- Expected output: `[4, 5]`

## 2. Case-insensitive encoding
- Input:
  ```python
  tokenizer = SimpleTokenizer()
  tokenizer.build_vocab(["hello world"])
  tokenizer.encode("Hello World")
  ```
- Expected output: `[4, 5]`

## 3. Unknown token handling
- Input:
  ```python
  tokenizer = SimpleTokenizer()
  tokenizer.build_vocab(["hello world"])
  tokenizer.encode("hello machine world")
  ```
- Expected output: `[4, 1, 5]`

## 4. Decoding token IDs
- Input:
  ```python
  tokenizer = SimpleTokenizer()
  tokenizer.build_vocab(["hello world"])
  tokenizer.decode([4, 1, 5])
  ```
- Expected output: `"hello <UNK> world"`

## 5. Unknown ID during decoding
- Input:
  ```python
  tokenizer = SimpleTokenizer()
  tokenizer.build_vocab(["hello world"])
  tokenizer.decode([999])
  ```
- Expected output: `"<UNK>"`

## 6. Encode before building vocabulary
- Input:
  ```python
  tokenizer = SimpleTokenizer()
  tokenizer.encode("hello world")
  ```
- Expected output: raises `ValueError("Vocab not built!")`
