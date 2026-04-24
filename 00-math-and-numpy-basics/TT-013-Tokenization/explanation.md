# Explanation for TT-013 — Tokenization

The tokenizer in `solution.py` is a simple word-level tokenizer implemented as a Python class.

It performs three main tasks:

1. Build a vocabulary from training texts.
2. Encode text into integer token IDs.
3. Decode token IDs back into text.

## How vocabulary building works

The `build_vocab` method:

1. Iterates through each text in the input list.
2. Lowercases the text and splits it using `.split()`.
3. Collects all unique words into a set.
4. Prepends the special tokens `<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>`.
5. Sorts the remaining vocabulary words so the mapping is deterministic.
6. Builds both `word_to_id` and `id_to_word`.
7. Stores the ID of `<UNK>` in `self.unk_id`.

## How encoding works

The `encode` method:

1. Checks whether the vocabulary has already been built.
2. Lowercases the input text and splits it by whitespace.
3. Converts each token to its integer ID.
4. Uses `self.unk_id` when a token is not found in the vocabulary.

This means unseen words are safely mapped to the `<UNK>` token.

## How decoding works

The `decode` method:

1. Takes a list of integer IDs.
2. Looks up each ID in `id_to_word`.
3. Uses `<UNK>` if an ID is missing.
4. Joins the decoded tokens back into a single string separated by spaces.

## Notes on behavior

- The tokenizer is case-insensitive because it lowercases all text.
- Tokenization is based only on whitespace splitting.
- The vocabulary order is deterministic because non-special words are sorted.
- Unknown words and invalid IDs both fall back to `<UNK>`.
