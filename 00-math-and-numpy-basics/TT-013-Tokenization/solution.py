import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        self.unk_id = None
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        unique_words = set()
        
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)

        
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Remove special tokens from unique_words to prevent duplicates if they exist in text
        vocab = special_tokens + sorted(list(unique_words - set(special_tokens)))

        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        

        self.unk_id = self.word_to_id[self.unk_token]   #   setting up unk_id after the vocab is built
        
    def encode(self, texts: str) -> List[int]:
        if self.unk_id is None:
            raise ValueError("Vocab not built!")
            
        tokens = texts.lower().split()
        
        token_ids = [self.word_to_id.get(token, self.unk_id) for token in tokens]       # Use self.unk_id as the default integer
        return token_ids

    def decode(self, ids: List[int]) -> str:
        
        tokens_to_word = [self.id_to_word.get(index, self.unk_token) for index in ids]      # Use self.unk_token (the string) as the default, NOT the ID
        return " ".join(tokens_to_word)

# --- Quick Test ---
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()
    data = ["hello world"]
    
    tokenizer.build_vocab(data)
    
    # "machine" is not in vocab, so it will trigger <UNK>
    encoded = tokenizer.encode("Hello machine world")
    print(f"Encoded: {encoded}") 
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")