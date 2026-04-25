import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    embedding_layer = nn.Embedding(num_embeddings= vocab_size, embedding_dim= d_model)
    return embedding_layer 

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    
    embedded = embedding(tokens)

    return embedded * math.sqrt(d_model)

create_embedding_layer(vocab_size = 10, d_model = 4)