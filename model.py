import torch
import torch.nn as nn
import torch.nn.functional as F

class CBoW(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        embeddings = self.embedding(x)
        v = torch.mean(embeddings, 1)
        output = self.output_layer(v)

        return output
    