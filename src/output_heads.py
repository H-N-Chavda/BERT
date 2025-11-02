import torch
import torch.nn as nn
import torch.nn.functional as F

class MLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_weights, dropout=0.1):
        """
        embedding_weights: nn.Embedding weights for weight tying
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.dropout = nn.Dropout(dropout)

        # tie weights with embedding layer
        self.decoder.weight = embedding_weights

    def forward(self, hidden_states):
        """
        hidden_states: [B, L, H]
        """
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        logits = self.decoder(x) + self.bias
        return logits  # [B, L, V]
    

class NSPHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output):
        """
        pooled_output: [B, H]  — CLS token embedding
        returns: logits [B, 2]
        """
        return self.classifier(pooled_output)