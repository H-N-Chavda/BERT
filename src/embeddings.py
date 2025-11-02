import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, max_position_embeddings=64, dropout_prob=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(2, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, token_type_ids):
        """
        input_ids: Tensor [B, L]
        token_type_ids: Tensor [B, L]
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        token_embed = self.token_embeddings(input_ids)
        position_embed = self.position_embeddings(position_ids)
        segment_embed = self.segment_embeddings(token_type_ids)

        embeddings = token_embed + position_embed + segment_embed
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # shape: [B, L, H]