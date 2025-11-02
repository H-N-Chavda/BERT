import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoder import TransformerEncoderLayer
from src.embeddings import Embeddings
from src.output_heads import MLMHead, NSPHead

class BertEncoder(nn.Module):
    def __init__(self, hidden_size=128, num_layers=4, num_heads=4, intermediate_size=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [B, L, H]
        attention_mask: [B, 1, 1, L] or None
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return self.norm(hidden_states)


class BertForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = Embeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob
        )
        self.encoder = BertEncoder(config)

        # Heads
        self.mlm_head = MLMHead(config, self.embeddings.token_embeddings.weight)
        self.nsp_head = NSPHead(config)

        # Loss functions
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mlm_labels=None, nsp_labels=None):
        """
        Forward pass for pretraining BERT.
        Returns dict containing losses and logits.
        """

        device = input_ids.device
        B, L = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros((B, L), dtype=torch.long, device=device)
        if attention_mask is None:
            attention_mask = torch.ones((B, L), dtype=torch.long, device=device)

        # Prepare attention mask for broadcasting
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]

        # Forward through encoder
        embeddings = self.embeddings(input_ids, token_type_ids)
        encoder_out = self.encoder(embeddings, extended_mask)

        # MLM logits
        mlm_logits = self.mlm_head(encoder_out)  # [B, L, V]

        # NSP logits — use [CLS] token embedding (first token)
        cls_output = encoder_out[:, 0]  # [B, H]
        nsp_logits = self.nsp_head(cls_output)  # [B, 2]

        # Compute losses if labels provided
        mlm_loss, nsp_loss, total_loss = None, None, None
        if mlm_labels is not None:
            mlm_loss = self.mlm_loss_fn(mlm_logits.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        if nsp_labels is not None:
            nsp_loss = self.nsp_loss_fn(nsp_logits.view(-1, 2), nsp_labels.view(-1))
        if mlm_loss is not None and nsp_loss is not None:
            total_loss = mlm_loss + nsp_loss
        elif mlm_loss is not None:
            total_loss = mlm_loss
        elif nsp_loss is not None:
            total_loss = nsp_loss

        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss,
            "nsp_loss": nsp_loss,
            "mlm_logits": mlm_logits,
            "nsp_logits": nsp_logits,
            "hidden_states": encoder_out,
        }