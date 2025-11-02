from dataclasses import dataclass

@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 64
    pad_token_id: int = 0