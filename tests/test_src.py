import torch
import pytest
from src.tokenizer import Tokenizer
from src.embeddings import Embeddings


def test_tokenizer():
    corpus = "hello world hello"
    tokenizer = Tokenizer(vocab_size=20)
    tokenizer.train(corpus)

    # check vocab & reverse vocab
    assert len(tokenizer.vocab) > 0
    assert all(tok in tokenizer.vocab for tok in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    assert isinstance(tokenizer.inv_vocab, dict)

    # internal merges & stats
    pairs = tokenizer._get_stats({"h e l l o </w>": 2})
    assert isinstance(pairs, dict) and len(pairs) > 0
    merged = tokenizer._merge_vocab(("l", "l"), {"h e l l o </w>": 2})
    assert isinstance(merged, dict)

    # tokenize & encode word
    tokens = tokenizer.tokenize("hello world")
    assert isinstance(tokens, list) and len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)

    unk = tokenizer._encode_word("zzzzz")
    assert "[UNK]" in unk

    # encode single and paired
    out_single = tokenizer.encode("hello world")
    out_pair = tokenizer.encode("hello", "world")

    for out in [out_single, out_pair]:
        assert set(out.keys()) == {"input_ids", "token_type_ids", "attention_mask"}
        assert all(isinstance(v, torch.Tensor) for v in out.values())
        assert all(len(v) == 64 for v in out.values())

    # truncation check
    out_short = tokenizer.encode("hello " * 100, max_len=10)
    assert all(len(v) == 10 for v in out_short.values())


def test_embeddings():
    vocab_size = 100
    hidden_size = 32
    seq_len = 10
    batch_size = 2

    model = Embeddings(vocab_size=vocab_size, hidden_size=hidden_size, max_position_embeddings=64, dropout_prob=0.2)

    # random integer inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    token_type_ids = torch.randint(0, 2, (batch_size, seq_len))

    # forward pass
    output = model(input_ids, token_type_ids)

    # check shape
    assert output.shape == (batch_size, seq_len, hidden_size)

    # check type & numerical properties
    assert isinstance(output, torch.Tensor)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    # layernorm should center roughly near 0
    mean_val = output.mean().item()
    assert abs(mean_val) < 1.0  # not too far from 0

    # dropout should not change shape
    model.eval()
    with torch.no_grad():
        output_eval = model(input_ids, token_type_ids)
    assert output_eval.shape == output.shape