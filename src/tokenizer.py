import re
import torch
import torch.nn as nn
from collections import Counter

class Tokenizer(nn.Module):
    def __init__(self, vocab_size=5000, lowercase=True, special_tokens=None):
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.special_tokens = special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab = {}
        self.inv_vocab = {}

    def train(self, corpus: str):
        if self.lowercase:
            corpus = corpus.lower()

        words = corpus.strip().split()
        vocab = Counter([' '.join(list(w)) + ' </w>' for w in words])

        # iterative merging based on frequency 
        for _ in range(self.vocab_size):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)

        # Final vocab extraction
        tokens = set()
        for word in vocab:
            tokens.update(word.split())

        # Add special tokens
        for t in self.special_tokens:
            tokens.add(t)

        self.vocab = {tok: i for i, tok in enumerate(sorted(tokens))}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

    def _get_stats(self, vocab):
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        pattern = re.escape(' '.join(pair))
        replacement = ''.join(pair)
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = re.sub(pattern, replacement, word)
            new_vocab[new_word] = freq
        return new_vocab

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        words = text.strip().split()
        output = []
        for w in words:
            output.extend(self._encode_word(w))
        return output

    def _encode_word(self, word):
        word = list(word) + ["</w>"]
        tokens = []
        while len(word) > 0:
            subword = None
            for i in range(len(word), 0, -1):
                candidate = ''.join(word[:i])
                if candidate in self.vocab:
                    subword = candidate
                    tokens.append(subword)
                    word = word[i:]
                    break
            if subword is None:
                tokens.append("[UNK]")
                break
        return tokens

    def encode(self, text_a, text_b=None, max_len=64):
        tokens_a = self.tokenize(text_a)
        tokens_b = self.tokenize(text_b) if text_b else []

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        token_type_ids = [0] * len(tokens)

        if text_b:
            tokens += tokens_b + ["[SEP]"]
            token_type_ids += [1] * (len(tokens_b) + 1)

        input_ids = [self.vocab.get(tok, self.vocab["[UNK]"]) for tok in tokens]
        attention_mask = [1] * len(input_ids)

        # pad if needed
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.vocab["[PAD]"]] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len

        # truncate if longer
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        token_type_ids = token_type_ids[:max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }
