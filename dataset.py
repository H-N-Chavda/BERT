import torch, random, math
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class WikiTextBERTDataset(Dataset):
    def __init__(self, split, tokenizer, dataset, max_len=64, mlm_prob=0.15):
        self.samples = dataset[split]["text"]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.sentences = [s for s in self.samples if len(s.strip()) > 0]

    def __len__(self):
        return len(self.sentences) - 1  # for pairing

    def random_mask(self, token_ids):
        """Applies 15% masking as per BERT rules (80/10/10)."""
        labels = [-100] * len(token_ids)
        for i in range(len(token_ids)):
            if random.random() < self.mlm_prob and token_ids[i] not in [self.tokenizer.cls_id, self.tokenizer.sep_id]:
                labels[i] = token_ids[i]
                rnd = random.random()
                if rnd < 0.8:
                    token_ids[i] = self.tokenizer.mask_id
                elif rnd < 0.9:
                    token_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
                # else: keep same token
        return token_ids, labels

    def __getitem__(self, idx):
        s1 = self.sentences[idx]
        if random.random() < 0.5:
            s2 = self.sentences[idx + 1]  # IsNext
            label = 1
        else:
            s2 = random.choice(self.sentences)  # NotNext
            label = 0

        tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(s1) + \
                 [self.tokenizer.sep_token] + self.tokenizer.tokenize(s2) + [self.tokenizer.sep_token]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = token_ids[: self.max_len]
        token_ids, mlm_labels = self.random_mask(token_ids)

        attn_mask = [1] * len(token_ids)
        token_type_ids = [0] * (len(self.tokenizer.tokenize(s1)) + 2) + \
                         [1] * (len(token_ids) - len(self.tokenizer.tokenize(s1)) - 2)

        # pad
        pad_len = self.max_len - len(token_ids)
        token_ids += [self.tokenizer.pad_id] * pad_len
        token_type_ids += [0] * pad_len
        attn_mask += [0] * pad_len
        mlm_labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(token_ids),
            "token_type_ids": torch.tensor(token_type_ids),
            "attention_mask": torch.tensor(attn_mask),
            "mlm_labels": torch.tensor(mlm_labels),
            "nsp_labels": torch.tensor(label)
        }
