"""
Preprocessing pipeline for headline generation:
  raw CSV -> clean text -> tokenize -> build vocab -> train/val/test splits -> PyTorch Dataset

Run standalone to build and cache the splits + vocab:
    python -m src.data
"""
import json
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src import config

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9.,!?'\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return text.split()


class Vocab:
    def __init__(self, token_counter: Counter, min_freq: int, size_cap: int):
        self.itos = list(SPECIAL_TOKENS)
        for tok, freq in token_counter.most_common(size_cap):
            if freq >= min_freq and tok not in SPECIAL_TOKENS:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, tokens: list[str], max_len: int, add_sos_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokens]
        if add_sos_eos:
            ids = [self.stoi[SOS]] + ids + [self.stoi[EOS]]
        ids = ids[:max_len]
        ids += [self.stoi[PAD]] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        toks = []
        for i in ids:
            tok = self.itos[i]
            if tok == EOS:
                break
            if tok not in (PAD, SOS):
                toks.append(tok)
        return toks

    def __len__(self):
        return len(self.itos)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            itos = json.load(f)
        v = cls.__new__(cls)
        v.itos = itos
        v.stoi = {tok: i for i, tok in enumerate(itos)}
        return v


@dataclass
class Example:
    src_text: str
    tgt_text: str


class HeadlineDataset(Dataset):
    def __init__(self, examples: list[Example], src_vocab: Vocab, tgt_vocab: Vocab):
        self.examples = examples
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        src_ids = self.src_vocab.encode(tokenize(ex.src_text), config.MAX_SRC_LEN)
        tgt_ids = self.tgt_vocab.encode(tokenize(ex.tgt_text), config.MAX_TGT_LEN, add_sos_eos=True)
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": ex.src_text,
            "tgt_text": ex.tgt_text,
        }


def load_and_split(path: str = config.RAW_DATA_PATH, article_col="text", headline_col="headlines"):
    """
    Expects a CSV with an article column and a headline column.
    Adjust article_col/headline_col to match your chosen dataset's schema.
    """
    df = pd.read_csv(path, encoding="latin-1")
    df = df[[article_col, headline_col]].dropna()
    df[article_col] = df[article_col].apply(clean_text)
    df[headline_col] = df[headline_col].apply(clean_text)
    df = df[(df[article_col].str.len() > 0) & (df[headline_col].str.len() > 0)]

    train_df, temp_df = train_test_split(df, test_size=(1 - config.TRAIN_SPLIT), random_state=config.SEED)
    rel_val = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - rel_val), random_state=config.SEED)

    def to_examples(d):
        return [Example(a, h) for a, h in zip(d[article_col], d[headline_col])]

    return to_examples(train_df), to_examples(val_df), to_examples(test_df)


def build_vocab(examples: list[Example], from_field: str) -> Vocab:
    counter = Counter()
    for ex in examples:
        text = ex.src_text if from_field == "src" else ex.tgt_text
        counter.update(tokenize(text))
    return Vocab(counter, config.MIN_VOCAB_FREQ, config.VOCAB_SIZE_CAP)


if __name__ == "__main__":
    train_ex, val_ex, test_ex = load_and_split()
    print(f"train/val/test sizes: {len(train_ex)}/{len(val_ex)}/{len(test_ex)}")

    src_vocab = build_vocab(train_ex, "src")
    tgt_vocab = build_vocab(train_ex, "tgt")
    print(f"src vocab: {len(src_vocab)} | tgt vocab: {len(tgt_vocab)}")

    src_vocab.save("checkpoints/src_vocab.json")
    tgt_vocab.save("checkpoints/tgt_vocab.json")
