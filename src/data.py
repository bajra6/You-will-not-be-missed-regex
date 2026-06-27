# [1] Data pipeline — load CSV, normalize, split into train/val/test JSONs, PyTorch Dataset + DataLoader

import csv
import json
import re
import random

import torch
from torch.utils.data import Dataset, DataLoader
# Dataset helps in memory management (lazy loading) and Dataloader with shape formatting and multithreading
# (NLP datasets have varying lengths) So we use collate_fn to pad the sequences to the same length in a batch
# Also needed for random shuffling


from src.config import (
    max_len_nl, max_len_regex, DATA_DIR,
    pad_id_nl, pad_id_regex, sos_id_regex, eos_id_regex, batch_size,
    RAW_CSV, TRAIN_JSON, VAL_JSON, TEST_JSON
)
from src.tokenizers import load_nl_tokenizer, load_regex_tokenizer

DATA_DIR.mkdir(exist_ok=True)

SEED = 42


def normalize_nl(text):
    # function is destructive in our case
    # text = text.lower()
    # text = re.sub(r"[^a-z0-9\s]", "", text)
    # text = re.sub(r"\s+", " ", text).strip()
    return text.strip()


def normalize_regex(text):
    # text = text.strip()
    # text = re.sub(r"\s+", "", text)
    return text


def load_and_split():
    pairs = []
    with open(RAW_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            nl = normalize_nl(row[0])
            rx = normalize_regex(row[1])
            pairs.append({"nl": nl, "regex": rx})

    print(f"Loaded {len(pairs)} pairs")

    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train = pairs[:train_end]
    val = pairs[train_end:val_end]
    test = pairs[val_end:]

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    with open(TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
    with open(VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2)
    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)

    return train, val, test


class NL2RegexDataset(Dataset):
    def __init__(self, json_path, nl_tokenizer, regex_tokenizer):
        with open(json_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.nl_tokenizer = nl_tokenizer
        self.regex_tokenizer = regex_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        nl = item["nl"]
        regex = item["regex"]

        src_ids = self.nl_tokenizer.encode(nl).ids
        tgt_ids = self.regex_tokenizer.encode(regex).ids

        src_ids = src_ids[:max_len_nl]
        tgt_ids = tgt_ids[:max_len_regex - 2]

        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)

        decoder_input = torch.cat([torch.tensor([sos_id_regex]), tgt_ids])
        decoder_target = torch.cat([tgt_ids, torch.tensor([eos_id_regex])])

        return src_ids, decoder_input, decoder_target


def collate_fn(batch):
    src_ids_list = [item[0] for item in batch]
    decoder_input_list = [item[1] for item in batch]
    decoder_target_list = [item[2] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_ids_list, batch_first=True, padding_value=pad_id_nl
    )
    decoder_input_padded = torch.nn.utils.rnn.pad_sequence(
        decoder_input_list, batch_first=True, padding_value=pad_id_regex
    )
    decoder_target_padded = torch.nn.utils.rnn.pad_sequence(
        decoder_target_list, batch_first=True, padding_value=pad_id_regex
    )

    return src_padded, decoder_input_padded, decoder_target_padded


def get_dataloaders():
    nl_tokenizer = load_nl_tokenizer()
    regex_tokenizer = load_regex_tokenizer()

    train_dataset = NL2RegexDataset(TRAIN_JSON, nl_tokenizer, regex_tokenizer)
    val_dataset = NL2RegexDataset(VAL_JSON, nl_tokenizer, regex_tokenizer)
    test_dataset = NL2RegexDataset(TEST_JSON, nl_tokenizer, regex_tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    load_and_split()
