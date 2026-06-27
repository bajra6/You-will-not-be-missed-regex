# [2] Tokenizers — train and load BPE tokenizers for NL descriptions (vocab=4000) and regexes (vocab=256)

import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from src.config import vocab_nl, vocab_regex, TRAIN_JSON

NL_TOKENIZER_FILE = Path("data") / "nl_tokenizer.json"
REGEX_TOKENIZER_FILE = Path("data") / "regex_tokenizer.json"


def train_nl_tokenizer():
    with open(TRAIN_JSON, encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["nl"] for item in data]
    print(f"Training NL tokenizer on {len(texts)} descriptions...")

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_nl,
        special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"],
        min_frequency=2,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.save(str(NL_TOKENIZER_FILE))
    print(f"Saved NL tokenizer with vocab={tokenizer.get_vocab_size()} to {NL_TOKENIZER_FILE}")

    return tokenizer


def load_nl_tokenizer():
    return Tokenizer.from_file(str(NL_TOKENIZER_FILE))


def train_regex_tokenizer():
    with open(TRAIN_JSON, encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["regex"] for item in data]
    print(f"Training regex tokenizer on {len(texts)} regexes...")

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_regex,
        special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"],
        min_frequency=2,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.save(str(REGEX_TOKENIZER_FILE))
    print(f"Saved regex tokenizer with vocab={tokenizer.get_vocab_size()} to {REGEX_TOKENIZER_FILE}")

    return tokenizer


def load_regex_tokenizer():
    return Tokenizer.from_file(str(REGEX_TOKENIZER_FILE))


if __name__ == "__main__":
    nl_tok = train_nl_tokenizer()
    test = "find a string of two uppercase letters"
    encoded = nl_tok.encode(test)
    print(f"NL test: '{test}' → tokens: {' | '.join(encoded.tokens)}")

    rx_tok = train_regex_tokenizer()
    test = "^[A-Z]{2}[0-9]{1}$"
    encoded = rx_tok.encode(test)
    print(f"Regex test: '{test}' → tokens: {' | '.join(encoded.tokens)}")
