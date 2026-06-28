# [6] Evaluation — exact match accuracy, regex validity rate, BLEU score

import re
import json
import math
import torch
from collections import Counter

from src.config import device
from src.tokenizers import load_nl_tokenizer, load_regex_tokenizer
from src.model import Transformer
from src.infer import predict, validate_regex
from src.data import TEST_JSON


def tokenize_regex_for_bleu(pattern):
    tokens = []
    i = 0
    while i < len(pattern):
        if pattern[i] == "\\" and i + 1 < len(pattern):
            tokens.append(pattern[i : i + 2])
            i += 2
        elif pattern[i] in ".^$*+?{}[]()|":
            tokens.append(pattern[i])
            i += 1
        else:
            j = i
            while j < len(pattern) and pattern[j] not in ".^$*+?{}[]()|\\":
                j += 1
            tokens.append(pattern[i:j])
            i = j
    return tokens


def bleu_score(predicted, reference, max_n=4):
    pred_tokens = tokenize_regex_for_bleu(predicted)
    ref_tokens = tokenize_regex_for_bleu(reference)

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )

        matches = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)

    geometric_mean = 1.0
    for p in precisions:
        if p == 0:
            return 0.0
        geometric_mean *= p

    bleu = geometric_mean ** (1.0 / max_n)

    bp_len = len(pred_tokens)
    ref_len = len(ref_tokens)

    if bp_len < ref_len:
        bleu *= math.exp(1 - ref_len / bp_len) if bp_len > 0 else 0.0

    return bleu


def evaluate():
    nl_tokenizer = load_nl_tokenizer()
    regex_tokenizer = load_regex_tokenizer()

    model = Transformer().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval()

    with open(TEST_JSON, encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Evaluating on {len(test_data)} test samples...\n")

    exact_matches = 0
    valid_count = 0
    total_bleu = 0.0

    for i, item in enumerate(test_data):
        nl = item["nl"]
        expected = item["regex"]

        predicted, is_valid = predict(model, nl, nl_tokenizer, regex_tokenizer)

        if is_valid:
            valid_count += 1

        if predicted == expected:
            exact_matches += 1

        total_bleu += bleu_score(predicted, expected)

        if i < 5:
            match = "✓" if predicted == expected else "✗"
            valid_str = "v" if is_valid else "!"
            print(f"  [{match}{valid_str}] NL: {nl}")
            print(f"        Pred: {predicted}")
            print(f"        Exp:  {expected}")
            print()

    total = len(test_data)
    exact_accuracy = exact_matches / total * 100
    validity_rate = valid_count / total * 100
    avg_bleu = total_bleu / total

    print("=" * 50)
    print(f"Exact match: {exact_matches}/{total} = {exact_accuracy:.2f}%")
    print(f"Valid regexes: {valid_count}/{total} = {validity_rate:.2f}%")
    print(f"Avg BLEU: {avg_bleu:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()
