# [5] Inference — greedy decode, regex detokenization, validation with re.compile

import re
import torch

from src.config import sos_id_regex, eos_id_regex, pad_id_regex, max_len_regex, device
from src.tokenizers import load_nl_tokenizer, load_regex_tokenizer
from src.model import Transformer


def greedy_decode(model, src_tokens, max_len=max_len_regex):
    model.eval()

    with torch.no_grad():
        src = src_tokens.unsqueeze(0).to(device)
        tgt = torch.tensor([[sos_id_regex]], device=device)

        for _ in range(max_len - 1):
            logits = model(src, tgt)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == eos_id_regex:
                break

    return tgt.squeeze(0)


def decode_regex(token_ids, regex_tokenizer):
    ids = token_ids.tolist()
    ids = [i for i in ids if i not in (pad_id_regex, sos_id_regex, eos_id_regex)]
    return regex_tokenizer.decode(ids)


def validate_regex(pattern):
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def predict(model, nl_text, nl_tokenizer, regex_tokenizer):
    encoded = nl_tokenizer.encode(nl_text)
    src_tokens = torch.tensor(encoded.ids, dtype=torch.long)

    token_ids = greedy_decode(model, src_tokens)
    pattern = decode_regex(token_ids, regex_tokenizer)
    is_valid = validate_regex(pattern)

    return pattern, is_valid


if __name__ == "__main__":
    nl_tokenizer = load_nl_tokenizer()
    regex_tokenizer = load_regex_tokenizer()

    model = Transformer().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    print("Model loaded from checkpoints/best_model.pt")

    tests = [
        "find a string of two uppercase letters followed by one digit",
        "find a string with eight to twelve characters, at least two uppercase letter, two lowercase letter, and four digits",
        "match a string containing the standalone whole word flag",
    ]

    for test in tests:
        pattern, is_valid = predict(model, test, nl_tokenizer, regex_tokenizer)
        print(f"NL: {test}")
        print(f"Regex: {pattern}  (valid={is_valid})\n")
