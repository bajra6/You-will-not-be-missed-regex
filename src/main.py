# [7] Orchestrator — runs pipeline steps in sequence: data -> tokenize -> model -> train

import sys

from src.config import device
from src.data import load_and_split
from src.tokenizers import train_nl_tokenizer, train_regex_tokenizer
from src.model import Transformer, count_parameters
from src.train import run_training


def main():
    print("=" * 50)
    print("NL2Regex Transformer")
    print("=" * 50)
    print(f"Device: {device}")
    print()

    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = "all"

    if command in ("data", "all"):
        print("[1/4] Loading and splitting data...")
        load_and_split()
        print()

    if command in ("tokenize", "all"):
        print("[2/4] Training NL tokenizer...")
        train_nl_tokenizer()
        print("[2/4] Training regex tokenizer...")
        train_regex_tokenizer()
        print()

    if command in ("model", "all"):
        print("[3/4] Building model...")
        model = Transformer()
        print(f"  Parameters: {count_parameters(model):,}")
        print()

    if command in ("train", "all"):
        print("[4/4] Training...")
        run_training()
        print()

    print("Done.")


if __name__ == "__main__":
    main()
