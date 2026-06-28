# [4] Training — training loop with warmup LR schedule, validation, checkpoint saving

import os
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import (
    d_model, warmup_steps, epochs, label_smoothing,
    pad_id_regex, device
)
from src.data import get_dataloaders
from src.model import Transformer, count_parameters

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, decoder_input, decoder_target in tqdm(dataloader, desc="Train"):
        src = src.to(device)
        decoder_input = decoder_input.to(device)
        decoder_target = decoder_target.to(device)

        optimizer.zero_grad()

        logits = model(src, decoder_input)

        loss = criterion(logits.view(-1, logits.size(-1)), decoder_target.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        num_tokens = (decoder_target != pad_id_regex).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, decoder_input, decoder_target in tqdm(dataloader, desc="Val"):
            src = src.to(device)
            decoder_input = decoder_input.to(device)
            decoder_target = decoder_target.to(device)

            logits = model(src, decoder_input)

            loss = criterion(logits.view(-1, logits.size(-1)), decoder_target.view(-1))

            num_tokens = (decoder_target != pad_id_regex).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def run_training():
    print(f"Using device: {device}")
    print(f"d_model={d_model}, warmup={warmup_steps}, epochs={epochs}")

    train_loader, val_loader, _ = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = Transformer().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id_regex, label_smoothing=label_smoothing
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    lr_scheduler = WarmupScheduler(optimizer, d_model, warmup_steps)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler)
        val_loss = validate(model, val_loader, criterion)

        epoch_time = time.time() - start_time
        current_lr = lr_scheduler.get_lr()

        ppl_train = math.exp(train_loss)
        ppl_val = math.exp(val_loss)

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"LR: {current_lr:.2e} | "
              f"Train loss: {train_loss:.4f} (ppl: {ppl_train:.2f}) | "
              f"Val loss: {val_loss:.4f} (ppl: {ppl_val:.2f}) | "
              f"Time: {epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    print("Training complete!")


if __name__ == "__main__":
    run_training()
