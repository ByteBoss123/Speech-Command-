"""
train.py

Trains SpeechCommandCNN from scratch on the synthetic MFCC dataset,
tracking real per-epoch train/val loss and accuracy. Saves the best
checkpoint (by val accuracy) and the training curve data.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_splits, SpeechCommandDataset
from model import SpeechCommandCNN

RESULTS_DIR = Path(__file__).parent / "results"
CHECKPOINT_PATH = RESULTS_DIR / "best_model.pt"
EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(13)

    train_rows, val_rows, unseen_rows = build_splits()
    train_loader = DataLoader(SpeechCommandDataset(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SpeechCommandDataset(val_rows), batch_size=BATCH_SIZE)

    model = SpeechCommandCNN(num_classes=len(set(r["word"] for r in train_rows)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        print(f"epoch {epoch:2d}  train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")
    return history


if __name__ == "__main__":
    main()
