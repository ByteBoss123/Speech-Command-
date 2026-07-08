"""
evaluate.py

The real test: how well does the model classify words spoken by voices it
never saw during training? This is evaluated separately from in-distribution
validation accuracy, because in-distribution accuracy on a small, clean,
synthetic dataset is expected to saturate near 100% and says little about
generalization.
"""

import json
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from corpus import VOCAB
from dataset import build_splits, SpeechCommandDataset
from model import SpeechCommandCNN

RESULTS_DIR = Path(__file__).parent / "results"
CHECKPOINT_PATH = RESULTS_DIR / "best_model.pt"


def evaluate():
    _, _, unseen_rows = build_splits()
    loader = DataLoader(SpeechCommandDataset(unseen_rows), batch_size=32)

    model = SpeechCommandCNN(num_classes=len(VOCAB))
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    word_to_idx = {w: i for i, w in enumerate(VOCAB)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    confusion = defaultdict(lambda: defaultdict(int))
    correct, total = 0, 0
    per_voice_correct = defaultdict(int)
    per_voice_total = defaultdict(int)

    # need voice per-sample; re-derive alignment since DataLoader doesn't
    # carry manifest metadata through
    idx = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = logits.argmax(dim=1)
            for p, t in zip(preds.tolist(), y.tolist()):
                row = unseen_rows[idx]
                voice = row["voice"]
                confusion[idx_to_word[t]][idx_to_word[p]] += 1
                correct += int(p == t)
                total += 1
                per_voice_correct[voice] += int(p == t)
                per_voice_total[voice] += 1
                idx += 1

    overall_acc = correct / total
    per_voice_acc = {v: per_voice_correct[v] / per_voice_total[v] for v in per_voice_total}

    print(f"Unseen-voice test accuracy: {overall_acc:.3f} ({correct}/{total})")
    for v, acc in per_voice_acc.items():
        print(f"  {v}: {acc:.3f} ({per_voice_correct[v]}/{per_voice_total[v]})")

    result = {
        "unseen_voice_overall_accuracy": overall_acc,
        "unseen_voice_per_voice_accuracy": per_voice_acc,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "n_test_samples": total,
    }
    with open(RESULTS_DIR / "unseen_voice_eval.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    evaluate()
