"""
dataset.py

Loads the manifest, extracts fixed-size MFCC features per clip, and
exposes a PyTorch Dataset. Also builds the label vocabulary and the
train/val split (stratified, sample-level, drawn only from the
'trainval' rows -- 'unseen' rows are never touched here).
"""

import csv
from pathlib import Path

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

from corpus import VOCAB

MANIFEST_PATH = Path(__file__).parent / "manifest.csv"
AUDIO_DIR = Path(__file__).parent
N_MFCC = 40
MAX_FRAMES = 48  # covers the longest clip (35 frames) with headroom

WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return list(csv.DictReader(f))


def extract_mfcc(wav_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(wav_path), sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)  # (n_mfcc, T)
    if mfcc.shape[1] >= MAX_FRAMES:
        mfcc = mfcc[:, :MAX_FRAMES]
    else:
        pad = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
    # per-clip normalization
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc.astype(np.float32)


class SpeechCommandDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        mfcc = extract_mfcc(AUDIO_DIR / row["path"])
        label = WORD_TO_IDX[row["word"]]
        x = torch.from_numpy(mfcc).unsqueeze(0)  # (1, n_mfcc, T)
        return x, label


def build_splits(val_fraction=0.2, seed=13):
    """Stratified sample-level train/val split from 'trainval' rows only,
    plus the fully-held-out 'unseen' rows returned separately, untouched."""
    manifest = load_manifest()
    trainval = [r for r in manifest if r["split"] == "trainval"]
    unseen = [r for r in manifest if r["split"] == "unseen"]

    rng = np.random.RandomState(seed)
    by_word = {}
    for r in trainval:
        by_word.setdefault(r["word"], []).append(r)

    train_rows, val_rows = [], []
    for word, rows in by_word.items():
        rows = rows.copy()
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * val_fraction))
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])

    return train_rows, val_rows, unseen
