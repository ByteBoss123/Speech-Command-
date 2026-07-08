import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from model import SpeechCommandCNN
from dataset import build_splits, extract_mfcc, N_MFCC, MAX_FRAMES, AUDIO_DIR
from corpus import VOCAB, TRAIN_VOICES, UNSEEN_VOICES


def test_model_output_shape():
    model = SpeechCommandCNN(num_classes=len(VOCAB))
    x = torch.randn(8, 1, N_MFCC, MAX_FRAMES)
    out = model(x)
    assert out.shape == (8, len(VOCAB))


def test_mfcc_feature_shape():
    sample_path = next(AUDIO_DIR.glob("audio/*.wav"))
    mfcc = extract_mfcc(sample_path)
    assert mfcc.shape == (N_MFCC, MAX_FRAMES)


def test_no_voice_leakage_between_train_and_unseen():
    """The most important test in this project: verify the held-out
    voices never appear in the train or val split."""
    train_rows, val_rows, unseen_rows = build_splits()

    train_voices = set(r["voice"] for r in train_rows)
    val_voices = set(r["voice"] for r in val_rows)
    unseen_voices_actual = set(r["voice"] for r in unseen_rows)

    assert train_voices.issubset(set(TRAIN_VOICES))
    assert val_voices.issubset(set(TRAIN_VOICES))
    assert unseen_voices_actual == set(UNSEEN_VOICES)
    assert train_voices.isdisjoint(unseen_voices_actual)
    assert val_voices.isdisjoint(unseen_voices_actual)


def test_all_vocab_words_present_in_each_split():
    train_rows, val_rows, unseen_rows = build_splits()
    for rows, name in [(train_rows, "train"), (val_rows, "val"), (unseen_rows, "unseen")]:
        words_present = set(r["word"] for r in rows)
        assert words_present == set(VOCAB), f"{name} split missing words: {set(VOCAB) - words_present}"


def test_train_val_split_sizes_reasonable():
    train_rows, val_rows, unseen_rows = build_splits()
    assert len(train_rows) > len(val_rows)
    total_trainval = len(train_rows) + len(val_rows)
    val_frac = len(val_rows) / total_trainval
    assert 0.15 <= val_frac <= 0.25


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
