import copy

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from speechlab.common import dump_json
from speechlab.intent import (IntentHead, classification_metrics, evaluation_index, load_head,
                              paired_bootstrap, train_statistics)
from speechlab.intent_data import INTENTS, balanced_split, text_key, validate_intent_rows
from speechlab.intent_features import pool_audio, pool_text


def source_rows():
    return [{"id": f"{lang}-{label}-{i}", "language": lang, "label": label,
             "text": f"{lang} request {label} sample {i}", "source_audio_sha256": f"{lang}-{label}-{i}"}
            for lang in ("en-US", "fr-FR") for label in range(len(INTENTS)) for i in range(5)]


def test_balanced_split_is_order_independent_and_removes_duplicates():
    rows = source_rows()
    duplicate = {**rows[0], "id": "different-id", "text": rows[0]["text"].upper() + "!"}
    quotas = {"train": 2, "validation": 1, "test": 1}
    first = balanced_split(rows + [duplicate], ["en-US", "fr-FR"], quotas)
    second = balanced_split(list(reversed(rows + [duplicate])), ["en-US", "fr-FR"], quotas)
    assert first == second
    assert len(first) == len({text_key(r["text"]) for r in first}) == 112
    for r in first:
        r["audio_sha256"] = r["source_audio_sha256"]
    validate_intent_rows(first)
    for language in ("en-US", "fr-FR"):
        for label in range(14):
            for split, count in quotas.items():
                assert sum(r["language"] == language and r["label"] == label and r["split"] == split for r in first) == count


def test_intent_validation_rejects_cross_split_transcript_and_audio_leakage():
    rows = balanced_split(source_rows(), ["en-US", "fr-FR"], {"train": 2, "validation": 1, "test": 1})
    for r in rows:
        r["audio_sha256"] = r["source_audio_sha256"]
    train = next(i for i, r in enumerate(rows) if r["split"] == "train")
    test = next(i for i, r in enumerate(rows) if r["split"] == "test")
    contaminated = copy.deepcopy(rows)
    contaminated[test]["text"] = contaminated[train]["text"]
    with pytest.raises(ValueError, match="transcript leakage"):
        validate_intent_rows(contaminated)
    contaminated = copy.deepcopy(rows)
    contaminated[test]["audio_sha256"] = contaminated[train]["audio_sha256"]
    with pytest.raises(ValueError, match="Duplicate recording"):
        validate_intent_rows(contaminated)
    with pytest.raises(ValueError, match="eligible unique clips"):
        balanced_split(source_rows(), ["en-US", "fr-FR"], {"train": 10, "validation": 1, "test": 1})


def test_pooling_does_not_include_padded_frames_or_tokens():
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [1000.0, 1000.0]]])
    assert torch.allclose(pool_audio(hidden, [2]), torch.tensor([[2.0, 4.0, 1.0, 1.0]]))
    assert torch.allclose(pool_text(hidden, torch.tensor([[1, 1, 0]])), torch.tensor([[2.0, 4.0]]))
    with pytest.raises(ValueError, match="frame count"):
        pool_audio(hidden, [4])


def test_normalization_cannot_fit_holdout_rows():
    features = np.array([[1, 2], [3, 2], [100, -100]], dtype=np.float32)
    first = train_statistics(features, features, [0, 1])
    features[2] = [1e6, -1e6]
    second = train_statistics(features, features, [0, 1])
    for k in first:
        assert torch.equal(first[k], second[k])
    assert torch.equal(first["audio_mean"], torch.tensor([2.0, 2.0]))
    assert (first["audio_scale"] > 0).all()


def test_public_index_excludes_record_content_and_provenance():
    row = {"id": "source-record-id", "split": "test", "language": "en-US", "label": 2,
           "text": "private example content", "path": "/audio/source.wav",
           "source_path": "original/recording", "audio_sha256": "source-fingerprint"}
    result = evaluation_index([row])
    assert result == [{"id": "example_00000", "split": "test", "language": "en-US", "label": 2}]
    assert not any(value in str(result) for value in [row["text"], row["id"], row["path"], row["audio_sha256"]])


def test_both_modalities_affect_fusion_and_checkpoint_preserves_predictions(tmp_path):
    torch.manual_seed(13)
    audio, text = torch.randn(12, 3), torch.randn(12, 2)
    stats = train_statistics(audio.numpy(), text.numpy(), list(range(8)))
    architecture = dict(mode="fusion", audio_dim=3, text_dim=2, hidden=8, classes=2, dropout=0.0)
    model = IntentHead(**architecture, statistics=stats).eval()
    labels = torch.arange(12) % 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    before = torch.nn.functional.cross_entropy(model(audio, text), labels).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(audio, text), labels)
        loss.backward()
        assert model.classifier[0].weight.grad[:, :3].abs().sum() > 0
        assert model.classifier[0].weight.grad[:, 3:].abs().sum() > 0
        optimizer.step()
    assert torch.nn.functional.cross_entropy(model(audio, text), labels).item() < before
    expected = model(audio, text)
    assert not torch.allclose(expected, model(audio, text, drop="audio"))
    assert not torch.allclose(expected, model(audio, text, drop="text"))
    save_file(model.state_dict(), str(tmp_path / "head.safetensors"))
    dump_json(tmp_path / "config.json", {"architecture": architecture})
    reloaded, _ = load_head(tmp_path)
    assert torch.equal(expected, reloaded(audio, text))
    assert torch.equal(model.audio_mean, reloaded.audio_mean)


def test_metrics_and_paired_bootstrap_keep_predictions_paired():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], classes=2)
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    result = paired_bootstrap([0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], ["en", "en", "fr", "fr"], samples=100)
    assert result["accuracy_difference"] == 0
    assert result["ci95"] == [0, 0]
