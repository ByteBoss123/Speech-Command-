"""Verify exported intent results offline from raw predictions and saved heads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import fingerprint
from .intent import MODES, classification_metrics, evaluation_index, load_head, paired_bootstrap, train_statistics
from .intent_data import INTENTS, validate_intent_rows


def verify(directory, features=None, manifest=None):
    directory = Path(directory)
    report = json.loads((directory / "metrics.json").read_text())
    predictions = json.loads((directory / "predictions_test.json").read_text())
    rows = json.loads((directory / "evaluation_index.json").read_text())
    if not rows or len({r["id"] for r in rows}) != len(rows) or any(
            r["split"] not in ("train", "validation", "test") or r["label"] not in range(len(INTENTS)) for r in rows):
        raise ValueError("Invalid public evaluation index")
    test_rows = [r for r in rows if r["split"] == "test"]
    if len(predictions) != len(test_rows) or any(
            p["id"] != r["id"] or p["true_label"] != r["label"] or p["language"] != r["language"]
            for p, r in zip(predictions, test_rows)):
        raise ValueError("Predictions do not align with test manifest")
    if report["counts"] != {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}:
        raise ValueError("Split counts mismatch")
    truth = np.array([r["label"] for r in test_rows])
    languages = np.array([r["language"] for r in test_rows])
    labels, heads = {}, {}
    for mode in MODES:
        probabilities = np.array([p["predictions"][mode]["probabilities"] for p in predictions])
        if probabilities.shape != (len(test_rows), len(INTENTS)) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1):
            raise ValueError("Invalid probability vectors")
        labels[mode] = probabilities.argmax(1)
        if labels[mode].tolist() != [p["predictions"][mode]["label"] for p in predictions]:
            raise ValueError("Stored argmax mismatch")
        actual = classification_metrics(truth, labels[mode])
        if actual != report["models"][mode]["test"]:
            raise ValueError("Aggregate metrics mismatch")
        for language in sorted(set(languages)):
            take = languages == language
            if classification_metrics(truth[take], labels[mode][take]) != report["models"][mode]["by_language"][language]:
                raise ValueError("Per-language metrics mismatch")
        heads[mode], config = load_head(directory / mode)
        if config["labels"] != INTENTS or config["trained_head_parameters"] != sum(p.numel() for p in heads[mode].parameters()):
            raise ValueError("Checkpoint labels or parameter count mismatch")
        history = json.loads((directory / mode / "history.json").read_text())
        selected = max(history, key=lambda r: (r["validation_macro_f1"], -r["validation_loss"]))["epoch"]
        if selected != config["best_epoch"] or selected != report["models"][mode]["best_epoch"]:
            raise ValueError("Checkpoint selection is inconsistent with validation history")
    comparison = paired_bootstrap(truth, labels["fusion"], labels["text"], languages,
                                  samples=report["fusion_minus_text"]["samples"], seed=report["fusion_minus_text"]["seed"])
    if comparison != report["fusion_minus_text"]:
        raise ValueError("Paired confidence interval mismatch")
    for ablation in report["fusion_ablations"].values():
        if classification_metrics(truth, ablation["predicted_labels"]) != ablation["metrics"]:
            raise ValueError("Ablation metrics mismatch")
    for name in ("audio_mean", "audio_scale", "text_mean", "text_scale"):
        if not all(torch.equal(getattr(heads["fusion"], name), getattr(heads[m], name)) for m in MODES):
            raise ValueError("Normalization differs between modes")
    if features is not None:
        if manifest is None:
            raise ValueError("Full cache replay requires the locally generated manifest")
        manifest = Path(manifest)
        local_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        validate_intent_rows(local_rows)
        if fingerprint(manifest) != report["manifest_sha256"] or evaluation_index(local_rows) != rows:
            raise ValueError("Local manifest does not match recorded experiment")
        features = Path(features)
        if fingerprint(features / "features.npz") != report["features_metadata"]["features_sha256"]:
            raise ValueError("Feature cache hash mismatch")
        with np.load(features / "features.npz", allow_pickle=False) as cache:
            if cache["ids"].tolist() != [r["id"] for r in local_rows]:
                raise ValueError("Cache IDs do not align")
            train_idx = [i for i, r in enumerate(rows) if r["split"] == "train"]
            stats = train_statistics(cache["audio"], cache["text"], train_idx)
            if not all(torch.equal(getattr(heads["fusion"], k), v) for k, v in stats.items()):
                raise ValueError("Normalization is not fitted to training rows")
            test_idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
            for mode in MODES:
                with torch.no_grad():
                    actual = heads[mode](torch.from_numpy(cache["audio"][test_idx]), torch.from_numpy(cache["text"][test_idx])).softmax(1).numpy()
                expected = np.array([p["predictions"][mode]["probabilities"] for p in predictions])
                if not np.allclose(actual, expected, atol=1e-6):
                    raise ValueError("Reloaded head probabilities mismatch")
    return {"status": "passed", "test_predictions_verified": len(predictions), "models_verified": list(MODES),
            "checks": ["public split index", "prediction alignment", "probability vectors",
                       "aggregate and per-language metrics", "validation-only checkpoint selection",
                       "bootstrap recomputation", "ablation metrics", "normalization consistency"],
            "feature_cache_replay": features is not None,
            "local_manifest_checksum_and_leakage_checks": features is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="evidence/2026-11-19/intent")
    parser.add_argument("--features", help="Optional local feature-cache directory for full checkpoint replay")
    parser.add_argument("--manifest", help="Local source manifest; required with --features")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(verify(args.directory, args.features, args.manifest), indent=2))


if __name__ == "__main__":
    main()
