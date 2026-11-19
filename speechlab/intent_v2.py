"""Expanded-data seed ensembles and validation-selected audio/text fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file

from .common import dump_json, environment, fingerprint, seed_everything
from .intent import (MODES, classification_metrics, evaluation_index, fit_head, load_head,
                     paired_bootstrap, train_statistics)
from .intent_data import INTENTS, read_intent_manifest

SEEDS = (13, 29, 47)
ALPHAS = (0., .1, .25, .5, .75, 1.)


def select_fusion(text, joint, truth):
    """Select probability blend only on caller-supplied development predictions."""
    history = []
    truth = np.asarray(truth)
    for alpha in ALPHAS:
        probabilities = (1 - alpha) * text + alpha * joint
        scores = classification_metrics(truth, probabilities.argmax(1))
        nll = float(-np.log(probabilities[np.arange(len(truth)), truth].clip(1e-12)).mean())
        history.append({"alpha": alpha, "validation_accuracy": scores["accuracy"], "validation_macro_f1": scores["macro_f1"],
                        "validation_nll": nll})
    best = max(history, key=lambda r: (r["validation_macro_f1"], -r["validation_nll"], -r["alpha"]))
    return best, history


def ensemble(models, audio, text):
    with torch.no_grad():
        return torch.stack([m(audio, text).softmax(1) for m in models]).mean(0).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/minds14-v2/manifest.jsonl")
    parser.add_argument("--features", default="runs/intent-features-v2")
    parser.add_argument("--output", default="runs/intent-v2")
    parser.add_argument("--previous-model", default="evidence/2026-11-19/intent/fusion")
    a = parser.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()): parser.error("Output must be empty")
    rows = read_intent_manifest(a.manifest)
    feature_root = Path(a.features)
    meta = json.loads((feature_root / "metadata.json").read_text())
    if meta["manifest_sha256"] != fingerprint(a.manifest) or meta["features_sha256"] != fingerprint(feature_root / "features.npz"):
        raise ValueError("Feature provenance mismatch")
    with np.load(feature_root / "features.npz", allow_pickle=False) as cache:
        if cache["ids"].tolist() != [r["id"] for r in rows]: raise ValueError("Cache row mismatch")
        audio_np, text_np = cache["audio"].copy(), cache["text"].copy()
    seed_everything(19, 3)
    indices = {s: torch.tensor([i for i, r in enumerate(rows) if r["split"] == s]) for s in ("train", "validation", "test")}
    stats = train_statistics(audio_np, text_np, indices["train"].numpy())
    audio, text, labels = torch.from_numpy(audio_np), torch.from_numpy(text_np), torch.tensor([r["label"] for r in rows])
    args = SimpleNamespace(epochs=80, patience=12, batch_size=32, hidden=128, dropout=.2, learning_rate=.001,
                           weight_decay=.01, threads=3, seed=13)
    out.mkdir(parents=True, exist_ok=True)
    models, development, training, per_seed = {}, {}, {}, {}
    for mode in MODES:
        models[mode], training[mode] = [], []
        for seed in SEEDS:
            args.seed = seed
            model, history, epoch = fit_head(mode, audio, text, labels, indices, stats, args)
            target = out / mode / f"seed_{seed}"; target.mkdir(parents=True)
            save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(target / "head.safetensors"))
            config = {"architecture": {"mode": mode, "audio_dim": audio.shape[1], "text_dim": text.shape[1],
                      "hidden": 128, "classes": 14, "dropout": .2}, "labels": INTENTS, "best_epoch": epoch,
                      "seed": seed, "frozen_encoders": meta, "training": vars(args).copy()}
            dump_json(target / "config.json", config); dump_json(target / "history.json", history)
            restored, _ = load_head(target); models[mode].append(restored)
            training[mode].append({"seed": seed, "best_epoch": epoch, "epochs_run": len(history),
                                   "validation_macro_f1": history[epoch - 1]["validation_macro_f1"]})
            print(mode, training[mode][-1], flush=True)
        dev_idx = indices["validation"]
        development[mode] = ensemble(models[mode], audio[dev_idx], text[dev_idx])
    selected, fusion_candidates = select_fusion(development["text"], development["fusion"], labels[indices["validation"]].numpy())
    public_validation = [r for r in evaluation_index(rows) if r["split"] == "validation"]
    dump_json(out / "predictions_validation.json", [{**r, "probabilities": {mode: values[i].tolist() for mode, values in development.items()}}
                                                     for i, r in enumerate(public_validation)])
    alpha = selected["alpha"]
    # Persist the decision before any test prediction is computed.
    dump_json(out / "selection.json", {"selected": selected, "candidates": fusion_candidates, "test_used": False})
    test_idx = indices["test"]; truth = labels[test_idx].numpy()
    probabilities = {mode: ensemble(models[mode], audio[test_idx], text[test_idx]) for mode in MODES}
    probabilities["selected_blend"] = (1 - alpha) * probabilities["text"] + alpha * probabilities["fusion"]
    old_model, old_config = load_head(a.previous_model)
    if old_config["frozen_encoders"]["audio_revision"] != meta["audio_revision"] or old_config["frozen_encoders"]["text_revision"] != meta["text_revision"]:
        raise ValueError("Earlier checkpoint uses incompatible encoders")
    probabilities["previous_v1_fusion"] = ensemble([old_model], audio[test_idx], text[test_idx])
    public = evaluation_index(rows); public_test = [r for r in public if r["split"] == "test"]
    languages = np.array([r["language"] for r in public_test])
    metrics = {}
    for name, values in probabilities.items():
        predicted = values.argmax(1)
        metrics[name] = {"test": classification_metrics(truth, predicted),
            "by_language": {lang: classification_metrics(truth[languages == lang], predicted[languages == lang]) for lang in sorted(set(languages))}}
        print(name, "accuracy", metrics[name]["test"]["accuracy"], "macro F1", metrics[name]["test"]["macro_f1"], flush=True)
    for mode in MODES:
        per_seed[mode] = []
        for seed, model in zip(SEEDS, models[mode]):
            values = ensemble([model], audio[test_idx], text[test_idx])
            per_seed[mode].append({"seed": seed, **classification_metrics(truth, values.argmax(1))})
    predictions = [{**r, "probabilities": {name: values[i].tolist() for name, values in probabilities.items()}}
                   for i, r in enumerate(public_test)]
    report = {"models": metrics, "training": training, "per_seed_test": per_seed, "selected_audio_text_weight": alpha,
              "counts": {s: len(idx) for s, idx in indices.items()}, "feature_metadata": meta, "environment": environment(),
              "selected_minus_previous": paired_bootstrap(truth, probabilities["selected_blend"].argmax(1), probabilities["previous_v1_fusion"].argmax(1), languages, seed=19),
              "selected_minus_text": paired_bootstrap(truth, probabilities["selected_blend"].argmax(1), probabilities["text"].argmax(1), languages, seed=19),
              "limitations": ["Utterance split; unknown speakers may overlap.", "Dataset-supplied transcripts required.",
                              "All three languages occur in training; no zero-shot language claim.",
                              "Training data size and seed ensembling both changed; gains cannot be attributed to architecture alone.",
                              "A zero fusion weight means validation selected text-only; do not claim multimodal improvement in that case."]}
    dump_json(out / "metrics.json", report); dump_json(out / "predictions_test.json", predictions)
    dump_json(out / "evaluation_index.json", public)
    dump_json(out / "ensemble_config.json", {"alpha": alpha, "seeds": list(SEEDS), "labels": INTENTS,
        "text_models": [f"text/seed_{s}" for s in SEEDS], "fusion_models": [f"fusion/seed_{s}" for s in SEEDS],
        "feature_metadata": meta, "selection": "Validation macro F1 then validation NLL; no test-driven choice"})


if __name__ == "__main__": main()
