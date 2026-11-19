"""Train and evaluate audio, text, and fused multilingual intent classifiers."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .common import dump_json, environment, fingerprint, seed_everything
from .intent_data import INTENTS, read_intent_manifest

MODES = ("audio", "text", "fusion")


def evaluation_index(rows):
    """Publish split/label evidence without transcripts or recording provenance."""
    return [{"id": f"example_{i:05d}", "split": r["split"], "language": r["language"],
             "label": r["label"]} for i, r in enumerate(rows)]


def train_statistics(audio, text, train_indices):
    """Fit normalization exclusively on explicitly supplied training indices."""
    if len(train_indices) == 0:
        raise ValueError("Training rows are required")
    stats = {}
    for name, values in (("audio", audio), ("text", text)):
        selected = values[train_indices]
        stats[name + "_mean"] = torch.as_tensor(selected.mean(0), dtype=torch.float32)
        stats[name + "_scale"] = torch.as_tensor(selected.std(0).clip(1e-5), dtype=torch.float32)
    return stats


class IntentHead(nn.Module):
    def __init__(self, mode, audio_dim, text_dim, hidden=128, classes=14, dropout=0.2, statistics=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Unknown input mode")
        self.mode = mode
        for name, dim in (("audio", audio_dim), ("text", text_dim)):
            self.register_buffer(name + "_mean", torch.zeros(dim))
            self.register_buffer(name + "_scale", torch.ones(dim))
        if statistics is not None:
            for key, value in statistics.items():
                getattr(self, key).copy_(value)
        inputs = audio_dim if mode == "audio" else text_dim if mode == "text" else audio_dim + text_dim
        self.classifier = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, classes))

    def forward(self, audio, text, drop=None):
        if drop not in (None, "audio", "text"):
            raise ValueError("Unknown ablation")
        audio = (audio - self.audio_mean) / self.audio_scale
        text = (text - self.text_mean) / self.text_scale
        # Zero means a training-mean embedding, not a silent waveform or blank text.
        if drop == "audio":
            audio = torch.zeros_like(audio)
        elif drop == "text":
            text = torch.zeros_like(text)
        values = audio if self.mode == "audio" else text if self.mode == "text" else torch.cat([audio, text], dim=-1)
        return self.classifier(values)


def classification_metrics(truth, predicted, classes=14):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.ndim != 1 or predicted.shape != truth.shape or not len(truth):
        raise ValueError("Nonempty equal-length label vectors required")
    if np.any(truth < 0) or np.any(truth >= classes) or np.any(predicted < 0) or np.any(predicted >= classes):
        raise ValueError("Labels out of range")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    tp = np.diag(confusion)
    support, predicted_count = confusion.sum(1), confusion.sum(0)
    precision = np.divide(tp, predicted_count, out=np.zeros(classes), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted_count, out=np.zeros(classes), where=(support + predicted_count) != 0)
    return {"n": len(truth), "correct": int(tp.sum()), "accuracy": float(tp.sum() / len(truth)),
            "macro_f1": float(f1.mean()), "confusion_matrix": confusion.tolist(),
            "per_class": [{"label": i, "support": int(support[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i])} for i in range(classes)]}


def paired_bootstrap(truth, left, right, languages, samples=2000, seed=13):
    """Descriptive language-stratified utterance bootstrap of left minus right."""
    truth, left, right, languages = map(np.asarray, (truth, left, right, languages))
    if not len(truth) or not all(len(v) == len(truth) for v in (left, right, languages)):
        raise ValueError("Bootstrap inputs must align")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(languages == lang) for lang in sorted(set(languages))]
    values = []
    for _ in range(samples):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        values.append(float(np.mean(left[idx] == truth[idx]) - np.mean(right[idx] == truth[idx])))
    return {"accuracy_difference": float(np.mean(left == truth) - np.mean(right == truth)),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(), "samples": samples,
            "resampling_unit": "utterance within language", "seed": seed,
            "limitation": "Speaker IDs unavailable; dependence between a speaker's utterances is not accounted for."}


def fit_head(mode, audio, text, labels, indices, stats, args):
    seed_everything(args.seed, args.threads)
    model = IntentHead(mode, audio.shape[1], text.shape[1], args.hidden, len(INTENTS), args.dropout, stats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_idx, val_idx = indices["train"], indices["validation"]
    best, best_state, best_epoch, stale, history = (-1.0, -float("inf")), None, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = train_idx[torch.randperm(len(train_idx))]
        loss_sum = 0.0
        for batch in shuffled.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(audio[batch], text[batch]), labels[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            logits = model(audio[val_idx], text[val_idx])
            val_loss = nn.functional.cross_entropy(logits, labels[val_idx]).item()
            val = classification_metrics(labels[val_idx].numpy(), logits.argmax(1).numpy())
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_idx),
                        "validation_loss": val_loss, "validation_accuracy": val["accuracy"], "validation_macro_f1": val["macro_f1"]})
        score = (val["macro_f1"], -val_loss)
        if score > best:
            best, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def load_head(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    model = IntentHead(**config["architecture"])
    model.load_state_dict(load_file(str(directory / "head.safetensors")))
    return model.eval(), config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/minds14-pilot/manifest.jsonl")
    p.add_argument("--features", default="runs/intent-features")
    p.add_argument("--output", default="runs/intent-pilot")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.hidden, args.threads) < 1 or not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        p.error("Invalid training configuration")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error("Output directory must be empty")
    rows = read_intent_manifest(args.manifest)
    features = Path(args.features)
    meta = json.loads((features / "metadata.json").read_text())
    if meta["manifest_sha256"] != fingerprint(args.manifest) or meta["features_sha256"] != fingerprint(features / "features.npz"):
        raise ValueError("Feature provenance mismatch")
    with np.load(features / "features.npz", allow_pickle=False) as cache:
        if cache["ids"].tolist() != [r["id"] for r in rows]:
            raise ValueError("Feature rows do not match manifest")
        audio_np, text_np = cache["audio"].copy(), cache["text"].copy()
    if any(v.ndim != 2 or len(v) != len(rows) or not np.isfinite(v).all() for v in (audio_np, text_np)):
        raise ValueError("Invalid feature matrix")
    seed_everything(args.seed, args.threads)
    torch.use_deterministic_algorithms(True)
    indices = {s: torch.tensor([i for i, r in enumerate(rows) if r["split"] == s]) for s in ("train", "validation", "test")}
    stats = train_statistics(audio_np, text_np, indices["train"].numpy())
    audio, text = torch.from_numpy(audio_np), torch.from_numpy(text_np)
    labels = torch.tensor([r["label"] for r in rows])
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    models, results = {}, {}
    # Complete training and checkpoint selection for every mode before test scoring.
    for mode in MODES:
        model, history, epoch = fit_head(mode, audio, text, labels, indices, stats, args)
        target = out / mode
        target.mkdir()
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(target / "head.safetensors"))
        config = {"architecture": {"mode": mode, "audio_dim": audio.shape[1], "text_dim": text.shape[1],
                                   "hidden": args.hidden, "classes": len(INTENTS), "dropout": args.dropout},
                  "labels": INTENTS, "training": vars(args), "best_epoch": epoch,
                  "selection": "Highest validation macro F1; lower validation cross-entropy breaks ties",
                  "normalization": "Training rows only; means/scales stored in head checkpoint buffers",
                  "trained_head_parameters": sum(p.numel() for p in model.parameters()),
                  "frozen_encoders": meta, "supported_training_languages": sorted({r["language"] for r in rows})}
        dump_json(target / "config.json", config)
        dump_json(target / "history.json", history)
        models[mode], _ = load_head(target)
        results[mode] = {"best_epoch": epoch, "epochs_run": len(history), "trained_parameters": config["trained_head_parameters"]}
        print(f"{mode}: selected epoch {epoch}/{len(history)}, validation macro F1 {history[epoch - 1]['validation_macro_f1']:.4f}", flush=True)
    test_idx = indices["test"]
    truth = labels[test_idx].numpy()
    test_rows = [rows[i] for i in test_idx.tolist()]
    public_index = evaluation_index(rows)
    public_test_rows = [r for r in public_index if r["split"] == "test"]
    languages = np.array([r["language"] for r in test_rows])
    predictions = []
    probs = {}
    with torch.no_grad():
        for mode, model in models.items():
            probs[mode] = model(audio[test_idx], text[test_idx]).softmax(1).numpy()
            predicted = probs[mode].argmax(1)
            results[mode]["test"] = classification_metrics(truth, predicted)
            results[mode]["by_language"] = {lang: classification_metrics(truth[languages == lang], predicted[languages == lang]) for lang in sorted(set(languages))}
        ablations = {}
        for drop in ("audio", "text"):
            prediction = models["fusion"](audio[test_idx], text[test_idx], drop=drop).argmax(1).numpy()
            ablations[drop + "_replaced_with_training_mean"] = {"metrics": classification_metrics(truth, prediction), "predicted_labels": prediction.tolist()}
    for i, row in enumerate(test_rows):
        predictions.append({"id": public_test_rows[i]["id"], "language": row["language"], "true_label": row["label"],
                            "true_intent": INTENTS[row["label"]], "predictions": {mode: {
                                "label": int(probs[mode][i].argmax()), "intent": INTENTS[int(probs[mode][i].argmax())],
                                "probabilities": probs[mode][i].tolist()} for mode in MODES}})
    comparison = paired_bootstrap(truth, probs["fusion"].argmax(1), probs["text"].argmax(1), languages, seed=args.seed)
    report = {"models": results, "fusion_minus_text": comparison, "fusion_ablations": ablations,
              "counts": {s: len(i) for s, i in indices.items()}, "labels": INTENTS,
              "config": vars(args), "manifest_sha256": fingerprint(args.manifest),
              "features_metadata": meta, "seconds_training_evaluation": time.perf_counter() - start,
              "environment": environment(), "limitations": [
                  "Three languages are present in training; this is not zero-shot cross-lingual transfer.",
                  "Utterance split only; missing speaker IDs prevent speaker-disjoint evaluation.",
                  "Inputs include dataset-provided transcription, not ASR produced by this pipeline.",
                  "Both pretrained encoders are frozen; only classification heads are trained.",
                  "Balanced 126-utterance test, one training seed, and possible pretraining overlap limit generalization claims.",
                  "Intent classification does not evaluate Siri-directed speech, out-of-domain rejection or production latency.",
                  "Ablations are descriptive tests of the selected fused model, not new trained missing-modality models."]}
    dump_json(out / "metrics.json", report)
    dump_json(out / "predictions_test.json", predictions)
    dump_json(out / "evaluation_index.json", public_index)
    for mode in MODES:
        print(f"{mode}: test accuracy {results[mode]['test']['accuracy']:.4f}, macro F1 {results[mode]['test']['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
