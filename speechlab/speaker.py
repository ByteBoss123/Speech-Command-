"""Speaker verification with a frozen WavLM model and development-only calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from .common import RATE, audio_array, dump_json, environment, fingerprint, read_manifest, seed_everything
from .metrics import approximate_eer, calibrate_threshold, make_trials, verification_rates


def embed(rows, extractor, model, device):
    vectors = []
    model.eval()
    # Individual clips avoid contamination of temporal statistics by padded samples.
    for i, row in enumerate(rows):
        signal = audio_array(row["absolute_path"])
        inputs = extractor(signal, sampling_rate=RATE, return_tensors="pt", padding=False)
        with torch.no_grad():
            vector = model(**{k: v.to(device) for k, v in inputs.items()}).embeddings
            vectors.append(torch.nn.functional.normalize(vector, dim=-1)[0].cpu().numpy())
        if (i + 1) % 12 == 0 or i == len(rows) - 1:
            print(f"Speaker embeddings {i+1}/{len(rows)}", flush=True)
    return np.stack(vectors)


def score_trials(rows, vectors, seed):
    return [{"left_id": rows[i]["id"], "right_id": rows[j]["id"],
             "left_speaker": rows[i]["speaker_id"], "right_speaker": rows[j]["speaker_id"],
             "same_chapter": rows[i]["chapter_id"] == rows[j]["chapter_id"],
             "label": label, "score": float(np.dot(vectors[i], vectors[j]))}
            for i, j, label in make_trials(rows, seed)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/librispeech-pilot/manifest.jsonl")
    p.add_argument("--output", default="runs/speaker-pilot")
    p.add_argument("--model", default="microsoft/wavlm-base-plus-sv")
    p.add_argument("--revision", default=None)
    p.add_argument("--target-far", type=float, default=.05)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a = p.parse_args()
    if not 0 <= a.target_far <= 1 or a.threads < 1:
        p.error("Invalid target FAR or thread count")
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        p.error("Output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(a.seed, a.threads)
    rows = read_manifest(a.manifest)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    revision = HfApi().model_info(a.model, revision=a.revision).sha
    dump_json(out / "run_config.json", {**vars(a), "revision": revision, "environment": environment(),
                                      "manifest_sha256": fingerprint(a.manifest), "device": device})
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(a.model, revision=revision)
    model = WavLMForXVector.from_pretrained(a.model, revision=revision).to(device)
    scored = {}
    # Calibration is completed before test scoring. Test never changes the threshold.
    calibration = None
    for split in ["validation", "test"]:
        subset = [r for r in rows if r["split"] == split]
        vectors = embed(subset, extractor, model, device)
        scored[split] = score_trials(subset, vectors, a.seed)
        dump_json(out / f"trials_{split}.json", scored[split])
        if split == "validation":
            calibration = calibrate_threshold([r["label"] for r in scored[split]], [r["score"] for r in scored[split]], a.target_far)
            dump_json(out / "calibration.json", calibration)
    test = scored["test"]
    labels, scores = [r["label"] for r in test], [r["score"] for r in test]
    report = {"status": "completed", "model": a.model, "revision": revision, "training": "none; frozen pretrained model",
              "calibration": calibration,
              "test_at_frozen_threshold": verification_rates(labels, scores, calibration["threshold"]),
              "test_eer_descriptive_only": approximate_eer(labels, scores),
              "test_speakers": len({r["speaker_id"] for r in rows if r["split"] == "test"}),
              "test_positive_same_chapter_fraction": float(np.mean([r["same_chapter"] for r in test if r["label"] == 1])),
              "limitations": ["Audiobook speaker verification, not speaker enrollment for authentication or Siri intent detection.",
                              "Positive pairs can share recording conditions; easier than cross-session trials.",
                              "Trials share utterances and speakers and are not statistically independent.",
                              "FAR target is an empirical calibration constraint, not a guaranteed test FAR.",
                              "Base-model pretraining overlap is unknown; no claim of unseen foundation-model speakers."]}
    dump_json(out / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
