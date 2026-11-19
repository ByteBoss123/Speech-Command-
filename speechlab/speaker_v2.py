"""Cross-chapter speaker verification with multi-recording enrollment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from .common import dump_json, environment, fingerprint, read_manifest, seed_everything
from .metrics import approximate_eer, calibrate_threshold, verification_rates
from .speaker import embed

MODEL = "microsoft/wavlm-base-plus-sv"
REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
METHODS = ("single_raw", "multi_raw", "multi_centered")


def unit(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-12)


def enrollment_scores(rows, vectors, method, center):
    if method not in METHODS: raise ValueError("Unknown verification method")
    speakers = sorted({r["speaker_id"] for r in rows})
    aliases = {s: f"speaker_{i:03d}" for i, s in enumerate(speakers)}
    transformed = unit(vectors - center) if method == "multi_centered" else unit(vectors)
    enrollment, chapter_sets = [], {}
    for speaker in speakers:
        ids = [i for i, r in enumerate(rows) if r["speaker_id"] == speaker and r["verification_role"] == "enroll"]
        if len(ids) < 3: raise ValueError("Three enrollment recordings required")
        chapter_sets[speaker] = {rows[i]["chapter_id"] for i in ids}
        use = ids[:1] if method == "single_raw" else ids
        enrollment.append(unit(transformed[use].mean(0, keepdims=True))[0])
    enrollment = np.stack(enrollment)
    trials = []
    for i, row in enumerate(rows):
        if row["verification_role"] != "probe": continue
        if row["chapter_id"] in chapter_sets[row["speaker_id"]]:
            raise ValueError("Enrollment/probe chapter leakage")
        scores = enrollment @ transformed[i]
        for j, speaker in enumerate(speakers):
            trials.append({"probe_id": f"example_{i:05d}", "probe_speaker": aliases[row["speaker_id"]],
                           "enrollment_speaker": aliases[speaker], "label": int(speaker == row["speaker_id"]),
                           "cross_chapter_positive": speaker == row["speaker_id"], "score": float(scores[j])})
    return trials


def rates(trials, threshold):
    return verification_rates([t["label"] for t in trials], [t["score"] for t in trials], threshold)


def clustered_comparison(baseline, selected, base_threshold, selected_threshold, seed=19, samples=2000):
    if len(baseline) != len(selected) or any((a["probe_id"], a["enrollment_speaker"], a["label"]) !=
                                         (b["probe_id"], b["enrollment_speaker"], b["label"]) for a, b in zip(baseline, selected)):
        raise ValueError("Trial pairing mismatch")
    groups = sorted({r["probe_speaker"] for r in baseline})
    counts = []
    for speaker in groups:
        indices = [i for i, r in enumerate(baseline) if r["probe_speaker"] == speaker]
        a = rates([baseline[i] for i in indices], base_threshold)
        b = rates([selected[i] for i in indices], selected_threshold)
        counts.append([a["false_accepts"], a["false_rejects"], b["false_accepts"], b["false_rejects"], a["negative_trials"], a["positive_trials"]])
    counts = np.array(counts)
    rng = np.random.default_rng(seed)
    totals = counts[rng.integers(len(groups), size=(samples, len(groups)))].sum(1)
    return {"frr_difference_ci95": np.quantile((totals[:, 3] - totals[:, 1]) / totals[:, 5], [.025, .975]).tolist(),
            "far_difference_ci95": np.quantile((totals[:, 2] - totals[:, 0]) / totals[:, 4], [.025, .975]).tolist(),
            "resampling_unit": "probe speaker", "samples": samples, "seed": seed,
            "limitation": "Conditional on this enrollment speaker cohort; shared enrollment dependence is not resampled."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/librispeech-v2/manifest.jsonl")
    p.add_argument("--output", default="runs/speaker-v2")
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--target-far", type=float, default=.01)
    args = p.parse_args()
    if not 0 <= args.target_far <= 1: p.error("Invalid target FAR")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()): p.error("Output must be empty")
    out.mkdir(parents=True, exist_ok=True); seed_everything(19, args.threads)
    rows = read_manifest(args.manifest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, revision=REVISION)
    model = WavLMForXVector.from_pretrained(MODEL, revision=REVISION).eval().requires_grad_(False).to(device)
    train_speakers = sorted({r["speaker_id"] for r in rows if r["split"] == "train"})
    cohort = [r for speaker in train_speakers for r in [x for x in rows if x["split"] == "train" and x["speaker_id"] == speaker][:2]]
    center = embed(cohort, extractor, model, device).mean(0)
    calibration, trials_by_split = {}, {}
    for split in ("validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        vectors = embed(subset, extractor, model, device)
        # Keep per-record embeddings local, not in the publication bundle.
        np.savez_compressed(out / f"local_embeddings_{split}.npz", vectors=vectors)
        scored = {method: enrollment_scores(subset, vectors, method, center) for method in METHODS}
        trials_by_split[split] = scored
        dump_json(out / f"trials_{split}.json", scored)
        if split == "validation":
            for method, trials in scored.items():
                calibration[method] = calibrate_threshold([r["label"] for r in trials], [r["score"] for r in trials], args.target_far)
            # Methods are fixed in advance. Threshold and method selection use dev only.
            selected = min(METHODS, key=lambda method: (calibration[method]["frr"], calibration[method]["far"], METHODS.index(method)))
            dump_json(out / "selection.json", {"method": selected, "calibration": calibration, "test_used": False})
            print("Selected on development:", selected, calibration[selected], flush=True)
    test = trials_by_split["test"]
    metrics = {method: {"at_frozen_threshold": rates(trials, calibration[method]["threshold"]),
               "descriptive_eer": approximate_eer([r["label"] for r in trials], [r["score"] for r in trials])} for method, trials in test.items()}
    comparison = clustered_comparison(test["single_raw"], test[selected], calibration["single_raw"]["threshold"], calibration[selected]["threshold"])
    report = {"models": metrics, "selected_method": selected, "calibration": calibration, "paired_selected_minus_single": comparison,
              "test_speakers": len({r["speaker_id"] for r in rows if r["split"] == "test"}),
              "positive_cross_chapter_fraction": 1., "cohort_recordings": len(cohort), "cohort_speakers": len(train_speakers),
              "encoder_training": "none; frozen pretrained WavLM", "environment": environment(), "manifest_sha256": fingerprint(args.manifest),
              "limitations": ["Cross-chapter does not guarantee cross-session or cross-microphone recording.",
                "Trials reuse enrollment/probe speakers; not independent authentication attempts.",
                "FAR target is empirical on calibration, not a guaranteed operating bound.",
                "Foundation pretraining overlap unknown; no spoofing or production authentication evaluation."]}
    dump_json(out / "metrics.json", report)
    dump_json(out / "verification_config.json", {"model": MODEL, "revision": REVISION, "method": selected,
        "threshold": calibration[selected]["threshold"], "cohort_center": center.tolist(), "enrollment_recordings": 3,
        "target_calibration_far": args.target_far})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__": main()
