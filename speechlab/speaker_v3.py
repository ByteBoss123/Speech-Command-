"""Use more enrollment speech and a stricter development false-accept budget."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from .common import dump_json, environment, fingerprint, read_manifest, seed_everything
from .metrics import approximate_eer, calibrate_threshold
from .speaker import embed
from .speaker_v2 import MODEL, REVISION, clustered_comparison, enrollment_scores, rates

CANDIDATES = ("five_raw", "five_centered")


def select(calibration, cap=.005):
    feasible = [name for name in CANDIDATES if calibration[name]["far"] <= cap]
    if not feasible: raise ValueError("No method satisfies development FAR budget")
    return min(feasible, key=lambda name: (calibration[name]["frr"], calibration[name]["far"], CANDIDATES.index(name)))


def compact_trials(scored):
    names = list(scored); primary = scored[names[0]]
    for name in names[1:]:
        if len(scored[name]) != len(primary) or any((a["probe_id"], a["enrollment_speaker"], a["label"]) !=
                (b["probe_id"], b["enrollment_speaker"], b["label"]) for a, b in zip(primary, scored[name])):
            raise ValueError("Mismatched score rows")
    return [{k: row[k] for k in ("probe_id", "probe_speaker", "enrollment_speaker", "label")} |
            {"scores": {name: scored[name][i]["score"] for name in names}} for i, row in enumerate(primary)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/librispeech-v3/manifest.jsonl")
    p.add_argument("--output", default="runs/speaker-v3")
    p.add_argument("--previous", default="evidence/2026-11-19-v2/speaker")
    a = p.parse_args(); out = Path(a.output)
    if out.exists() and any(out.iterdir()): p.error("Output must be empty")
    out.mkdir(parents=True); seed_everything(31, 2)
    rows = read_manifest(a.manifest)
    previous = json.loads((Path(a.previous) / "metrics.json").read_text())
    thresholds = {"v2_single": previous["calibration"]["single_raw"]["threshold"],
                  "v2_three": previous["calibration"]["multi_raw"]["threshold"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, revision=REVISION)
    model = WavLMForXVector.from_pretrained(MODEL, revision=REVISION).eval().requires_grad_(False).to(device)
    train_speakers = sorted({r["speaker_id"] for r in rows if r["split"] == "train"})
    cohort = [r for s in train_speakers for r in [r for r in rows if r["speaker_id"] == s and r["split"] == "train"][:2]]
    center = embed(cohort, extractor, model, device).mean(0)
    calibration = {}
    for split in ("validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        vectors = embed(subset, extractor, model, device)
        np.savez_compressed(out / f"local_embeddings_{split}.npz", vectors=vectors)
        scored = {"v2_single": enrollment_scores(subset, vectors, "single_raw", center),
                  "v2_three": enrollment_scores(subset, vectors, "multi_raw", center, 3),
                  "five_raw": enrollment_scores(subset, vectors, "multi_raw", center, 5),
                  "five_centered": enrollment_scores(subset, vectors, "multi_centered", center, 5)}
        dump_json(out / f"trials_{split}.json", compact_trials(scored))
        if split == "validation":
            for name in CANDIDATES:
                trials = scored[name]
                calibration[name] = calibrate_threshold([r["label"] for r in trials], [r["score"] for r in trials], .005)
                thresholds[name] = calibration[name]["threshold"]
            selected = select(calibration)
            dump_json(out / "selection.json", {"method": selected, "calibration": calibration,
                "frozen_thresholds": thresholds, "test_used": False, "development_far_cap": .005,
                "baseline_validation": {name: rates(scored[name], thresholds[name]) for name in ("v2_single", "v2_three")}})
            print("Selected development operating point:", selected, calibration[selected], flush=True)
        else:
            metrics = {name: {"at_frozen_threshold": rates(trials, thresholds[name]),
                       "descriptive_eer": approximate_eer([r["label"] for r in trials], [r["score"] for r in trials])}
                       for name, trials in scored.items()}
            report = {"models": metrics, "selected_method": selected, "thresholds": thresholds,
                "selected_minus_v2_three": clustered_comparison(scored["v2_three"], scored[selected], thresholds["v2_three"], thresholds[selected], seed=31),
                "selected_minus_v2_single": clustered_comparison(scored["v2_single"], scored[selected], thresholds["v2_single"], thresholds[selected], seed=31),
                "test_speakers": len({r["speaker_id"] for r in subset}), "manifest_sha256": fingerprint(a.manifest),
                "environment": environment(), "encoder_training": "none; frozen WavLM", "enrollment_recordings": 5,
                "limitations": ["More enrollment recordings increase user effort.", "Empirical development FAR cap is not a population guarantee.",
                    "Bootstrap conditions on this enrollment cohort.", "Source-prefix audiobook sample; pretraining overlap unknown; no spoofing evaluation."]}
            dump_json(out / "metrics.json", report)
            print(json.dumps(metrics, indent=2), flush=True)
    dump_json(out / "verification_config.json", {"model": MODEL, "revision": REVISION,
        "method": "multi_raw" if selected == "five_raw" else "multi_centered", "threshold": thresholds[selected],
        "cohort_center": center.tolist(), "enrollment_recordings": 5, "target_calibration_far": .005})


if __name__ == "__main__": main()
