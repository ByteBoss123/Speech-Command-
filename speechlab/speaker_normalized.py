"""Fit a regularized within-speaker covariance backend using training speakers."""
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
from .speaker_v2 import MODEL, REVISION, clustered_comparison, enrollment_scores, rates, unit
from .speaker_v3 import compact_trials

METHODS = ("five_raw", "five_centered", "wccn_0.1", "wccn_0.5", "wccn_0.9")


def fit_normalizer(vectors, labels, shrinkage):
    vectors, labels = np.asarray(vectors, dtype=np.float64), np.asarray(labels)
    if vectors.ndim != 2 or len(vectors) != len(labels) or not 0 < shrinkage <= 1:
        raise ValueError("Invalid normalization inputs")
    if not np.isfinite(vectors).all(): raise ValueError("Nonfinite training vectors")
    groups = sorted(set(labels))
    if len(groups) < 2 or any((labels == s).sum() < 2 for s in groups): raise ValueError("Repeated training speakers required")
    residuals = vectors.copy()
    for speaker in groups:
        take = labels == speaker
        residuals[take] -= vectors[take].mean(0)
    covariance = residuals.T @ residuals / (len(vectors) - len(groups))
    variance = float(np.trace(covariance) / covariance.shape[0])
    if variance <= 0: raise ValueError("No within-speaker variation")
    covariance = (1 - shrinkage) * covariance + shrinkage * variance * np.eye(covariance.shape[0])
    values, basis = np.linalg.eigh(covariance)
    transform = (basis * (1 / np.sqrt(values.clip(variance * 1e-8)))) @ basis.T
    return {"mean": vectors.mean(0).astype(np.float32), "transform": transform.astype(np.float32)}


def apply_normalizer(vectors, normalizer):
    return unit((np.asarray(vectors, dtype=np.float32) - normalizer["mean"]) @ normalizer["transform"])


def choose(calibration):
    feasible = [name for name in METHODS if calibration[name]["far"] <= .005]
    if not feasible: raise ValueError("No feasible normalization method")
    return min(feasible, key=lambda name: (calibration[name]["frr"], calibration[name]["far"], METHODS.index(name)))


def score(rows, vectors, center, normalizers):
    result = {"v2_single": enrollment_scores(rows, vectors, "single_raw", center),
              "v2_three": enrollment_scores(rows, vectors, "multi_raw", center, 3),
              "five_raw": enrollment_scores(rows, vectors, "multi_raw", center, 5),
              "five_centered": enrollment_scores(rows, vectors, "multi_centered", center, 5)}
    result.update({name: enrollment_scores(rows, apply_normalizer(vectors, params), "multi_raw", center, 5) for name, params in normalizers.items()})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/librispeech-v3/manifest.jsonl")
    p.add_argument("--cache", default="runs/speaker-v3")
    p.add_argument("--output", default="runs/speaker-normalized-v3")
    a = p.parse_args(); out, cache = Path(a.output), Path(a.cache)
    if out.exists() and any(out.iterdir()): p.error("Output must be empty")
    out.mkdir(parents=True); seed_everything(31, 2)
    dump_json(out / "extension_record.json", {"protocol_sha256": fingerprint("experiments/v3_speaker_extension_protocol.json"),
        "raw_test_score_export_exists_at_start": (cache / "trials_test.json").exists(), "test_scores_read_for_selection": False})
    rows = read_manifest(a.manifest)
    train_speakers = sorted({r["speaker_id"] for r in rows if r["split"] == "train"})
    cohort = [r for s in train_speakers for r in [r for r in rows if r["split"] == "train" and r["speaker_id"] == s][:8]]
    if len(cohort) != 384: raise ValueError("Expected 384 training recordings")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, revision=REVISION)
    model = WavLMForXVector.from_pretrained(MODEL, revision=REVISION).eval().requires_grad_(False).to(device)
    vectors = embed(cohort, extractor, model, device)
    del model
    labels = np.array([r["speaker_id"] for r in cohort])
    center = vectors[np.array([i % 8 < 2 for i in range(len(cohort))])].mean(0)
    normalizers = {f"wccn_{alpha}": fit_normalizer(vectors, labels, alpha) for alpha in (.1, .5, .9)}
    # These are aggregate learned parameters, not per-speaker embeddings.
    for name, params in normalizers.items(): np.savez_compressed(out / f"{name}.npz", **params)
    subset = [r for r in rows if r["split"] == "validation"]
    with np.load(cache / "local_embeddings_validation.npz", allow_pickle=False) as f: dev_vectors = f["vectors"].copy()
    dev = score(subset, dev_vectors, center, normalizers)
    calibration = {name: calibrate_threshold([r["label"] for r in dev[name]], [r["score"] for r in dev[name]], .005) for name in METHODS}
    selected = choose(calibration)
    previous = json.loads(Path("evidence/2026-11-19-v2/speaker/metrics.json").read_text())
    thresholds = {"v2_single": previous["calibration"]["single_raw"]["threshold"], "v2_three": previous["calibration"]["multi_raw"]["threshold"],
                  **{name: values["threshold"] for name, values in calibration.items()}}
    dump_json(out / "trials_validation.json", compact_trials(dev))
    dump_json(out / "selection.json", {"method": selected, "calibration": calibration, "frozen_thresholds": thresholds,
        "test_used": False, "development_far_cap": .005, "baseline_validation": {name: rates(dev[name], thresholds[name]) for name in ("v2_single", "v2_three")}})
    print("Selected normalization on development:", selected, calibration[selected], flush=True)
    # Test vectors are first opened only after the development decision is saved.
    with np.load(cache / "local_embeddings_test.npz", allow_pickle=False) as f: test_vectors = f["vectors"].copy()
    subset = [r for r in rows if r["split"] == "test"]
    test = score(subset, test_vectors, center, normalizers)
    dump_json(out / "trials_test.json", compact_trials(test))
    metrics = {name: {"at_frozen_threshold": rates(trials, thresholds[name]), "descriptive_eer": approximate_eer([r["label"] for r in trials], [r["score"] for r in trials])} for name, trials in test.items()}
    report = {"models": metrics, "selected_method": selected, "thresholds": thresholds, "test_speakers": 30,
        "selected_minus_v2_three": clustered_comparison(test["v2_three"], test[selected], thresholds["v2_three"], thresholds[selected], seed=31),
        "selected_minus_v2_single": clustered_comparison(test["v2_single"], test[selected], thresholds["v2_single"], thresholds[selected], seed=31),
        "manifest_sha256": fingerprint(a.manifest), "environment": environment(), "encoder_training": "none; frozen WavLM",
        "backend_training_recordings": 384, "backend_training_speakers": 48, "enrollment_recordings": 5,
        "limitations": ["Development-driven normalization extension; test scores not used for selection.", "More enrollment recordings increase user effort.",
            "Empirical FAR cap is not a population guarantee; bootstrap conditions on this enrollment cohort.", "Source-prefix audiobook sample; no spoofing evaluation."]}
    dump_json(out / "metrics.json", report)
    config = {"model": MODEL, "revision": REVISION, "method": "wccn" if selected.startswith("wccn") else "multi_raw" if selected == "five_raw" else "multi_centered",
        "threshold": thresholds[selected], "cohort_center": center.tolist(), "enrollment_recordings": 5, "target_calibration_far": .005}
    if selected.startswith("wccn"): config["normalizer_path"] = selected + ".npz"
    dump_json(out / "verification_config.json", config)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__": main()
