"""Evaluate a fixed larger pretrained ASR candidate; select capacity on dev WER."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .asr import predict
from .asr_v2 import compact_predictions
from .common import dump_json, environment, fingerprint, read_manifest, seed_everything
from .metrics import asr_summary

MODEL = "openai/whisper-base"
REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"


def choose_model(development_scores):
    """Input contains validation WER only; smaller candidate wins exact ties."""
    return min(development_scores, key=lambda name: (development_scores[name], name != "tiny_selected"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/librispeech-v2/manifest.jsonl")
    p.add_argument("--output", default="runs/asr-capacity-v2")
    a = p.parse_args(); out = Path(a.output)
    if out.exists() and any(out.iterdir()): p.error("Output must be empty")
    out.mkdir(parents=True, exist_ok=True); seed_everything(19, 3)
    rows = read_manifest(a.manifest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL, revision=REVISION, language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, attn_implementation="sdpa").to(device).eval()
    model.generation_config.forced_decoder_ids = None; model.config.forced_decoder_ids = None
    dev = compact_predictions(predict(model, processor, [r for r in rows if r["split"] == "validation"], device, seed=19))
    dump_json(out / "predictions_validation.json", dev)
    dump_json(out / "selection_input.json", {"model": MODEL, "validation_wer": asr_summary(dev)["wer"], "test_used": False})
    results = {}
    for name, snr in (("clean", None), ("gaussian_noise_10db", 10.)):
        predictions = compact_predictions(predict(model, processor, [r for r in rows if r["split"] == "test"], device, noise_snr=snr, seed=19))
        dump_json(out / f"predictions_{name}.json", predictions)
        results[name] = asr_summary(predictions)
    dump_json(out / "metrics.json", {"model": MODEL, "revision": REVISION, "training": "none; pretrained capacity candidate",
        "parameters": sum(p.numel() for p in model.parameters()), "validation_wer": asr_summary(dev)["wer"], "results": results,
        "manifest_sha256": fingerprint(a.manifest), "environment": environment(),
        "selection_rule": "Compare this validation WER with selected tiny checkpoint validation WER; smaller model wins ties"})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__": main()
