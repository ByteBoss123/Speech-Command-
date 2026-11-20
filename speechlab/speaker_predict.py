"""Compare the configured enrollment recordings with a local probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from .common import RATE, audio_array, dump_json, seed_everything
from .speaker import embed
from .speaker_v2 import MODEL, REVISION, unit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enroll", nargs="+", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--config", default="evidence/2026-11-19-v2/speaker/verification_config.json")
    p.add_argument("--output")
    a = p.parse_args()
    if a.output and Path(a.output).exists(): p.error("Output already exists")
    config = json.loads(Path(a.config).read_text())
    if config["model"] != MODEL or config["revision"] != REVISION: raise ValueError("Encoder mismatch")
    enrollment_count = config["enrollment_recordings"]
    if len(a.enroll) != enrollment_count: p.error(f"This configuration requires {enrollment_count} enrollment recordings")
    paths = a.enroll + [a.probe]
    if len({str(Path(x).resolve()) for x in paths}) != enrollment_count + 1: p.error("Enrollment and probe files must be distinct")
    if any(not 2 <= len(audio_array(path)) / RATE <= 20 for path in paths): p.error("Recordings must be 2–20 seconds")
    seed_everything(19, 3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, revision=REVISION)
    model = WavLMForXVector.from_pretrained(MODEL, revision=REVISION).eval().requires_grad_(False).to(device)
    vectors = embed([{"absolute_path": str(Path(path).resolve())} for path in paths], extractor, model, device)
    if config["method"] == "multi_centered": vectors = unit(vectors - np.array(config["cohort_center"]))
    chosen = vectors[:1] if config["method"] == "single_raw" else vectors[:enrollment_count]
    reference = unit(chosen.mean(0, keepdims=True))[0]
    score = float(reference @ vectors[enrollment_count])
    result = {"similarity": score, "threshold": config["threshold"], "match": score >= config["threshold"],
              "method": config["method"], "scope": "Research demonstration; not a spoof-resistant authentication system"}
    if a.output: dump_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
