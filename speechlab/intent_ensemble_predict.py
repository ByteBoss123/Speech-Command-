"""Run the validation-selected v2 intent ensemble on local audio and text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import dump_json, seed_everything
from .intent import load_head
from .intent_features import (AUDIO_REVISION, TEXT_REVISION, encode_audio, encode_text,
                              load_audio_encoder, load_text_encoder)
from .intent_v2 import ensemble


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio")
    p.add_argument("--transcript", required=True)
    p.add_argument("--model", default="evidence/2026-11-19-v2/intent")
    p.add_argument("--output")
    a = p.parse_args()
    if not a.transcript.strip(): p.error("Transcript required")
    if a.output and Path(a.output).exists(): p.error("Output already exists")
    seed_everything(19, 3)
    root = Path(a.model); config = json.loads((root / "ensemble_config.json").read_text())
    metadata = config["feature_metadata"]
    if metadata["audio_revision"] != AUDIO_REVISION or metadata["text_revision"] != TEXT_REVISION:
        raise ValueError("Encoder revision mismatch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = config["alpha"]
    if alpha > 0:
        extractor, encoder = load_audio_encoder(device)
        audio = encode_audio([a.audio], extractor, encoder, device)
        del encoder
    else:
        audio = np.zeros((1, metadata["audio_dim"]), dtype=np.float32)
    tokenizer, encoder = load_text_encoder(device)
    text = encode_text([a.transcript], tokenizer, encoder, device)
    audio, text = torch.from_numpy(audio), torch.from_numpy(text)
    probs = np.zeros((1, len(config["labels"])))
    if alpha < 1:
        models = [load_head(root / path)[0] for path in config["text_models"]]
        probs += (1 - alpha) * ensemble(models, audio, text)
    if alpha > 0:
        models = [load_head(root / path)[0] for path in config["fusion_models"]]
        probs += alpha * ensemble(models, audio, text)
    top = np.argsort(probs[0])[::-1][:3]
    result = {"top_intents": [{"intent": config["labels"][int(i)], "probability": float(probs[0, i])} for i in top],
              "fusion_weight": alpha, "uses_audio": alpha > 0,
              "scope": "Research closed-set classifier with supplied text; uncalibrated probabilities; no unknown-intent rejection"}
    if a.output: dump_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
