"""Run an exported fused intent head on a recording and supplied transcript."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .common import dump_json, fingerprint, seed_everything
from .intent import load_head
from .intent_features import (AUDIO_MODEL, AUDIO_REVISION, TEXT_MODEL, TEXT_REVISION,
                              encode_audio, encode_text, load_audio_encoder, load_text_encoder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="One local recording, 1–30 seconds")
    parser.add_argument("--transcript", required=True, help="Native-language text corresponding to the recording")
    parser.add_argument("--model", default="evidence/2026-11-19/intent/fusion")
    parser.add_argument("--output", help="Optional JSON file, must not exist")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        parser.error("Output already exists")
    seed_everything()
    head, config = load_head(args.model)
    if config["architecture"]["mode"] != "fusion":
        parser.error("This command requires a fused head")
    provenance = config["frozen_encoders"]
    expected = {"audio_model": AUDIO_MODEL, "audio_revision": AUDIO_REVISION,
                "text_model": TEXT_MODEL, "text_revision": TEXT_REVISION}
    if any(provenance[k] != v for k, v in expected.items()):
        raise ValueError("Encoder configuration does not match checkpoint")
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor, audio_encoder = load_audio_encoder(device)
    audio_features = encode_audio([args.audio], extractor, audio_encoder, device)
    del audio_encoder
    tokenizer, text_encoder = load_text_encoder(device)
    text_features = encode_text([args.transcript], tokenizer, text_encoder, device)
    with torch.no_grad():
        probabilities = head(torch.from_numpy(audio_features), torch.from_numpy(text_features)).softmax(1)[0]
    top = torch.argsort(probabilities, descending=True)[:3].tolist()
    result = {"top_intents": [{"intent": config["labels"][i], "probability": probabilities[i].item()} for i in top],
              "audio_sha256": fingerprint(args.audio), "head_sha256": fingerprint(Path(args.model) / "head.safetensors"),
              "supplied_transcript": args.transcript, "seconds_including_model_load": time.perf_counter() - start,
              "scope": "Closed-set banking intent prediction; requires both modalities; probabilities are not calibrated."}
    if args.output:
        dump_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
