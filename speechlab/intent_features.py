"""Extract frozen audio and multilingual text features for paired intent routing."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

from .common import RATE, audio_array, dump_json, environment, fingerprint, seed_everything
from .intent_data import read_intent_manifest

AUDIO_MODEL = "openai/whisper-tiny"
AUDIO_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MAX_TOKENS = 128


def pool_audio(hidden, lengths):
    """Mean/std over valid output frames, excluding 30-second padding frames."""
    if hidden.ndim != 3 or len(lengths) != len(hidden):
        raise ValueError("Expected one valid-frame length per recording")
    result = []
    for value, length in zip(hidden, lengths):
        if not 1 <= int(length) <= value.shape[0]:
            raise ValueError("Invalid audio frame count")
        valid = value[:int(length)]
        result.append(torch.cat([valid.mean(0), valid.std(0, unbiased=False)]))
    return torch.stack(result)


def pool_text(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


def load_audio_encoder(device):
    extractor = WhisperFeatureExtractor.from_pretrained(AUDIO_MODEL, revision=AUDIO_REVISION)
    encoder = WhisperModel.from_pretrained(AUDIO_MODEL, revision=AUDIO_REVISION).encoder
    return extractor, encoder.eval().requires_grad_(False).to(device)


def load_text_encoder(device):
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION)
    encoder = AutoModel.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION)
    return tokenizer, encoder.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def encode_audio(paths, extractor, encoder, device):
    audio = [audio_array(p) for p in paths]
    if any(not 1 <= len(a) / RATE <= 30 for a in audio):
        raise ValueError("Each recording must be between 1 and 30 seconds")
    features = extractor(audio, sampling_rate=RATE, return_tensors="pt").input_features.to(device)
    hidden = encoder(features).last_hidden_state
    # Whisper uses a 10 ms log-Mel hop followed by a stride-two convolution.
    lengths = [min(hidden.shape[1], (len(a) + 319) // 320) for a in audio]
    return pool_audio(hidden, lengths).float().cpu().numpy()


@torch.inference_mode()
def encode_text(texts, tokenizer, encoder, device):
    if any(not text.strip() for text in texts):
        raise ValueError("Nonempty supplied transcripts are required")
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)
    hidden = encoder(**tokens).last_hidden_state
    return pool_text(hidden, tokens["attention_mask"]).float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/minds14-pilot/manifest.jsonl")
    parser.add_argument("--output", default="runs/intent-features")
    parser.add_argument("--audio-batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.audio_batch_size, args.text_batch_size, args.threads) < 1:
        parser.error("Batch sizes and thread count must be positive")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be empty")
    rows = read_intent_manifest(args.manifest)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(13, args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    extractor, encoder = load_audio_encoder(device)
    audio = []
    for i in range(0, len(rows), args.audio_batch_size):
        audio.append(encode_audio([r["absolute_path"] for r in rows[i:i + args.audio_batch_size]], extractor, encoder, device))
        if i % (args.audio_batch_size * 10) == 0:
            print(f"Audio features {min(i + args.audio_batch_size, len(rows))}/{len(rows)}", flush=True)
    audio_parameters = sum(p.numel() for p in encoder.parameters())
    del encoder, extractor
    gc.collect()
    tokenizer, encoder = load_text_encoder(device)
    text = []
    for i in range(0, len(rows), args.text_batch_size):
        text.append(encode_text([r["text"] for r in rows[i:i + args.text_batch_size]], tokenizer, encoder, device))
    token_lengths = [len(tokenizer(r["text"], truncation=False)["input_ids"]) for r in rows]
    text_parameters = sum(p.numel() for p in encoder.parameters())
    cache = out / "features.npz"
    np.savez_compressed(cache, ids=np.array([r["id"] for r in rows]), audio=np.concatenate(audio), text=np.concatenate(text))
    dump_json(out / "metadata.json", {
        "manifest_sha256": fingerprint(args.manifest), "features_sha256": fingerprint(cache),
        "audio_model": AUDIO_MODEL, "audio_revision": AUDIO_REVISION,
        "text_model": TEXT_MODEL, "text_revision": TEXT_REVISION,
        "audio_dim": audio[0].shape[1], "text_dim": text[0].shape[1],
        "frozen_audio_encoder_parameters": audio_parameters, "frozen_text_encoder_parameters": text_parameters,
        "encoder_trainable_parameters": 0, "audio_pooling": "valid-frame mean and population std; 20 ms encoder frames",
        "text_pooling": "attention-mask mean pooling", "max_text_tokens": MAX_TOKENS,
        "truncated_transcripts": sum(n > MAX_TOKENS for n in token_lengths),
        "seconds": time.perf_counter() - start, "device": device, "environment": environment(),
        "inputs": "Only waveform and source native-language transcription; no label, translation, ID or language feature",
        "cache_scope": "Independent frozen encoding only; all normalization is deferred to training split"})
    print(f"Saved paired frozen features for {len(rows)} recordings", flush=True)


if __name__ == "__main__":
    main()
