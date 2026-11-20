from __future__ import annotations

import hashlib
import json
import math
import platform
import random
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 16000


def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    import torch

    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: version(p) for p in ["torch", "transformers", "peft", "datasets", "numpy", "jiwer"]},
            "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads()}


def seed_everything(seed=13, threads=4):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def audio_array(source, expected_rate=RATE):
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and finite")
    audio = audio.mean(axis=1)
    if rate != expected_rate:
        divisor = math.gcd(rate, expected_rate)
        audio = resample_poly(audio, expected_rate // divisor, rate // divisor).astype(np.float32)
    return audio


def add_noise(audio, snr_db, seed):
    rng = np.random.default_rng(seed)
    power = float(np.mean(np.square(audio)))
    if power <= 0:
        raise ValueError("Cannot define an SNR for silent audio")
    noise = rng.normal(size=len(audio)).astype(np.float32)
    noise *= np.sqrt(power / (10 ** (snr_db / 10) * np.mean(noise ** 2)))
    return audio + noise


def validate_splits(records):
    groups, ids, hashes = defaultdict(set), set(), set()
    for row in records:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if row["id"] in ids:
            raise ValueError("Duplicate utterance ID")
        if row["audio_sha256"] in hashes:
            raise ValueError("Duplicate audio content")
        if not row["text"].strip():
            raise ValueError("Empty transcript")
        ids.add(row["id"])
        hashes.add(row["audio_sha256"])
        groups[row["split"]].add(str(row["speaker_id"]))
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if groups[left] & groups[right]:
            raise ValueError(f"Speaker leakage between {left} and {right}")
    if any(not groups[s] for s in ("train", "validation", "test")):
        raise ValueError("Every split must contain records")


def read_manifest(path):
    path = Path(path).resolve()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_splits(records)
    for row in records:
        audio_path = (path.parent / row["path"]).resolve()
        if not audio_path.is_relative_to(path.parent):
            raise ValueError("Audio path escapes the dataset directory")
        if fingerprint(audio_path) != row["audio_sha256"]:
            raise ValueError(f"Audio checksum mismatch: {row['id']}")
        row["absolute_path"] = str(audio_path)
    return records


def split_summary(records):
    return {s: {"utterances": sum(r["split"] == s for r in records),
                "speakers": len({r["speaker_id"] for r in records if r["split"] == s}),
                "minutes": sum(r["duration_seconds"] for r in records if r["split"] == s) / 60}
            for s in ("train", "validation", "test")}
