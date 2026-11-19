"""Expand MInDS training while reserving a new holdout outside all v1 examples."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

from .common import RATE, audio_array, dump_json, fingerprint
from .intent_data import DATASET, INTENTS, sample_key, text_key, validate_intent_rows

REVISION = "40ce77cb32a384e4d50a568e1ec39ac804019d33"


def split_fresh(rows, previous, test_per_class=5, validation_per_class=4, seed=19):
    old_ids = {r["id"] for r in previous}
    old_text = {text_key(r["text"]) for r in previous}
    old_audio = {r["source_audio_sha256"] for r in previous}
    old_val = {r["id"] for r in previous if r["split"] == "validation"}
    groups, seen_text, seen_audio = defaultdict(list), set(), set()
    # Preserve previously selected representatives when exact duplicates exist.
    for row in sorted(rows, key=lambda r: (r["id"] not in old_ids, sample_key(r, seed))):
        key = text_key(row["text"])
        if key in seen_text or row["source_audio_sha256"] in seen_audio:
            continue
        seen_text.add(key); seen_audio.add(row["source_audio_sha256"])
        groups[(row["language"], row["label"])].append(row)
    selected = []
    for key in sorted(groups):
        group = groups[key]
        fresh = sorted([r for r in group if r["id"] not in old_ids and text_key(r["text"]) not in old_text
                        and r["source_audio_sha256"] not in old_audio], key=lambda r: sample_key(r, seed))
        if len(fresh) < test_per_class:
            raise ValueError(f"Not enough untouched test examples for {key}: {len(fresh)}")
        test_ids = {r["id"] for r in fresh[:test_per_class]}
        development = sorted([r for r in group if r["id"] not in test_ids], key=lambda r: (r["id"] not in old_val, sample_key(r, seed)))
        if len(development) <= validation_per_class:
            raise ValueError("No training examples remain")
        val_ids = {r["id"] for r in development[:validation_per_class]}
        selected.extend({**r, "split": "test" if r["id"] in test_ids else "validation" if r["id"] in val_ids else "train"} for r in group)
    return sorted(selected, key=lambda r: (r["split"], r["language"], r["id"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", default="data/minds14-pilot/manifest.jsonl")
    p.add_argument("--output", default="data/minds14-v2")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()): p.error("Output must be empty")
    previous = [json.loads(x) for x in Path(args.previous).read_text().splitlines()]
    rows, payloads = [], {}
    for lang in ["en-US", "es-ES", "fr-FR"]:
        source = hf_hub_download(DATASET, f"{lang}/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
        with pq.ParquetFile(source) as f:
            for batch in f.iter_batches(batch_size=32, use_threads=False):
                for r in batch.to_pylist():
                    payload = r["audio"]["bytes"]
                    audio = audio_array(io.BytesIO(payload))
                    if not 1 <= len(audio) / RATE <= 30 or not text_key(r["transcription"]): continue
                    sid = hashlib.sha256((lang + ":" + r["path"]).encode()).hexdigest()[:24]
                    rows.append({"id": sid, "language": lang, "label": r["intent_class"], "intent": INTENTS[r["intent_class"]],
                                 "text": r["transcription"], "source_audio_sha256": hashlib.sha256(payload).hexdigest(),
                                 "duration_seconds": len(audio) / RATE})
                    payloads[sid] = payload
    rows = split_fresh(rows, previous)
    for r in rows:
        relative = Path("audio") / r["split"] / (r["id"] + ".wav")
        target = out / relative; target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, audio_array(io.BytesIO(payloads[r["id"]])), RATE, subtype="PCM_16")
        r.update(path=relative.as_posix(), sample_rate=RATE, audio_sha256=fingerprint(target))
    validate_intent_rows(rows)
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    dump_json(out / "dataset_card.json", {"dataset": DATASET, "revision": REVISION, "license": "CC-BY-4.0",
        "counts": dict(Counter(r["split"] for r in rows)), "manifest_sha256": fingerprint(manifest),
        "previous_manifest_sha256": fingerprint(args.previous), "test_per_language_intent": 5,
        "validation_per_language_intent": 4, "seed": 19,
        "fresh_test": "Excludes every previously used ID, exact normalized transcript and source-audio hash",
        "development": "Earlier test is retired and may be used for v2 training; prior validation prioritized for new validation",
        "limitations": ["Speaker IDs unavailable; utterance-level splits only.", "Source transcripts required at inference.",
                        "Training uses all remaining eligible examples with natural class imbalance; validation/test balanced."]})
    print(dict(Counter(r["split"] for r in rows)), flush=True)


if __name__ == "__main__": main()
