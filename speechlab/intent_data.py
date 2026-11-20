"""Reproducible MInDS-14 audio/text splits, with no fabricated speaker IDs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

from .common import RATE, audio_array, dump_json, fingerprint

DATASET = "PolyAI/minds14"
INTENTS = ["abroad", "address", "app_error", "atm_limit", "balance", "business_loan",
           "card_issues", "cash_deposit", "direct_debit", "freeze", "high_value_payment",
           "joint_account", "latest_transactions", "pay_bill"]


def text_key(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split())


def sample_key(row, seed):
    return hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()


def balanced_split(rows, languages, quotas, seed=13):
    """Choose unique audio and normalized transcripts before stratified splitting."""
    groups, seen_audio, seen_text = defaultdict(list), set(), set()
    for row in sorted(rows, key=lambda x: sample_key(x, seed)):
        key = text_key(row["text"])
        if not key or key in seen_text or row["source_audio_sha256"] in seen_audio:
            continue
        seen_text.add(key)
        seen_audio.add(row["source_audio_sha256"])
        groups[(row["language"], row["label"])].append(row)
    needed = sum(quotas.values())
    selected = []
    for language in languages:
        for label in range(len(INTENTS)):
            candidates = groups[(language, label)]
            if len(candidates) < needed:
                raise ValueError(f"{language}/{INTENTS[label]} has {len(candidates)} eligible unique clips; need {needed}")
            offset = 0
            for split, count in quotas.items():
                selected.extend({**row, "split": split} for row in candidates[offset:offset + count])
                offset += count
    return sorted(selected, key=lambda r: (r["split"], r["language"], r["id"]))


def validate_intent_rows(rows):
    seen_ids, seen_audio, text_splits, counts = set(), set(), {}, Counter()
    for row in rows:
        split = row["split"]
        if split not in {"train", "validation", "test"} or row["label"] not in range(len(INTENTS)):
            raise ValueError("Unknown split or intent label")
        if row["id"] in seen_ids or row["audio_sha256"] in seen_audio:
            raise ValueError("Duplicate recording")
        key = text_key(row["text"])
        if not key:
            raise ValueError("Empty transcript")
        if key in text_splits and text_splits[key] != split:
            raise ValueError("Exact transcript leakage across splits")
        seen_ids.add(row["id"])
        seen_audio.add(row["audio_sha256"])
        text_splits[key] = split
        counts[(split, row["language"], row["label"])] += 1
    languages = {r["language"] for r in rows}
    if not languages or any(counts[(s, lang, k)] == 0 for s in ["train", "validation", "test"]
                            for lang in languages for k in range(len(INTENTS))):
        raise ValueError("Every split must contain every language/intent stratum")


def read_intent_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_intent_rows(rows)
    for row in rows:
        audio_path = (path.parent / row["path"]).resolve()
        if not audio_path.is_relative_to(path.parent):
            raise ValueError("Audio path escapes dataset directory")
        if fingerprint(audio_path) != row["audio_sha256"]:
            raise ValueError(f"Audio checksum mismatch: {row['id']}")
        row["absolute_path"] = str(audio_path)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/minds14-pilot")
    p.add_argument("--languages", nargs="+", default=["en-US", "es-ES", "fr-FR"])
    p.add_argument("--revision", default="40ce77cb32a384e4d50a568e1ec39ac804019d33")
    p.add_argument("--train-per-class", type=int, default=8)
    p.add_argument("--validation-per-class", type=int, default=3)
    p.add_argument("--test-per-class", type=int, default=3)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    quotas = {"train": a.train_per_class, "validation": a.validation_per_class, "test": a.test_per_class}
    if min(quotas.values()) < 1 or len(set(a.languages)) != len(a.languages):
        p.error("Positive quotas and distinct languages are required")
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        p.error("Output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(DATASET, revision=a.revision)
    candidates, source_counts, excluded = [], Counter(), Counter()
    audio_bytes = {}
    for lang in a.languages:
        files = sorted(f.rfilename for f in info.siblings if f.rfilename.startswith(lang + "/") and f.rfilename.endswith(".parquet"))
        if not files:
            raise ValueError(f"No data for language {lang}")
        for name in files:
            local = hf_hub_download(DATASET, name, repo_type="dataset", revision=info.sha)
            with pq.ParquetFile(local) as parquet:
                # Assert label order against the source Arrow metadata when available.
                metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
                assert metadata["info"]["features"]["intent_class"]["names"] == INTENTS
                for batch in parquet.iter_batches(batch_size=32, use_threads=False):
                    for row in batch.to_pylist():
                        source_counts[lang] += 1
                        payload = row["audio"]["bytes"]
                        audio = audio_array(io.BytesIO(payload))
                        if not 1 <= len(audio) / RATE <= 30 or not text_key(row["transcription"]):
                            excluded[lang] += 1
                            continue
                        sid = hashlib.sha256((lang + ":" + row["path"]).encode()).hexdigest()[:24]
                        digest = hashlib.sha256(payload).hexdigest()
                        candidates.append({"id": sid, "language": lang, "label": row["intent_class"],
                                           "intent": INTENTS[row["intent_class"]], "text": row["transcription"],
                                           "source_path": row["path"], "source_split": "train",
                                           "source_audio_sha256": digest, "duration_seconds": len(audio) / RATE})
                        audio_bytes[sid] = payload
        print(f"Read {source_counts[lang]} source recordings for {lang}", flush=True)
    selected = balanced_split(candidates, a.languages, quotas, a.seed)
    for row in selected:
        relative = Path("audio") / row["split"] / f"{row['id']}.wav"
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, audio_array(io.BytesIO(audio_bytes[row["id"]])), RATE, subtype="PCM_16")
        row.update(path=relative.as_posix(), audio_sha256=fingerprint(target), sample_rate=RATE)
    validate_intent_rows(selected)
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    dump_json(out / "dataset_card.json", {
        "dataset": DATASET, "revision": info.sha, "license": "CC-BY-4.0", "source_rows": dict(source_counts),
        "duration_or_empty_text_exclusions": dict(excluded), "config": vars(a), "labels": INTENTS,
        "split_counts": dict(Counter(r["split"] for r in selected)),
        "per_language_split_counts": {lang: dict(Counter(r["split"] for r in selected if r["language"] == lang)) for lang in a.languages},
        "manifest_sha256": fingerprint(manifest), "sample_method": "SHA-256 seeded order; equal language/intent quotas; exact transcript/audio deduplication",
        "text_source": "Dataset-provided native-language transcription; English translations and class names are not model inputs",
        "limitations": ["New utterance-level splits of the source train partition, not an official benchmark split.",
                        "Speaker identifiers are not exposed in this dataset schema; speaker independence cannot be established.",
                        "Near-duplicate phrasing and foundation pretraining overlap remain possible.",
                        "Both audio and supplied transcript are available at inference; no end-to-end ASR accuracy is measured."]})
    print(f"Prepared {len(selected)} paired recordings: {dict(Counter(r['split'] for r in selected))}", flush=True)


if __name__ == "__main__":
    main()
