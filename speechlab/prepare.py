"""Download a bounded, explicitly nonrepresentative LibriSpeech pilot subset."""
from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

from .common import RATE, audio_array, dump_json, fingerprint, split_summary, validate_splits

DATASET = "openslr/librispeech_asr"


def source_records(source_split, revision):
    """Read explicit local Parquet shards with no asynchronous scanner at shutdown."""
    prefix = f"clean/{source_split}/"
    files = sorted(f for f in HfApi().list_repo_files(DATASET, repo_type="dataset", revision=revision)
                   if f.startswith(prefix) and f.endswith(".parquet"))
    if not files:
        raise RuntimeError(f"No Parquet files for {source_split}")
    for name in files:
        local = hf_hub_download(DATASET, name, repo_type="dataset", revision=revision)
        with pq.ParquetFile(local) as parquet:
            for batch in parquet.iter_batches(batch_size=32, use_threads=False):
                yield from batch.to_pylist()


def select_split(source_split, logical_split, revision, speakers, clips, out, scan_limit):
    grouped, scanned = defaultdict(list), 0
    # A deterministic source-prefix sample with equal quotas. No selection by model scores.
    with closing(source_records(source_split, revision)) as stream:
        for source in stream:
            scanned += 1
            sid = str(source["speaker_id"])
            if len(grouped[sid]) < clips:
                payload = source["audio"]
                signal = audio_array(io.BytesIO(payload["bytes"]) if payload.get("bytes") else payload["path"])
                duration = len(signal) / RATE
                if 2.0 <= duration <= 15.0 and source["text"].strip():
                    grouped[sid].append((source, signal, duration))
            complete = [sid for sid, rows in grouped.items() if len(rows) == clips]
            if len(complete) >= speakers or scanned >= scan_limit:
                break
    chosen = [sid for sid, rows in grouped.items() if len(rows) == clips][:speakers]
    if len(chosen) != speakers:
        raise RuntimeError(f"Only {len(chosen)} complete speakers for {logical_split}; increase --scan-limit or reduce quotas")
    records = []
    for sid in chosen:
        for source, signal, duration in grouped[sid]:
            name = str(source["id"])
            # IDs are written as a file name, never as a path supplied by the dataset.
            if not all(c.isalnum() or c in "-_" for c in name):
                raise ValueError("Unsafe utterance ID")
            relative = Path("audio") / logical_split / f"{name}.wav"
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, signal, RATE, subtype="PCM_16")
            records.append({"id": name, "speaker_id": sid, "chapter_id": str(source["chapter_id"]),
                            "split": logical_split, "source_split": source_split,
                            "text": source["text"], "language": "en", "path": relative.as_posix(),
                            "sample_rate": RATE, "duration_seconds": duration,
                            "audio_sha256": fingerprint(target)})
    print(f"{logical_split}: {len(records)} clips from {len(chosen)} speakers; scanned {scanned}", flush=True)
    return records, scanned


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/librispeech-pilot")
    p.add_argument("--revision", default=None)
    p.add_argument("--train-speakers", type=int, default=16)
    p.add_argument("--train-clips", type=int, default=8)
    p.add_argument("--eval-speakers", type=int, default=12)
    p.add_argument("--eval-clips", type=int, default=4)
    p.add_argument("--scan-limit", type=int, default=12000)
    a = p.parse_args()
    if min(a.train_speakers, a.train_clips, a.eval_speakers, a.eval_clips, a.scan_limit) <= 0:
        p.error("Quotas and scan limit must be positive")
    if a.eval_clips < 2 or a.eval_speakers < 2:
        p.error("Speaker verification needs at least two speakers and two clips per speaker")
    out = Path(a.output)
    if (out / "manifest.jsonl").exists():
        p.error("Manifest already exists; reuse it or choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    revision = HfApi().dataset_info(DATASET, revision=a.revision).sha
    rows, scans = [], {}
    for logical, source in [("train", "train.100"), ("validation", "validation"), ("test", "test")]:
        n, k = (a.train_speakers, a.train_clips) if logical == "train" else (a.eval_speakers, a.eval_clips)
        selected, scanned = select_split(source, logical, revision, n, k, out, a.scan_limit)
        rows.extend(selected)
        scans[logical] = scanned
    validate_splits(rows)
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    dump_json(out / "dataset_card.json", {"dataset": DATASET, "revision": revision, "config": "clean",
        "license": "CC-BY-4.0", "source": "https://www.openslr.org/12/", "sample": "deterministic source-prefix speaker quotas",
        "selection": vars(a), "source_rows_scanned": scans, "splits": split_summary(rows),
        "manifest_sha256": fingerprint(manifest),
        "limitations": ["Small English audiobook subset; not a full-corpus benchmark or conversational audio.",
                        "Speakers are disjoint across adaptation/calibration/test; foundation pretraining overlap is unknown."]})
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
