"""Prepare expanded training and fresh cross-chapter LibriSpeech evaluation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import soundfile as sf

from .common import RATE, audio_array, dump_json, fingerprint, split_summary, validate_splits
from .prepare import DATASET, source_records

REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"


def order(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def choose_chapters(groups, count, clips, seed):
    chosen = []
    for speaker in sorted(groups, key=lambda x: order(x, seed)):
        chapters = [c for c in groups[speaker] if len(groups[speaker][c]) >= clips]
        if len(chapters) >= 2:
            chapters.sort(key=lambda x: order(x, seed))
            chosen.append((speaker, chapters[:2]))
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f"Only {len(chosen)} eligible fresh cross-chapter speakers; requested {count}")
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", default="data/librispeech-pilot/manifest.jsonl")
    p.add_argument("--output", default="data/librispeech-v2")
    p.add_argument("--train-speakers", type=int, default=48)
    p.add_argument("--train-clips", type=int, default=8)
    p.add_argument("--eval-speakers", type=int, default=20)
    p.add_argument("--chapter-clips", type=int, default=3)
    p.add_argument("--seed", type=int, default=19)
    args = p.parse_args()
    if min(args.train_speakers, args.train_clips, args.eval_speakers, args.chapter_clips) < 1:
        p.error("Positive quotas required")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error("Output must be empty")
    old = [json.loads(x) for x in Path(args.previous).read_text().splitlines()]
    excluded = {r["speaker_id"] for r in old if r["split"] != "train"}
    out.mkdir(parents=True, exist_ok=True)
    records, scans = [], {}
    for split, source_split in (("train", "train.100"), ("validation", "validation"), ("test", "test")):
        groups, scanned = defaultdict(lambda: defaultdict(list)), 0
        with closing(source_records(source_split, REVISION)) as stream:
            for row in stream:
                scanned += 1
                speaker, chapter = str(row["speaker_id"]), str(row["chapter_id"])
                if split != "train" and speaker in excluded:
                    continue
                bucket = groups[speaker][chapter if split != "train" else "train"]
                limit = args.chapter_clips if split != "train" else args.train_clips
                if len(bucket) >= limit:
                    continue
                payload = row["audio"]["bytes"]
                with sf.SoundFile(io.BytesIO(payload)) as f:
                    duration = len(f) / f.samplerate
                if 2 <= duration <= 20 and row["text"].strip():
                    bucket.append((row, payload))
                if split == "train" and sum(len(v["train"]) >= args.train_clips for v in groups.values()) >= args.train_speakers:
                    break
        if split == "train":
            candidates = [s for s in groups if len(groups[s]["train"]) >= args.train_clips]
            candidates.sort(key=lambda s: order(s, args.seed))
            if len(candidates) < args.train_speakers:
                raise ValueError("Insufficient training speakers")
            chosen = [(s, ["train"]) for s in candidates[:args.train_speakers]]
        else:
            chosen = choose_chapters(groups, args.eval_speakers, args.chapter_clips, args.seed)
        for speaker, chapters in chosen:
            for chapter_number, chapter in enumerate(chapters):
                for row, payload in groups[speaker][chapter]:
                    sid = str(row["id"])
                    if not all(c.isalnum() or c in "-_" for c in sid):
                        raise ValueError("Invalid recording ID")
                    audio = audio_array(io.BytesIO(payload))
                    relative = Path("audio") / split / f"{sid}.wav"
                    target = out / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(target, audio, RATE, subtype="PCM_16")
                    records.append({"id": sid, "speaker_id": speaker, "chapter_id": str(row["chapter_id"]),
                                    "split": split, "source_split": source_split, "text": row["text"], "language": "en",
                                    "path": relative.as_posix(), "sample_rate": RATE, "duration_seconds": len(audio) / RATE,
                                    "audio_sha256": fingerprint(target),
                                    "verification_role": "cohort" if split == "train" else "enroll" if chapter_number == 0 else "probe"})
        scans[split] = scanned
        print(f"Prepared {split}: {sum(r['split'] == split for r in records)} recordings", flush=True)
    validate_splits(records)
    if any(r["speaker_id"] in excluded for r in records if r["split"] != "train"):
        raise ValueError("Earlier evaluation speaker leaked into new evaluation")
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    dump_json(out / "dataset_card.json", {"dataset": DATASET, "revision": REVISION, "license": "CC-BY-4.0",
              "config": vars(args), "splits": split_summary(records), "source_rows_scanned": scans,
              "manifest_sha256": fingerprint(manifest), "previous_manifest_sha256": fingerprint(args.previous),
              "fresh_evaluation_speakers": True, "evaluation_chapters_per_speaker": 2,
              "limitations": ["Training is a bounded source-prefix sample, not full LibriSpeech.",
                               "Evaluation excludes all v1 evaluation speakers but foundation pretraining overlap is unknown.",
                               "Cross-chapter recordings can still share book, microphone and session conditions."]})


if __name__ == "__main__":
    main()
