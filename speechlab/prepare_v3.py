"""Create a fresh speaker-disjoint experiment from unused train.100 speakers."""
from __future__ import annotations
import argparse
import io
import json
from collections import defaultdict
from contextlib import closing
from pathlib import Path
import soundfile as sf
from .common import RATE, audio_array, dump_json, fingerprint, split_summary, validate_splits
from .prepare import DATASET, source_records
from .prepare_v2 import REVISION, choose_chapters, order


def fresh_partition(groups, excluded, seed=31, validation=20, test=30, clips=5):
    eligible = {s: chapters for s, chapters in groups.items() if s not in excluded}
    chosen = choose_chapters(eligible, validation + test, clips, seed)
    return {"validation": chosen[:validation], "test": chosen[validation:]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", nargs="+", default=["data/librispeech-pilot/manifest.jsonl", "data/librispeech-v2/manifest.jsonl"])
    p.add_argument("--output", default="data/librispeech-v3")
    a = p.parse_args(); out = Path(a.output)
    if out.exists() and any(out.iterdir()): p.error("Output must be empty")
    previous = [[json.loads(line) for line in Path(path).read_text().splitlines()] for path in a.previous]
    excluded = {r["speaker_id"] for manifest in previous for r in manifest}
    train_ids = {r["speaker_id"] for r in previous[-1] if r["split"] == "train"}
    if len(train_ids) != 48: raise ValueError("Expected the existing 48 training speakers")
    train, groups = defaultdict(list), defaultdict(lambda: defaultdict(list))
    eligible, scanned = set(), 0
    with closing(source_records("train.100", REVISION)) as stream:
        for row in stream:
            scanned += 1
            speaker, chapter = str(row["speaker_id"]), str(row["chapter_id"])
            if speaker in train_ids: bucket, limit = train[speaker], 16
            elif speaker not in excluded: bucket, limit = groups[speaker][chapter], 5
            else: continue
            if len(bucket) < limit:
                payload = row["audio"]["bytes"]
                with sf.SoundFile(io.BytesIO(payload)) as f: duration = len(f) / f.samplerate
                if 2 <= duration <= 20 and row["text"].strip(): bucket.append((row, payload))
            if speaker not in excluded and sum(len(v) >= 5 for v in groups[speaker].values()) >= 2: eligible.add(speaker)
            if scanned % 1000 == 0: print(f"Scanned {scanned}; fresh eligible speakers {len(eligible)}", flush=True)
            if len(eligible) >= 50 and all(len(train[s]) >= 16 for s in train_ids): break
    if not all(len(train[s]) == 16 for s in train_ids): raise ValueError("Training quota not met")
    partition = fresh_partition(groups, excluded)
    selected = [("train", s, [("cohort", train[s])]) for s in sorted(train_ids, key=lambda s: order(s, 31))]
    for split, speakers in partition.items():
        for speaker, chapters in speakers:
            selected.append((split, speaker, [(role, groups[speaker][chapter]) for role, chapter in zip(("enroll", "probe"), chapters)]))
    rows = []
    for split, speaker, role_groups in selected:
        for role, clips in role_groups:
            for row, payload in clips:
                sid = str(row["id"])
                if not all(c.isalnum() or c in "-_" for c in sid): raise ValueError("Unsafe source ID")
                signal = audio_array(io.BytesIO(payload))
                relative = Path("audio") / split / f"{sid}.wav"
                target = out / relative; target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, signal, RATE, subtype="PCM_16")
                rows.append({"id": sid, "speaker_id": speaker, "chapter_id": str(row["chapter_id"]),
                    "split": split, "source_split": "train.100", "text": row["text"], "language": "en",
                    "path": relative.as_posix(), "sample_rate": RATE, "duration_seconds": len(signal) / RATE,
                    "audio_sha256": fingerprint(target), "verification_role": role})
    validate_splits(rows)
    if {r["speaker_id"] for r in rows if r["split"] != "train"} & excluded: raise ValueError("Earlier speaker leaked")
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    dump_json(out / "dataset_card.json", {"dataset": DATASET, "revision": REVISION, "license": "CC-BY-4.0",
        "source_split": "train.100", "source_rows_scanned": scanned, "seed": 31,
        "splits": split_summary(rows), "asr_splits": split_summary([r for r in rows if r["split"] == "train" or r["verification_role"] == "probe"]),
        "manifest_sha256": fingerprint(manifest), "previous_manifest_sha256": [fingerprint(x) for x in a.previous],
        "evaluation_speakers_excluded_from_all_prior_manifests": True,
        "limitations": ["Bounded source-prefix sample, not an official LibriSpeech test benchmark.",
            "Foundational-model pretraining overlap unknown.", "Chapter separation does not guarantee session or microphone independence."]})
    print(json.dumps(split_summary(rows)), flush=True)


if __name__ == "__main__": main()
