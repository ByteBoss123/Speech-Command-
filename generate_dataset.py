"""
generate_dataset.py

Synthesizes every (word, voice, rate, pitch) combination defined in
corpus.py and writes a manifest CSV mapping clip_id -> (split, word, path).
"""

import csv
from pathlib import Path

from corpus import flat_dataset_spec
from synth import synthesize

AUDIO_DIR = Path(__file__).parent / "audio"


def run():
    AUDIO_DIR.mkdir(exist_ok=True)
    rows = flat_dataset_spec()
    manifest = []

    for i, (split, word, voice, rate, pitch, clip_id) in enumerate(rows):
        wav_path = AUDIO_DIR / f"{clip_id}.wav"
        synthesize(word, wav_path, voice=voice, rate=rate, pitch=pitch)
        manifest.append({
            "clip_id": clip_id, "split": split, "word": word,
            "voice": voice, "rate": rate, "pitch": pitch,
            "path": str(wav_path.relative_to(Path(__file__).parent)),
        })
        if (i + 1) % 100 == 0:
            print(f"  synthesized {i + 1}/{len(rows)}")

    manifest_path = Path(__file__).parent / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    n_trainval = sum(1 for r in manifest if r["split"] == "trainval")
    n_unseen = sum(1 for r in manifest if r["split"] == "unseen")
    print(f"\nWrote {len(manifest)} clips to {AUDIO_DIR}")
    print(f"  trainval (5 voices): {n_trainval}")
    print(f"  unseen-voice test (2 voices): {n_unseen}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    run()
