"""Compare Whisper-base with its own LoRA adaptation on fresh speakers."""
from .asr_capacity import MODEL, REVISION
from .asr_v2 import main

if __name__ == "__main__":
    main(model_name=MODEL, revision=REVISION, seed=31,
         manifest="data/librispeech-v3/manifest.jsonl", output="runs/asr-v3", probe_only=True)
