---
base_model: openai/whisper-tiny
library_name: peft
pipeline_tag: automatic-speech-recognition
language:
  - en
tags:
  - lora
  - research
  - negative-result
---

# Whisper decoder LoRA — small LibriSpeech pilot

Trained on CPU on November 19, 2026: decoder query/value attention projections,
rank 8, 98,304 trainable parameters. Base revision is pinned in `adapter_config.json`.

Training used 128 real English audiobook recordings from 16 speakers (25.1 minutes),
two epochs, and 128 optimizer steps. Epoch 2 was selected by validation token loss on
48 recordings from 12 separate speakers. The encoder was frozen and cached.

**Known regression:** clean test WER increased from 5.61% to 11.52%; 10 dB Gaussian-noise
test WER increased from 12.73% to 14.13%. Do not promote this adapter as an improvement.
It is retained as reproducible evidence of training and failure analysis.

Intended use: education, reproduction, and error analysis. No conversational,
multilingual, speaker-intent, or deployment evaluation. Foundation pretraining overlap
is unknown. Source-model and dataset terms apply; see the upstream Whisper checkpoint
and LibriSpeech dataset card. Source model weights are not redistributed here.

From the repository root:

```bash
python -m speechlab.transcribe recording.wav --adapter evidence/2026-11-19/asr/adapter
```

See the parent directory for predictions, metrics, training history, and configuration.
