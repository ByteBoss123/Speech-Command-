# Speech adaptation, speaker verification, and multilingual intent

Three implemented and executed speech ML experiments. V2 adds fresh holdouts,
validation-only decisions, repeated intent-training seeds, harder speaker trials,
saved models and recomputable scorecards.

## Fresh-holdout results

Comparisons use the **same new test examples within each experiment**.

| Experiment | Reference on the new test | Improved system on the same test |
|---|---|---|
| Clean ASR | Whisper-tiny: 6.42% WER | Tiny + LoRA: **5.91%**; selected Whisper-base: **4.66%** |
| ASR, 10 dB Gaussian noise | Whisper-tiny: 14.96% WER | Tiny + LoRA: 14.80%; selected Whisper-base: **9.51%** |
| Cross-chapter speaker verification | Single enrollment: 36.67% false rejects, 0.70% false accepts | Three-recording enrollment: **10.00% false rejects, 0.96% false accepts** |
| Multilingual intent | Unchanged v1 fusion: 95.71% accuracy | Validation-selected v2 blend: **97.62% accuracy**, **0.9760 macro F1** |

**Scope:** research pilots on LibriSpeech and MInDS-14. The chosen ASR model is
pretrained Whisper-base: its larger gain is a **capacity/model-selection gain**, not
a LoRA gain. The tiny adapter separately improves clean transcription. No production
authentication or generative conversational speech-LLM capability is claimed.

### ASR adaptation and model selection

Training: **384 recordings / 48 speakers / 84.2 minutes**. Validation and test each
have **120 recordings / 20 speakers**, excluding all earlier evaluation speakers.
The new test contains **2,554 reference words**.

Train **98,304** decoder LoRA parameters with normalized targets, learning rate
`5e-5`, warmup/decay and 15 dB noise on half the training recordings. Epoch 2 wins
among the unchanged baseline and three epochs, using **generated validation WER**.
A larger pretrained candidate is added after development inspection and before
ASR test scoring. Its 4.80% validation WER beats the selected tiny adapter's 7.51%.

The tiny adapter's clean WER difference is **−0.51 percentage points**, with a
speaker-bootstrap 95% CI of **[−1.02, −0.05]**; its noisy-speech gain is uncertain.
Selected Whisper-base minus original tiny: **−1.76 points [−2.75, −0.86]** clean,
and **−5.44 points [−7.04, −4.07]** with synthetic noise. Base has 72.6M parameters
versus tiny's 37.8M; no fair latency benchmark was run.

[ASR comparison](evidence/2026-11-19-v2/asr_selection.json) ·
[LoRA evidence](evidence/2026-11-19-v2/asr/metrics.json) ·
[Choice recorded before test exports](evidence/2026-11-19-v2/asr_choice_record.json).

### Cross-chapter speaker verification

Frozen WavLM embeddings compare one enrollment recording, three-recording averaging,
and optional training-cohort centering. Each probe comes from a **different chapter**
from its speaker's enrollment. Method and threshold are selected on validation at a
**1% empirical FAR target**.

On **20 new test speakers / 1,200 trials** (60 positive, 1,140 negative), the selected
three-recording method reduces false rejects from **22/60 to 6/60**. False accepts
increase from **8/1,140 to 11/1,140**. Descriptive EER decreases **5.18% → 3.03%**.
The paired false-reject difference is −26.67 points, with a probe-speaker bootstrap
interval **[−41.67, −13.33]**, conditional on the enrolled cohort.

Cross-chapter does not guarantee different sessions or microphones. Trials reuse
recordings/speakers; observed FAR is not a guaranteed bound. WavLM remains frozen.
[Speaker scorecard](evidence/2026-11-19-v2/speaker/metrics.json).

### Multilingual speech–text intent routing

MInDS-14: **English, Spanish, French**, 14 banking intents. **1,135 train / 168
validation / 210 test** recordings. The new test excludes all previous IDs,
normalized transcripts and source-audio hashes. The previous test is retired and
may enter the expanded training pool.

Frozen Whisper audio and multilingual MiniLM text features feed trained PyTorch
heads. Three seeds (`13, 29, 47`) are trained per modality. Validation chooses
**75% text-ensemble + 25% joint-ensemble probabilities**.

| Model on the new test | Accuracy | Macro F1 |
|---|---:|---:|
| Unchanged v1 fusion | 95.71% | 0.9571 |
| v2 audio ensemble | 83.33% | 0.8318 |
| v2 text ensemble | 97.14% | 0.9712 |
| v2 joint ensemble | 98.10% | 0.9809 |
| **Validation-selected blend** | **97.62%** | **0.9760** |

The final choice remains the validation-selected blend despite the pure joint
ensemble's higher test score. Individual joint heads score 97.62–98.10% across
three seeds. The final blend's gain over v1 is +1.90 points, with a descriptive
utterance-bootstrap CI **[−0.48, +4.29]**; the gain is still uncertain.

Dataset-provided native-language transcripts are inputs. Speaker IDs are unavailable:
this is **not speaker-disjoint**, zero-shot language transfer or end-to-end ASR.
The audio contribution over text only is small.
[Intent scorecard](evidence/2026-11-19-v2/intent/metrics.json).

## Reproduce

Python 3.12 / CPU is tested. Install a compatible CUDA Torch wheel instead for GPU use.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-speech.txt
python -m pytest tests_speech -q

# Recreate original manifests for reproducible exclusion rules.
python -m speechlab.prepare --revision 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1
python -m speechlab.intent_data
python -m speechlab.prepare_v2
python -m speechlab.intent_data_v2
python -m speechlab.asr_v2
python -m speechlab.asr_capacity
python -m speechlab.select_asr
python -m speechlab.speaker_v2
python -m speechlab.intent_features --manifest data/minds14-v2/manifest.jsonl --output runs/intent-features-v2
python -m speechlab.intent_v2

# Recompute committed scorecards without downloading audio/pretrained models.
python -m speechlab.verify_v2
```

Use fresh output directories for reruns. On Windows activate with
`.venv\Scripts\Activate.ps1`. Allow several GB for datasets, checkpoints and local
encoder caches. No paid APIs are used.
[Run in Colab](https://colab.research.google.com/github/ByteBoss123/Speech-Command-/blob/codex/siri-real-speech-lab-20261119/notebooks/Speech_Experiments_v2.ipynb).
Notebook cells are unexecuted; equivalent CLI commands produced the saved evidence.

## Use saved models

```bash
python -m speechlab.asr_predict_v2 recording.wav
python -m speechlab.intent_ensemble_predict recording.wav --transcript "How can I freeze my card?"
python -m speechlab.speaker_predict --enroll enrollment1.wav enrollment2.wav enrollment3.wav --probe probe.wav
```

The intent transcript must correspond to the recording. ASR accepts up to 30 seconds,
intent audio 1–30 seconds, and the speaker demo 2–20 seconds per recording. These are
local inference demonstrations, with no service deployment or action execution.

## Evidence and limits

- **28 offline tests** cover leakage, normalization, checkpoint reload, baseline
  retention, capacity ties, cross-chapter enforcement, enrollment and modality selection.
- All nine intent heads were reloaded and reproduced per-seed/ensemble test scores;
  normalization matched training-only statistics. Raw-input inference was checked.
- Public v2 evidence includes error counts, probabilities, positional example aliases,
  anonymous speaker groups, learned heads and aggregate data cards. Source transcripts,
  original recording IDs and per-record speaker embeddings remain local.
- Foundation pretraining overlap is unknown. LibriSpeech training is a bounded prefix
  sample; noise is synthetic; MInDS speaker independence is unverified.
- [Protocol](experiments/v2_protocol.json) · [Interpretation](docs/V2_RESULTS.md) ·
  [Resume wording](docs/RESUME_PROJECTS.md).

Earlier experiments and their negative results remain under `evidence/2026-11-19/`.
[Earlier pilot README](https://github.com/ByteBoss123/Speech-Command-/blob/86453ef5ddf75d4e63f9797d4dd34c5236383a91/README.md) ·
[Synthetic baseline](docs/SYNTHETIC_BASELINE.md).

Sources: [LibriSpeech](https://www.openslr.org/12/),
[MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) (both CC BY 4.0),
[Whisper](https://huggingface.co/openai/whisper-base),
[WavLM](https://huggingface.co/microsoft/wavlm-base-plus-sv),
[MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
[PEFT](https://huggingface.co/docs/peft/package_reference/lora).
Implemented and evaluated with AI assistance; limitations and earlier negative results are retained.
