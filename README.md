# Speech Adaptation, Speaker Verification, and Multilingual Intent

Three implemented and executed real-speech experiments, plus the original synthetic keyword classifier.

| Experiment | Implementation | Execution evidence |
|---|---|---|
| Generative speech recognition | Whisper-tiny decoder LoRA; frozen encoder cache; PyTorch/PEFT | 128 real training clips, 128 optimizer steps, saved adapter, baseline/adapted predictions |
| Speaker verification | Frozen WavLM X-vector embeddings, cosine scores, development-only threshold calibration | 144 test trials from 12 speakers; raw trials and calibration artifact |
| Multilingual speech–text intent routing | Frozen Whisper audio + multilingual MiniLM text encoders; trained PyTorch fusion head | 588 real recordings in English, Spanish, French; 95.24% accuracy on 126 test utterances; audio/text baselines and saved heads |
| Original isolated-word classifier | MFCC CNN trained from scratch on synthetic speech | [Original documentation](docs/SYNTHETIC_BASELINE.md) and root-level historical artifacts |

**Scope:** small LibriSpeech and MInDS-14 research pilots. The ASR experiment trains
Whisper decoder LoRA weights; the intent experiment trains classification heads on
frozen audio/text encoders. These experiments do not establish a conversational speech
LLM, production authentication, or classification of whether someone is addressing Siri.

## Measured results — November 19, 2026

Training: **128 recordings / 16 speakers / 25.1 minutes**. Validation: **48 recordings /
12 speakers**. Test: **48 recordings / 12 different speakers / 998 reference words**.
Speakers and audio content are checked for overlap across all three splits.

### ASR: adaptation regressed

| Test condition | Pretrained WER | LoRA WER | LoRA minus base, percentage points [speaker-bootstrap 95% CI] |
|---|---:|---:|---:|
| Clean | **5.61%** | 11.52% | +5.91 [+0.34, +13.43] |
| Gaussian noise, 10 dB SNR | **12.73%** | 14.13% | +1.40 [+0.43, +2.63] |

Lower WER is better. Fine-tuning **did not improve** this held-out pilot. Validation
token loss fell from 2.146 to 1.339, but generated-transcript accuracy worsened. Two
large regressions account for 54 of the 59 additional clean-speech word errors.
The pretrained model remains the better transcription choice in this experiment.
The negative result, predictions, error analysis, and adapter are retained.

- Base: `openai/whisper-tiny`, **37,760,640** base parameters.
- Trained: **98,304** decoder LoRA parameters; rank 8, two epochs, learning rate 0.0002.
- Selection: lowest validation token loss among trained epochs. Test scores did not select the checkpoint.
- [ASR metrics](evidence/2026-11-19/asr/metrics.json), [training history](evidence/2026-11-19/asr/training_history.json), [errors](evidence/2026-11-19/asr/error_analysis.json).

### Speaker verification

The threshold was calibrated on validation speakers with a target empirical FAR of at
most 5%, then fixed at **0.91006** for different test speakers.

| Test metric | Result |
|---|---:|
| False-accept rate | **1.39%** — 1 / 72 different-speaker trials |
| False-reject rate | **4.17%** — 3 / 72 same-speaker trials |
| Approximate empirical EER | 4.17% |

The 144 trials reuse recordings; **all same-speaker test pairs share a chapter**.
This is a small, relatively easy evaluation, not independent authentication trials or
evidence of general performance. WavLM is frozen; no WavLM training is claimed.
[Metrics](evidence/2026-11-19/speaker/metrics.json) · [raw trials](evidence/2026-11-19/speaker/trials_test.json).

### Multilingual speech–text intent routing — third project

MInDS-14 banking requests in **English, Spanish, and French**, with 14 intents.
Training: **336** recordings; validation: **126**; test: **126** (42 per language).
Both the recording and its dataset-provided native-language transcript are inputs.

| Input | Test accuracy | Macro F1 | Correct / total |
|---|---:|---:|---:|
| Audio only | 56.35% | 0.5527 | 71 / 126 |
| Text only | 93.65% | 0.9371 | 118 / 126 |
| Audio + text | **95.24%** | **0.9518** | 120 / 126 |

The fusion head trains **149,390 parameters**. Its accuracy lead over text only is
**1.59 percentage points**, with a descriptive utterance-bootstrap 95% interval of
**[−2.38, +6.35]** points; this small run does **not** establish a reliable improvement.
Speaker IDs are unavailable, so this is an **utterance split, not a speaker-disjoint test**.
All three languages appear in training. No end-to-end ASR accuracy is claimed.

[Design, results, and reproduction](docs/MULTILINGUAL_INTENT.md) ·
[metrics](evidence/2026-11-19/intent/metrics.json) ·
[raw predictions](evidence/2026-11-19/intent/predictions_test.json) ·
[Colab notebook](https://colab.research.google.com/github/ByteBoss123/Speech-Command-/blob/codex/siri-real-speech-lab-20261119/notebooks/Multilingual_Intent_Lab.ipynb).

## Run

Python 3.12 is tested. CPU setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-speech.txt
python -m pytest tests_speech -q

python -m speechlab.prepare --output data/librispeech-pilot \
  --revision 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1
python -m speechlab.asr --output runs/whisper-pilot --noise-snr 10 \
  --revision 169d4a4341b33bc18d8881c4b69c2e104e1cc0af
python -m speechlab.speaker --output runs/speaker-pilot \
  --revision feb593a6c23c1cc3d9510425c29b0a14d2b07b1e

# Third project: pinned data/encoder revisions are defaults in these modules.
python -m speechlab.intent_data
python -m speechlab.intent_features
python -m speechlab.intent
python -m speechlab.verify_intent evidence/2026-11-19/intent
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For GPU use, install a compatible
Torch GPU wheel instead; the experiment selects CUDA automatically. Output directories
must be empty to preserve earlier evidence. Downloads include several Parquet shards
and checkpoints; allow several GB for encoder caches. No credentials or paid API calls.

[Open the Colab notebook](https://colab.research.google.com/github/ByteBoss123/Speech-Command-/blob/codex/siri-real-speech-lab-20261119/notebooks/Real_Speech_Lab.ipynb).

Test the exported adapter on a local English recording, up to 30 seconds:

```bash
python -m speechlab.transcribe recording.wav --adapter evidence/2026-11-19/asr/adapter
```

This demonstrates the trained artifact. The pretrained model had better quality in this pilot.

## Reproducibility and limitations

- [Manifest](evidence/2026-11-19/data/manifest.jsonl): every sampled ID, speaker, split,
  transcript, duration, and audio SHA-256. Audio is downloaded, not committed.
- [Dataset card](evidence/2026-11-19/data/dataset_card.json): exact dataset revision,
  source rows scanned, and quotas. Sampling uses a deterministic source prefix with
  speaker quotas, **not a representative random sample**. Official source splits are retained.
- WER/CER use Whisper's English normalization and corpus-level edit counts. WER can
  exceed 100%. Confidence intervals resample speakers, not individual clips.
- Validation selects the ASR checkpoint and speaker threshold. Test never trains the
  models or calibrates the threshold. Test EER is a descriptive curve summary only.
- Foundation-model pretraining overlap is unknown. No claim of previously unseen
  foundation-model speakers, conversational performance, or real environmental-noise robustness.
- Twenty-one tests check leakage, checksums, quotas, reader cleanup, audio resampling,
  noise SNR, trial labels, threshold ties, WER aggregation, clustered bootstrap,
  adapter-only weight updates, cached-encoder equivalence, checkpoint reload,
  paired intent splits, padding-aware pooling, training-only normalization, fusion gradients,
  and exclusion of source record content from the public intent index.
- The first streaming reader wrote correct files but crashed at interpreter shutdown.
  The final synchronous Parquet reader reproduced all 224 records and hashes and exited normally.

`speechlab/` contains implementation; `tests_speech/` contains offline tests;
`notebooks/` contains the runner; `evidence/2026-11-19/` contains measured artifacts.
[Portfolio audit](docs/PORTFOLIO_AUDIT.md) · [Evidence-based resume wording](docs/RESUME_PROJECTS.md).

Future experiments should select by generated validation WER, investigate more training
speakers and conservative learning rates on development data, and use cross-chapter
verification trials. New claims after this test inspection need a fresh holdout.
Those experiments have **not** been run here.

## Sources

- [LibriSpeech / OpenSLR](https://www.openslr.org/12/) and [dataset card](https://huggingface.co/datasets/openslr/librispeech_asr): CC BY 4.0.
- [Whisper documentation](https://huggingface.co/docs/transformers/v4.57.1/model_doc/whisper).
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).
- [WavLM speaker-verification checkpoint](https://huggingface.co/microsoft/wavlm-base-plus-sv).
- [MInDS-14 dataset](https://huggingface.co/datasets/PolyAI/minds14) (CC BY 4.0) and [paper](https://aclanthology.org/2021.emnlp-main.591/).
- [Multilingual MiniLM text encoder](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

Implemented with AI assistance. Measured limitations are preserved for reproduction and review.
