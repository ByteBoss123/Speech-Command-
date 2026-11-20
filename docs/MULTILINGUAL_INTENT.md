# Multilingual speech–text intent router — v1 pilot

The expanded experiment and current interpretation are in [V2 results](V2_RESULTS.md).
This page preserves the original pilot protocol and measurements.

This third project classifies 14 banking requests from paired audio and text in
English (`en-US`), Spanish (`es-ES`) and French (`fr-FR`). It is implemented, trained,
and evaluated on real recordings; the saved classification heads can be reloaded.

## Architecture and training

1. Resample audio from the source 8 kHz to 16 kHz. Encode with frozen Whisper-tiny
   and pool the mean and standard deviation over valid output frames: 768 features.
   Padding frames are excluded from pooling; Whisper's encoder still sees its
   normal 30-second padded input. This is not streaming inference.
2. Encode the dataset's native-language `transcription` with frozen multilingual
   MiniLM; attention-mask mean pooling yields 384 features, at most 128 tokens.
   None of the selected transcripts required truncation. The dataset's English
   translation, intent name, language ID, and sample ID are never model inputs.
3. Standardize each feature using **training rows only**, concatenate audio/text,
   and train a 128-unit GELU classifier with dropout 0.2 and 14 logits. The
   normalization buffers are part of the saved head, preventing inference drift.

Whisper's encoder has 8,208,384 frozen parameters; MiniLM has 117,653,760. The
fusion head trains **149,390** parameters. Audio-only and text-only models use the
same hidden width, optimizer, data and selection rule, but naturally have different
parameter counts (100,238 and 51,086). This is not a parameter-matched ablation.

AdamW: learning rate 0.001, weight decay 0.01, batch 32, gradient norm limit 1.0,
seed 13, up to 80 epochs, patience 12. Select the highest validation macro F1, using
lower validation cross-entropy to break ties. All three checkpoints are finalized
before test scoring. Selected epochs: audio 20, text 43, fusion 9. No test-driven
hyperparameter search or repeated seeds were run.

## Data protocol

Source: [PolyAI MInDS-14](https://huggingface.co/datasets/PolyAI/minds14), revision
`40ce77cb32a384e4d50a568e1ec39ac804019d33`, CC BY 4.0.

The source exposes only a `train` partition. This project creates its own deterministic
utterance splits after filtering to 1–30 seconds and removing exact source-audio and
normalized-transcript duplicates. SHA-256 ordering with seed 13 chooses equal quotas
for each language/intent combination. Converted audio checksums are checked again.

| Split | Per language / intent | Per language | Total |
|---|---:|---:|---:|
| Train | 8 | 112 | 336 |
| Validation | 3 | 42 | 126 |
| Test | 3 | 42 | 126 |

These are **custom balanced splits**, not official MInDS benchmark results. The
schema does not expose speaker IDs: speakers may overlap, and speaker independence
cannot be assessed. Near-duplicate phrasing and pretrained-model data overlap are
also possible. All three languages occur in training; no zero-shot transfer is tested.

## Executed results

| Input | Test correct | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Audio | 71/126 | 56.35% | 0.5527 |
| Text | 118/126 | 93.65% | 0.9371 |
| Audio + text | 120/126 | 95.24% | 0.9518 |

| Language | Audio accuracy | Text accuracy | Fusion accuracy |
|---|---:|---:|---:|
| English | 61.90% | 88.10% | 92.86% |
| Spanish | 59.52% | 100.00% | 100.00% |
| French | 47.62% | 92.86% | 92.86% |

Each language has only **42** test recordings. The fusion-minus-text accuracy
difference is **+1.59 percentage points**, with a descriptive 95% bootstrap interval
of **[−2.38, +6.35]** points (2,000 paired resamples within language). This interval
crosses zero and does not account for unknown speaker clustering. A reliable fusion
advantage has **not** been established. Validation macro F1 was higher for text only.
The simpler text model is therefore a credible alternative when a transcript is available.

Replacing the fused model's audio embedding with its training mean gives 92.06%
accuracy; replacing text gives 26.98%. These interventions demonstrate dependence
on both feature streams, with much stronger text dependence. They are not models
trained for missing inputs, nor evidence of robustness to absent modalities.

Feature extraction took 162.5 seconds and head training/evaluation 2.5 seconds on
this CPU run with four Torch threads. These include differing stages and are **not**
production latency benchmarks. Use the raw run metadata for the exact environment.

## Reproduce and use

Install `requirements-speech.txt` as described in the root README. From the repository:

```bash
python -m speechlab.intent_data
python -m speechlab.intent_features
python -m speechlab.intent
python -m pytest tests_speech -q
python -m speechlab.verify_intent evidence/2026-11-19/intent

python -m speechlab.intent_predict recording.wav \
  --transcript "How can I freeze my card?" \
  --model evidence/2026-11-19/intent/fusion
```

Use a transcript corresponding to your recording. This CLI does not generate a
transcript, perform actions, reject unknown intents or calibrate probabilities. It
returns the top three closed-set predictions. Audio must be between 1 and 30 seconds.

The encoder revisions are pinned in `speechlab/intent_features.py`; the head loader
checks encoder compatibility. Data preparation, feature extraction and training
require new/empty output directories to protect earlier evidence. To rerun, use
new `--output` paths and pass matching `--manifest` and `--features` paths downstream.
The notebook provides the same commands, shows raw scores, and runs exported inference.

Saved in `evidence/2026-11-19/intent/`: three trained heads with normalization and
configuration; epoch histories; full test probabilities; per-language/per-intent
metrics; confusion matrices; ablations; a minimal split/label index and dataset card;
encoder provenance; and a summary of an actual CLI inference result. Raw transcripts,
recording identifiers, audio paths and checksums are kept in the locally generated
manifest rather than republished. Audio and pretrained encoders are downloaded from
their sources. The feature cache is regenerated locally.

The offline verifier recomputes metrics and bootstrap intervals from predictions,
checks the selected epochs against validation histories, and compares serialized
normalization and parameter counts. To also check the full local manifest, training-only
normalization and replay all saved heads against the local feature cache, run:

```bash
python -m speechlab.verify_intent evidence/2026-11-19/intent \
  --features runs/intent-features --manifest data/minds14-pilot/manifest.jsonl
```

Unit tests additionally exercise leakage
rejection, paired splitting, padded pooling, train-only statistics, gradients through
both modalities, and checkpoint round trips.

## Relevance and limits

This demonstrates Python/PyTorch model training, speech/text fusion, multilingual
evaluation and reproducible experiments. It does **not** demonstrate generative
multimodal LLM training, large-scale training, assistant-directed-speech detection,
end-to-end ASR quality or production serving. Future improvement claims need a fresh,
larger holdout with speaker IDs and repeated training seeds.

Sources: [MInDS-14 paper](https://aclanthology.org/2021.emnlp-main.591/),
[dataset card](https://huggingface.co/datasets/PolyAI/minds14),
[Whisper checkpoint](https://huggingface.co/openai/whisper-tiny),
[multilingual MiniLM card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
Implementation and evaluation were completed with AI assistance.
