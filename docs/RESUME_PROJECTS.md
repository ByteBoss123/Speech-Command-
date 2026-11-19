# Three implemented speech projects — evidence-based resume wording

## Speech Model Adaptation and Selection | Python, PyTorch, Whisper, PEFT

- Trained 98K decoder LoRA parameters on 84 minutes of speech, reducing clean WER
  from 6.42% to 5.91% on 120 held-out recordings; benchmarked and selected Whisper-base
  on validation, achieving 4.66% clean WER and 9.51% WER under 10 dB Gaussian noise.

Interview detail: 384 training recordings / 48 speakers; validation and test each
have 120 recordings / 20 new speakers. Epoch 2 was selected by generated validation
WER. The larger pretrained model's gain is a capacity/selection gain, not LoRA
training. Tiny LoRA's clean gain has a speaker-bootstrap interval excluding zero;
its noisy-speech gain does not. The first smaller pilot regressed and remains recorded.

## Cross-Chapter Speaker Verification | Python, WavLM, PyTorch

- Implemented multi-recording speaker enrollment with frozen WavLM embeddings,
  reducing false rejection from 36.7% to 10.0% at a 0.96% observed false-accept rate
  across 1,200 held-out cross-chapter trials from 20 speakers.

Interview detail: 60 same-speaker and 1,140 different-speaker trials. Enrollment uses
three recordings; probes come from another chapter. Method and threshold were
selected on different validation speakers at a 1% empirical FAR target. Single-recording
enrollment had 0.70% FAR, so lower rejection came with a small FAR increase. Trials
are dependent; no production authentication or spoof-resistance claim is supported.

## Multilingual Speech–Text Intent Routing | Python, PyTorch, Whisper, MiniLM

- Trained and ensembled classifiers over frozen audio/text encoders on 1,135
  recordings across English, Spanish and French, achieving 97.6% accuracy and 0.976
  macro F1 on 210 fresh held-out requests with three-seed and modality comparisons.

Interview detail: 14 banking intents, 168 validation recordings, and 70 test examples
per language. Validation chose a 75% text / 25% joint-ensemble probability blend.
The unchanged previous fusion model scored 95.7% on the same new holdout. The pure
joint ensemble scored 98.1%, but was not selected retrospectively using test results.
The gain over v1 remains uncertain; speaker IDs are absent and supplied transcripts
are inputs. This trains classification heads, not a generative multimodal LLM.

These describe AI-assisted repository projects, not employment experience. Use the
bullets when you can explain the implementation, contribution, data decisions and
limitations. Exact artifacts are under `evidence/2026-11-19-v2/`.
