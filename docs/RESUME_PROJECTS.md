# Evidence-based project wording for the Siri Speech application

## Real-Speech Model Adaptation | Python, PyTorch, Whisper, PEFT

- Implemented and benchmarked decoder LoRA adaptation of a 37.8M-parameter speech
  model on real LibriSpeech recordings, training 98,304 adapter parameters and evaluating
  clean/noisy transcription with WER, CER, and speaker-cluster bootstrap intervals.

Interview detail: 128 training clips, 16 training speakers; 48 test clips, 12 test
speakers. Adaptation regressed: clean WER 5.61% to 11.52%. Validation token loss
improved while generated-transcript quality worsened; retain the baseline as the
better model. Do not claim improved ASR or training a large conversational speech LLM.

## Speaker Verification Evaluation | Python, WavLM, Transformers

- Implemented a speaker-verification evaluation with separate calibration and test
  speakers, measuring 1.4% false accepts and 4.2% false rejects across 144 held-out
  same/different-speaker trials using a frozen WavLM model.

Interview detail: 12 test speakers, 72 positive and 72 negative trials. Threshold
0.91006 came from validation speakers. All positive test pairs shared a chapter,
and trials reused recordings; avoid production authentication claims.

## Multilingual Speech–Text Intent Router | Python, PyTorch, Whisper, MiniLM

- Trained a 149K-parameter audio–text fusion classifier for 14 spoken banking intents
  across English, Spanish, and French, achieving 95.2% accuracy and 0.952 macro F1 on
  a 126-utterance holdout with audio-only and text-only baselines.

Interview detail: MInDS-14; 336 training and 126 validation recordings; 588 total.
Whisper and multilingual MiniLM encoders are frozen; the fusion head is trained.
The model receives a dataset-provided transcript along with audio. Text only scored
93.65%, audio only 56.35%; the 1.59-point fusion lead has an interval crossing zero.
Speaker IDs are unavailable, so do not describe the split as speaker independent.
This is multilingual classification, not generative multimodal LLM training.

These describe repository work, including today's AI-assisted extension. Use them
when you can accurately explain your contribution, code, splits, evaluation, and
limitations. These projects are not employment experience.
