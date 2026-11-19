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

## Legal Case-Holding LoRA | Python, PyTorch, Qwen2.5, PEFT

- Implemented Qwen2.5-1.5B LoRA fine-tuning for five-way legal case-holding selection,
  with answer-token loss, zero-shot and Legal-BERT comparisons, and paired bootstrap
  evaluation on CaseHOLD.

The existing repository reports 5,000 training examples and 77.4% test accuracy
versus 58.1% zero-shot on 3,600 examples. The inspected notebook has no saved outputs
and its default branch did not include the GPU result archive. Recover that archive
before presenting these numerical gains as independently verified artifacts.

These describe repository work, including today's AI-assisted extension. Use them
when you can accurately explain your contribution, code, splits, evaluation, and
limitations. These projects are not employment experience.
