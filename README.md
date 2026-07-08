# SpeechCommandNet: Isolated Word Recognition Trained from Scratch

## What this is

A small convolutional neural network, trained **from scratch** (no
pretrained weights, no transfer learning, no fine-tuning) in PyTorch, to
classify isolated spoken words (digits zero-nine plus yes/no/stop/go) from
MFCC features. Training data is entirely synthetic, generated with
espeak-ng across 7 distinct voices (4 formant-synthesis accents + 3
mbrola diphone voices, spanning genders and English varieties).

The central design decision, and the reason this project is more than a
toy classifier: **two full voices are held out entirely from training**
and used only for a final generalization test. This directly tests
whether the model learned the words or just memorized the specific
voices it was trained on.

## What this is NOT

This is **not** a fine-tuned foundation model, and not a claim of
production-scale ASR. It is a small-vocabulary (14-word), isolated-word
(not continuous speech) classifier, trained on ~630 synthetic samples on
CPU in under a minute. Real production ASR systems (Whisper, wav2vec2,
etc.) are trained on thousands of hours of real human speech at a scale
this project doesn't approach. This project demonstrates genuine,
hands-on neural network training applied to a speech task, evaluated
rigorously -- it does not demonstrate foundation-model-scale experience.

## Why synthetic data and small scale

This was built in a sandboxed environment without GPU access and without
network access to model/dataset hosting (e.g., Hugging Face Hub) --
only package indexes (PyPI, npm) were reachable. Downloading a real
speech corpus or a pretrained checkpoint to fine-tune was not possible.
Given that constraint, the honest options were: (a) don't do hands-on
speech model training at all, or (b) build something smaller but real,
entirely from resources available offline. This project is (b), and the
scope above reflects that constraint rather than a claim that this is
what production ASR development looks like.

## Methodology

```
corpus.py         - 14-word vocabulary; 5 voices for train/val,
                     2 voices held out entirely for the generalization test
synth.py          - espeak-ng wrapper with voice/rate/pitch control
generate_dataset.py - synthesizes all 882 (word, voice, rate, pitch) clips
dataset.py        - MFCC feature extraction (40 coefficients, fixed-length
                     48 frames), stratified sample-level train/val split
model.py          - small CNN: 2 conv+pool blocks -> FC -> 14-way softmax
train.py          - training loop, real per-epoch train/val loss & accuracy
evaluate.py       - the actual test: accuracy on the 2 held-out unseen voices
tests/            - 5 unit tests, including an explicit voice-leakage check
```

Data split:
- **Train**: 504 clips, 5 voices (en-us, en-gb, en-gb-x-rp, mb-us1, mb-us2)
- **Val**: 126 clips, same 5 voices (in-distribution, held-out samples)
- **Unseen-voice test**: 252 clips, 2 voices never seen in training
  (en-gb-scotland, mb-us3)

## Results

- **In-distribution validation accuracy: 100%** by epoch 7. This is
  expected and not the interesting number -- small vocabulary, clean
  synthetic audio, same voices as training.
- **Unseen-voice test accuracy: 95.2%** (240/252) -- the real
  generalization result. Broken down by voice:
  - en-gb-scotland (unseen accent): 93.7%
  - mb-us3 (unseen diphone voice, same accent family as training voices): 96.8%

The ~5-point gap between in-distribution and unseen-voice accuracy is a
genuine, honest finding: the model generalizes well but not perfectly to
acoustic conditions it never trained on, and generalizes slightly worse
to an unseen *accent* than to an unseen *voice within a familiar accent
family* -- consistent with the intuitive expectation that accent shift is
a harder generalization case than speaker/synthesis-method shift.

**Error analysis:** the dominant confusion was "go" being misclassified
as "four" (8 of 12 total errors). Both are short, single-syllable/short
words; this is the kind of specific, actionable failure a real error
analysis should surface, rather than reporting only an aggregate accuracy
number.

## Honest scope note

This project demonstrates: hands-on PyTorch model design and training
from scratch, MFCC feature engineering, and rigorous generalization
testing (holding out entire voices, not just samples) for a speech task.
It does not demonstrate: foundation-model fine-tuning, continuous-speech
ASR, training on real human speech, or production-scale deployment. Any
resume bullet drawn from this should reflect that scope accurately.

## Running it

```bash
pip install torch librosa matplotlib
apt-get install espeak-ng sox mbrola mbrola-us1 mbrola-us2 mbrola-us3

python generate_dataset.py   # ~10 seconds, 882 clips
python train.py              # ~1-2 minutes on CPU, 40 epochs
python evaluate.py           # unseen-voice generalization test
python -m pytest tests/      # 5 tests, includes voice-leakage check
```
