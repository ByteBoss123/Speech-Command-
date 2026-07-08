"""
corpus.py

Defines the vocabulary and the synthesis grid (voice x rate x pitch) for
SpeechCommandNet, a small isolated-word speech classifier trained from
scratch.

Critically, the train/val split and the generalization-test split are
separated at the *voice* level, not just the sample level: two voices
(one formant-synthesis, one diphone/mbrola) are held out entirely from
training and only ever appear in the "unseen-voice" test set. This is
the same discipline used elsewhere in this portfolio (never trust an
aggregate number without checking whether the split actually tests
generalization) applied to acoustic domain shift specifically.
"""

VOCAB = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "yes", "no", "stop", "go",
]

# Voices available via espeak-ng: 4 formant-synthesis voices (different accents)
# + 3 mbrola diphone voices (different synthesis method entirely, different
# genders). This is genuine acoustic diversity, not just the same voice at
# different speeds.
TRAIN_VOICES = ["en-us", "en-gb", "en-gb-x-rp", "mb-us1", "mb-us2"]
UNSEEN_VOICES = ["en-gb-scotland", "mb-us3"]  # held out entirely from training

RATES = [130, 175, 220]   # slow, normal, fast (words per minute)
PITCHES = [30, 50, 70]    # low, normal, high (0-99 scale)


def grid(voices):
    """All (voice, rate, pitch) combinations for the given voice list."""
    combos = []
    for v in voices:
        for r in RATES:
            for p in PITCHES:
                combos.append((v, r, p))
    return combos


def flat_dataset_spec():
    """Returns (split, word, voice, rate, pitch, clip_id) rows for the full
    synthesis grid: 'trainval' rows use TRAIN_VOICES (further split into
    train/val by the dataset builder), 'unseen' rows use UNSEEN_VOICES and
    are never touched until final evaluation."""
    rows = []
    for split, voices in [("trainval", TRAIN_VOICES), ("unseen", UNSEEN_VOICES)]:
        for voice, rate, pitch in grid(voices):
            for word in VOCAB:
                clip_id = f"{word}__{voice}__r{rate}__p{pitch}"
                rows.append((split, word, voice, rate, pitch, clip_id))
    return rows
