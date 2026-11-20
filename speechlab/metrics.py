from __future__ import annotations

import itertools
import random
from collections import defaultdict

import jiwer
import numpy as np


def transcript_counts(reference, prediction):
    result = jiwer.process_words(reference, prediction)
    chars = jiwer.process_characters(reference, prediction)
    return {"word_errors": result.substitutions + result.deletions + result.insertions,
            "reference_words": result.hits + result.substitutions + result.deletions,
            "character_errors": chars.substitutions + chars.deletions + chars.insertions,
            "reference_characters": chars.hits + chars.substitutions + chars.deletions}


def asr_summary(predictions):
    if not predictions:
        raise ValueError("No predictions")
    words = sum(p["reference_words"] for p in predictions)
    characters = sum(p["reference_characters"] for p in predictions)
    if words <= 0 or characters <= 0:
        raise ValueError("References must contain words and characters")
    return {"wer": sum(p["word_errors"] for p in predictions) / words,
            "cer": sum(p["character_errors"] for p in predictions) / characters,
            "reference_words": words, "utterances": len(predictions),
            "speakers": len({p["speaker_id"] for p in predictions})}


def paired_wer_bootstrap(base, adapted, seed=13, repetitions=2000):
    if len(base) != len(adapted) or not base:
        raise ValueError("Paired predictions required")
    groups = defaultdict(lambda: [0, 0, 0])
    seen = set()
    for a, b in zip(base, adapted):
        if a["id"] != b["id"] or a["speaker_id"] != b["speaker_id"] or a["reference_words"] != b["reference_words"]:
            raise ValueError("Prediction pairing mismatch")
        if a["id"] in seen:
            raise ValueError("Duplicate prediction ID")
        seen.add(a["id"])
        g = groups[a["speaker_id"]]
        g[0] += a["word_errors"]
        g[1] += b["word_errors"]
        g[2] += a["reference_words"]
    if len(groups) < 2:
        raise ValueError("At least two speaker clusters required")
    counts = np.asarray(list(groups.values()), dtype=float)
    rng = np.random.default_rng(seed)
    totals = counts[rng.integers(len(counts), size=(repetitions, len(counts)))].sum(axis=1)
    diffs = (totals[:, 1] - totals[:, 0]) / totals[:, 2]
    return {"adapted_minus_base_wer": asr_summary(adapted)["wer"] - asr_summary(base)["wer"],
            "ci95": np.quantile(diffs, [.025, .975]).tolist(), "resampling_unit": "speaker",
            "resamples": repetitions, "seed": seed}


def make_trials(rows, seed=13):
    by_speaker = defaultdict(list)
    for i, row in enumerate(rows):
        by_speaker[str(row["speaker_id"])].append(i)
    positives = [(i, j, 1) for ids in by_speaker.values() for i, j in itertools.combinations(ids, 2)]
    negatives = [(i, j, 0) for i, j in itertools.combinations(range(len(rows)), 2)
                 if rows[i]["speaker_id"] != rows[j]["speaker_id"]]
    if not positives or not negatives:
        raise ValueError("Need same-speaker and different-speaker trials")
    rng = random.Random(seed)
    n = min(len(positives), len(negatives))
    trials = rng.sample(positives, n) + rng.sample(negatives, n)
    rng.shuffle(trials)
    return trials


def verification_rates(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Invalid verification inputs")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Both positive and negative labels are required")
    positive, negative = labels == 1, labels == 0
    accepted = scores >= threshold
    fa, fr = int((accepted & negative).sum()), int((~accepted & positive).sum())
    return {"far": fa / int(negative.sum()), "frr": fr / int(positive.sum()),
            "false_accepts": fa, "false_rejects": fr,
            "negative_trials": int(negative.sum()), "positive_trials": int(positive.sum())}


def calibrate_threshold(labels, scores, target_far=.05):
    if not 0 <= target_far <= 1:
        raise ValueError("target_far must be in [0, 1]")
    scores = np.asarray(scores, dtype=float)
    verification_rates(labels, scores, 0)  # validates labels and finiteness
    # Lowest feasible threshold maximizes acceptance subject to the empirical FAR cap.
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    for threshold in candidates:
        rates = verification_rates(labels, scores, threshold)
        if rates["far"] <= target_far:
            return {"threshold": float(threshold), "target_far": target_far, **rates}
    raise AssertionError("A reject-all threshold must exist")


def approximate_eer(labels, scores):
    scores = np.asarray(scores, dtype=float)
    verification_rates(labels, scores, 0)
    rates = [verification_rates(labels, scores, t) for t in np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]]
    nearest = min(rates, key=lambda x: abs(x["far"] - x["frr"]))
    return {"eer": (nearest["far"] + nearest["frr"]) / 2,
            "method": "mean FAR/FRR at the empirical ROC point with minimum absolute difference"}
