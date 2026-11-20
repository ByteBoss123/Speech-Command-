"""Recompute the v2 scorecards from public error counts and predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .asr_v2 import select_checkpoint
from .intent import classification_metrics, paired_bootstrap
from .intent_v2 import select_fusion
from .metrics import approximate_eer, asr_summary, calibrate_threshold, paired_wer_bootstrap
from .speaker_v2 import METHODS, clustered_comparison, rates
from .select_asr import select


def read(root, name): return json.loads((root / name).read_text())


def equal(actual, expected, name):
    if actual != expected: raise ValueError(f"Evidence mismatch: {name}")


def verify_asr(root, seed=19):
    report = read(root, "metrics.json")
    history = read(root, "training_history.json")
    equal(select_checkpoint(history)["epoch"], report["selected_epoch"], "ASR checkpoint")
    for row in history:
        predictions = read(root, f"development_epoch_{row['epoch']}.json")
        equal(asr_summary(predictions)["wer"], row["validation_wer"], "development WER")
    for condition, scores in report["results"].items():
        base = read(root, f"predictions_{condition}_base.json")
        selected = read(root, f"predictions_{condition}_selected.json")
        equal(asr_summary(base), scores["base"], "base ASR metrics")
        equal(asr_summary(selected), scores["selected"], "selected ASR metrics")
        equal(paired_wer_bootstrap(base, selected, seed), scores["paired_comparison"], "ASR bootstrap")
    return {"conditions": list(report["results"]), "selected_epoch": report["selected_epoch"], "status": "passed"}


def verify_speaker(root):
    report = read(root, "metrics.json")
    dev, test = read(root, "trials_validation.json"), read(root, "trials_test.json")
    for method in METHODS:
        calibration = report["calibration"][method]
        equal(calibrate_threshold([r["label"] for r in dev[method]], [r["score"] for r in dev[method]], calibration["target_far"]), calibration, "speaker calibration")
        equal(rates(test[method], calibration["threshold"]), report["models"][method]["at_frozen_threshold"], "speaker rates")
        equal(approximate_eer([r["label"] for r in test[method]], [r["score"] for r in test[method]]), report["models"][method]["descriptive_eer"], "speaker EER")
        if any(not r["cross_chapter_positive"] for r in test[method] if r["label"]): raise ValueError("Within-chapter positive trial")
    chosen = min(METHODS, key=lambda m: (report["calibration"][m]["frr"], report["calibration"][m]["far"], METHODS.index(m)))
    equal(chosen, report["selected_method"], "speaker method selection")
    equal(clustered_comparison(test["single_raw"], test[chosen], report["calibration"]["single_raw"]["threshold"], report["calibration"][chosen]["threshold"]), report["paired_selected_minus_single"], "speaker bootstrap")
    return {"trials_per_method": len(test[chosen]), "selected_method": chosen, "status": "passed"}


def verify_intent(root):
    report = read(root, "metrics.json")
    dev, test = read(root, "predictions_validation.json"), read(root, "predictions_test.json")
    arrays = {k: np.array([r["probabilities"][k] for r in test], dtype=np.float32) for k in report["models"]}
    truth = np.array([r["label"] for r in test]); languages = np.array([r["language"] for r in test])
    for mode, values in arrays.items():
        if not np.isfinite(values).all() or not np.allclose(values.sum(1), 1) or np.any(values < 0): raise ValueError("Invalid intent probabilities")
        equal(classification_metrics(truth, values.argmax(1)), report["models"][mode]["test"], "intent metrics")
        for lang in sorted(set(languages)):
            take = languages == lang
            equal(classification_metrics(truth[take], values[take].argmax(1)), report["models"][mode]["by_language"][lang], "language scores")
    dev_truth = np.array([r["label"] for r in dev])
    selected, candidates = select_fusion(np.array([r["probabilities"]["text"] for r in dev], dtype=np.float32),
                                        np.array([r["probabilities"]["fusion"] for r in dev], dtype=np.float32), dev_truth)
    equal(selected, read(root, "selection.json")["selected"], "fusion selection")
    equal(candidates, read(root, "selection.json")["candidates"], "fusion candidate metrics")
    alpha = selected["alpha"]
    if not np.allclose(arrays["selected_blend"], (1 - alpha) * arrays["text"] + alpha * arrays["fusion"], atol=1e-7):
        raise ValueError("Selected probability blend mismatch")
    for key, other in (("selected_minus_previous", "previous_v1_fusion"), ("selected_minus_text", "text")):
        equal(paired_bootstrap(truth, arrays["selected_blend"].argmax(1), arrays[other].argmax(1), languages, seed=19), report[key], "intent bootstrap")
    index = read(root, "evaluation_index.json")
    equal([r for r in index if r["split"] == "test"], [{k: r[k] for k in ("id", "split", "language", "label")} for r in test], "test index")
    return {"test_examples": len(test), "fusion_weight": alpha, "status": "passed"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", nargs="?", default="evidence/2026-11-19-v2")
    a = p.parse_args(); root = Path(a.directory)
    capacity = read(root / "asr_capacity", "metrics.json")
    equal(asr_summary(read(root / "asr_capacity", "predictions_validation.json"))["wer"], capacity["validation_wer"], "capacity validation WER")
    for condition, scores in capacity["results"].items():
        equal(asr_summary(read(root / "asr_capacity", f"predictions_{condition}.json")), scores, "capacity test WER")
    equal(select(root / "asr", root / "asr_capacity"), read(root, "asr_selection.json"), "model-size selection")
    print(json.dumps({"asr": verify_asr(root / "asr"), "asr_capacity": "passed", "speaker": verify_speaker(root / "speaker"), "intent": verify_intent(root / "intent")}, indent=2))


if __name__ == "__main__": main()
