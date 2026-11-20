"""Recompute the v3 evaluation from public sufficient statistics, offline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .common import fingerprint
from .intent import classification_metrics
from .intent_external import comparison
from .metrics import approximate_eer, calibrate_threshold
from .speaker_v2 import clustered_comparison, rates
from .speaker_v3 import CANDIDATES, select
from .speaker_normalized import METHODS, choose
from .verify_v2 import equal, read, verify_asr


def expand(rows):
    return {name: [{k: r[k] for k in ("probe_id", "probe_speaker", "enrollment_speaker", "label")} |
                   {"score": r["scores"][name]} for r in rows] for name in rows[0]["scores"]}


def verify_speaker(root):
    report, choice = read(root, "metrics.json"), read(root, "selection.json")
    dev, test = expand(read(root, "trials_validation.json")), expand(read(root, "trials_test.json"))
    extended = "backend_training_recordings" in report
    for name in METHODS if extended else CANDIDATES:
        values = dev[name]
        equal(calibrate_threshold([r["label"] for r in values], [r["score"] for r in values], .005), choice["calibration"][name], "speaker calibration")
    selected = choose(choice["calibration"]) if extended else select(choice["calibration"])
    equal(selected, report["selected_method"], "speaker method")
    equal(selected, choice["method"], "persisted speaker choice")
    equal(choice["frozen_thresholds"], report["thresholds"], "speaker frozen thresholds")
    thresholds = report["thresholds"]
    for name, trials in test.items():
        equal(rates(trials, thresholds[name]), report["models"][name]["at_frozen_threshold"], "speaker rates")
        equal(approximate_eer([r["label"] for r in trials], [r["score"] for r in trials]), report["models"][name]["descriptive_eer"], "speaker EER")
    for name in ("v2_single", "v2_three"):
        equal(rates(dev[name], thresholds[name]), choice["baseline_validation"][name], "baseline development")
        equal(clustered_comparison(test[name], test[selected], thresholds[name], thresholds[selected], seed=31),
              report[f"selected_minus_{name}"], "speaker paired CI")
    config = read(root, "verification_config.json")
    equal(config["threshold"], thresholds[selected], "inference threshold")
    equal(config["enrollment_recordings"], 5, "inference enrollment size")
    if extended:
        initial = read(root, "metrics_initial.json")
        for name, values in initial["models"].items(): equal(values, report["models"][name], "unchanged initial speaker result")
        if config["method"] == "wccn":
            with np.load(root / config["normalizer_path"], allow_pickle=False) as params:
                if not np.isfinite(params["transform"]).all() or params["transform"].shape != (len(params["mean"]), len(params["mean"])):
                    raise ValueError("Invalid normalization artifact")
    return {"status": "passed", "selected": selected, "trials": len(test[selected])}


def verify_intent(root):
    report, rows = read(root, "metrics.json"), read(root, "predictions.json")
    truth, languages = np.array([r["label"] for r in rows]), np.array([r["language"] for r in rows])
    predictions = {name: np.array([r["predicted"][name] for r in rows]) for name in report["models"]}
    for name, pred in predictions.items():
        if np.any((pred < 0) | (pred >= 14)): raise ValueError("Invalid predicted intent")
        equal(classification_metrics(truth, pred), report["models"][name]["test"], "external intent scores")
        for lang in sorted(set(languages)):
            take = languages == lang
            equal(classification_metrics(truth[take], pred[take]), report["models"][name]["by_language"][lang], "external language scores")
    for name, baseline in (("primary_selected_minus_previous", "previous_v1_fusion"), ("secondary_selected_minus_text", "text")):
        equal(comparison(truth, predictions["selected_blend"], predictions[baseline], languages), report[name], "external paired statistics")
    lock = read(root, "frozen_models.json")
    for directory, checksum in lock["model_weight_sha256"].items():
        equal(fingerprint(Path(directory) / "head.safetensors"), checksum, "frozen intent weights")
    equal(fingerprint(Path(lock["model_directory"]) / "ensemble_config.json"), lock["config_sha256"], "frozen intent config")
    equal(fingerprint(lock.get("protocol_path", "experiments/v3_protocol.json")), lock["protocol_sha256"], "frozen protocol")
    equal(len(rows), read(root, "dataset_card.json")["total"], "external sample size")
    return {"status": "passed", "recordings": len(rows), "models_unchanged": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", nargs="?", default="evidence/2026-11-20-v3")
    a = p.parse_args(); root = Path(a.directory)
    for name, checksum in read(root, "SHA256SUMS.json").items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError("Unsafe checksum path")
        equal(fingerprint(path), checksum, "bundle checksum")
    family = read(root, "external_comparison_family.json")
    for name, values in family["comparisons"].items():
        pvalue = read(root / name, "metrics.json")["primary_selected_minus_previous"]["exact_mcnemar_two_sided_p"]
        equal(values, {"raw_p": pvalue, "bonferroni_adjusted_p": min(1., 2 * pvalue), "significant_at_family_alpha_0_05": 2 * pvalue < .05}, "external multiplicity adjustment")
    print(json.dumps({"asr": verify_asr(root / "asr", seed=31), "speaker": verify_speaker(root / "speaker"),
                      "intent_english": verify_intent(root / "intent_external"),
                      "intent_unseen_languages": verify_intent(root / "intent_unseen_languages")}, indent=2))


if __name__ == "__main__": main()
