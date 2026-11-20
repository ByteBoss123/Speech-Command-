"""Export an allowlisted evaluation bundle without raw records or embeddings."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from .asr_capacity import MODEL, REVISION
from .common import dump_json, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="evidence/2026-11-20-v3")
    a = p.parse_args(); out = Path(a.output)
    if out.exists() and any(out.iterdir()): p.error("Output must be empty")
    for folder in ("asr-v3", "speaker-normalized-v3", "intent-external-v3", "intent-unseen-v3"):
        if not (Path("runs") / folder / "metrics.json").is_file(): raise ValueError(f"Incomplete run: {folder}")
    asr = Path("runs/asr-v3")
    report = json.loads((asr / "metrics.json").read_text())
    history = json.loads((asr / "training_history.json").read_text())
    allowlists = {
        "asr": (asr, ["metrics.json", "run_config.json", "training_history.json", "selection.json",
             "adapter/adapter_config.json", "adapter/adapter_model.safetensors"] +
             [f"development_epoch_{r['epoch']}.json" for r in history] +
             [f"predictions_{condition}_{arm}.json" for condition in report["results"] for arm in ("base", "selected")]),
        "speaker": (Path("runs/speaker-normalized-v3"), ["metrics.json", "selection.json", "verification_config.json", "extension_record.json", "trials_validation.json", "trials_test.json"]),
        "intent_external": (Path("runs/intent-external-v3"), ["metrics.json", "predictions.json", "frozen_models.json", "dataset_card.json"]),
        "intent_unseen_languages": (Path("runs/intent-unseen-v3"), ["metrics.json", "predictions.json", "frozen_models.json", "dataset_card.json"])
    }
    speaker_config = json.loads(Path("runs/speaker-normalized-v3/verification_config.json").read_text())
    if "normalizer_path" in speaker_config: allowlists["speaker"][1].append(speaker_config["normalizer_path"])
    for directory, (source, files) in allowlists.items():
        for name in files:
            target = out / directory / name; target.parent.mkdir(parents=True, exist_ok=True)
            if name in ("trials_validation.json", "trials_test.json", "predictions.json"):
                target.write_text(json.dumps(json.loads((source / name).read_text()), separators=(",", ":")) + "\n")
            else: shutil.copyfile(source / name, target)
    for name in ("metrics", "selection"):
        shutil.copyfile(f"runs/speaker-v3/{name}.json", out / "speaker" / f"{name}_initial.json")
    shutil.copyfile("data/librispeech-v3/dataset_card.json", out / "speech_dataset_card.json")
    dump_json(out / "asr_selection.json", {"model": MODEL, "revision": REVISION,
        "uses_adapter": report["selected_epoch"] != 0, "adapter_path": "asr/adapter", "selected_epoch": report["selected_epoch"],
        "selection": "Generated validation WER; pretrained baseline wins ties; test excluded", "same_backbone_comparison": True})
    comparisons = {}
    for name in ("intent_external", "intent_unseen_languages"):
        pvalue = json.loads((out / name / "metrics.json").read_text())["primary_selected_minus_previous"]["exact_mcnemar_two_sided_p"]
        comparisons[name] = {"raw_p": pvalue, "bonferroni_adjusted_p": min(1., 2 * pvalue), "significant_at_family_alpha_0_05": 2 * pvalue < .05}
    dump_json(out / "external_comparison_family.json", {"comparisons": comparisons, "primary_comparisons": 2,
        "intervals": "Nominal descriptive 95% intervals; no claim of simultaneous coverage",
        "scope": "English varieties and unseen languages are separate populations; do not pool or reinterpret the English result"})
    dump_json(out / "SHA256SUMS.json", {str(path.relative_to(out)): fingerprint(path) for path in sorted(out.rglob("*")) if path.is_file()})
    print(f"Exported bundle to {out}")


if __name__ == "__main__": main()
