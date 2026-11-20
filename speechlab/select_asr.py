"""Apply the predeclared validation-only model-size selection rule."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .asr_capacity import MODEL, REVISION, choose_model
from .common import dump_json
from .intent_features import AUDIO_MODEL, AUDIO_REVISION
from .metrics import asr_summary, paired_wer_bootstrap


def select(tiny, base, output_parent=None):
    tiny, base = Path(tiny), Path(base)
    scores = {"tiny_selected": json.loads((tiny / "selection.json").read_text())["validation_wer"],
              "base_pretrained": json.loads((base / "selection_input.json").read_text())["validation_wer"]}
    chosen = choose_model(scores)
    # No test score is read until the choice has been fixed by development WER.
    results = {}
    for condition in ("clean", "gaussian_noise_10db"):
        original = json.loads((tiny / f"predictions_{condition}_base.json").read_text())
        adapted = json.loads((tiny / f"predictions_{condition}_selected.json").read_text())
        larger = json.loads((base / f"predictions_{condition}.json").read_text())
        selected = larger if chosen == "base_pretrained" else adapted
        results[condition] = {"original_tiny": asr_summary(original), "adapted_tiny": asr_summary(adapted),
                              "pretrained_base": asr_summary(larger), "selected": asr_summary(selected),
                              "selected_minus_original_tiny": paired_wer_bootstrap(original, selected, 19)}
    return {"chosen": chosen, "development_wer": scores,
            "model": MODEL if chosen == "base_pretrained" else AUDIO_MODEL,
            "revision": REVISION if chosen == "base_pretrained" else AUDIO_REVISION,
            "uses_adapter": chosen == "tiny_selected", "adapter_path": Path(os.path.relpath(tiny.resolve() / "adapter", Path(output_parent or tiny.parent).resolve())).as_posix() if chosen == "tiny_selected" else None,
            "selection_rule": "Lowest generated validation WER; tiny wins exact ties", "test_used_for_selection": False,
            "results": results,
            "interpretation": "A gain from selecting the larger pretrained model is a capacity/model-selection gain, not a LoRA training gain."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiny", default="runs/asr-v2")
    p.add_argument("--base", default="runs/asr-capacity-v2")
    p.add_argument("--output", default="runs/asr-selection-v2.json")
    a = p.parse_args()
    if Path(a.output).exists(): p.error("Output already exists")
    result = select(a.tiny, a.base, Path(a.output).parent); dump_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
