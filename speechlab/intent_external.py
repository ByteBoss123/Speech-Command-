"""Replicate frozen intent models on previously unused English varieties."""
from __future__ import annotations
import argparse
import gc
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import binomtest
from .common import RATE, audio_array, dump_json, environment, fingerprint, seed_everything
from .intent import classification_metrics, load_head, paired_bootstrap
from .intent_data import DATASET, text_key
from .intent_data_v2 import REVISION
from .intent_features import (AUDIO_REVISION, TEXT_REVISION, encode_audio, encode_text,
                              load_audio_encoder, load_text_encoder)
from .intent_v2 import ensemble


def exclude_used(rows, previous):
    ids = {r["id"] for r in previous}
    texts = {text_key(r["text"]) for r in previous}
    audio = {r["source_audio_sha256"] for r in previous}
    accepted = []
    for row in sorted(rows, key=lambda r: r["id"]):
        key = text_key(row["text"])
        if row["id"] in ids or key in texts or row["source_audio_sha256"] in audio: continue
        accepted.append(row)
        ids.add(row["id"]); texts.add(key); audio.add(row["source_audio_sha256"])
    return accepted


def comparison(truth, selected, baseline, languages):
    truth, selected, baseline = map(np.asarray, (truth, selected, baseline))
    if truth.shape != selected.shape or truth.shape != baseline.shape or not truth.size:
        raise ValueError("Paired predictions required")
    improved = int(((selected == truth) & (baseline != truth)).sum())
    regressed = int(((selected != truth) & (baseline == truth)).sum())
    result = paired_bootstrap(truth, selected, baseline, np.asarray(languages), samples=10000, seed=31)
    result.update(improved=improved, regressed=regressed,
                  exact_mcnemar_two_sided_p=float(binomtest(improved, improved + regressed, .5).pvalue) if improved + regressed else 1.)
    counts = np.array([[((selected[truth == label] == label).astype(int) - (baseline[truth == label] == label)).sum(),
                        (truth == label).sum()] for label in np.unique(truth)])
    totals = counts[np.random.default_rng(31).integers(len(counts), size=(10000, len(counts)))].sum(1)
    result["intent_cluster_accuracy_difference_ci95"] = np.quantile(totals[:, 0] / totals[:, 1], [.025, .975]).tolist()
    result["dependence_limit"] = "Both intervals omit speaker dependence because speaker identities are unavailable. Intent clustering is a sensitivity analysis, not a speaker-level CI."
    return result


def prepare(directory):
    previous_paths = [Path("data/minds14-pilot/manifest.jsonl"), Path("data/minds14-v2/manifest.jsonl")]
    previous = [json.loads(line) for path in previous_paths for line in path.read_text().splitlines()]
    rows, payloads, source_counts = [], {}, Counter()
    for lang in ("en-AU", "en-GB"):
        path = hf_hub_download(DATASET, f"{lang}/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
        with pq.ParquetFile(path) as f:
            for batch in f.iter_batches(batch_size=32, use_threads=False):
                for record in batch.to_pylist():
                    source_counts[lang] += 1
                    payload = record["audio"]["bytes"]; signal = audio_array(io.BytesIO(payload))
                    if not 1 <= len(signal) / RATE <= 30 or not text_key(record["transcription"]): continue
                    sid = hashlib.sha256((lang + ":" + record["path"]).encode()).hexdigest()[:24]
                    rows.append({"id": sid, "language": lang, "label": record["intent_class"], "text": record["transcription"],
                                 "source_audio_sha256": hashlib.sha256(payload).hexdigest(), "duration_seconds": len(signal) / RATE})
                    payloads[sid] = payload
    rows = exclude_used(rows, previous)
    for row in rows:
        relative = Path("audio") / f"{row['id']}.wav"; path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio_array(io.BytesIO(payloads[row["id"]])), RATE, subtype="PCM_16")
        row.update(path=relative.as_posix(), audio_sha256=fingerprint(path), split="external_test")
    manifest = directory / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    card = {"dataset": DATASET, "revision": REVISION, "license": "CC-BY-4.0", "source_counts": dict(source_counts),
            "retained": dict(Counter(r["language"] for r in rows)), "total": len(rows), "manifest_sha256": fingerprint(manifest),
            "previous_manifest_sha256": [fingerprint(p) for p in previous_paths],
            "exclusions": "All prior IDs, normalized transcripts, source audio hashes; duplicate external recordings; duration outside 1–30 seconds",
            "training_or_selection_on_external_data": False,
            "limitations": ["New English varieties only; does not validate French or Spanish transfer.",
                "Speaker identities unavailable; recording independence unverified.", "Source transcripts supplied, not ASR outputs."]}
    dump_json(directory / "dataset_card.json", card)
    return rows, card


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/minds14-external-v3")
    p.add_argument("--output", default="runs/intent-external-v3")
    p.add_argument("--model", default="evidence/2026-11-19-v2/intent")
    p.add_argument("--baseline", default="evidence/2026-11-19/intent/fusion")
    a = p.parse_args(); out, data, root = Path(a.output), Path(a.data), Path(a.model)
    if any(path.exists() and any(path.iterdir()) for path in (out, data)): p.error("Outputs must be empty")
    out.mkdir(parents=True); data.mkdir(parents=True); seed_everything(31, 2)
    config = json.loads((root / "ensemble_config.json").read_text())
    if config["alpha"] != .25: raise ValueError("Expected frozen alpha 0.25")
    if config["feature_metadata"]["audio_revision"] != AUDIO_REVISION or config["feature_metadata"]["text_revision"] != TEXT_REVISION:
        raise ValueError("Encoder mismatch")
    paths = [Path(a.baseline)] + [root / path for path in config["text_models"] + config["fusion_models"]]
    lock = {"model_weight_sha256": {str(path): fingerprint(path / "head.safetensors") for path in paths},
            "config_sha256": fingerprint(root / "ensemble_config.json"), "model_directory": str(root), "alpha": .25,
            "protocol_sha256": fingerprint("experiments/v3_protocol.json"), "external_predictions_computed": False}
    dump_json(out / "frozen_models.json", lock)
    rows, card = prepare(data); dump_json(out / "dataset_card.json", card)
    print(f"External holdout: {len(rows)} recordings, {card['retained']}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor, encoder = load_audio_encoder(device)
    audio_batches = []
    for i in range(0, len(rows), 4):
        audio_batches.append(encode_audio([data / r["path"] for r in rows[i:i+4]], extractor, encoder, device))
        if i % 100 == 0: print(f"Audio features {i}/{len(rows)}", flush=True)
    audio = np.concatenate(audio_batches)
    del encoder, extractor, audio_batches; gc.collect()
    tokenizer, encoder = load_text_encoder(device)
    text = np.concatenate([encode_text([r["text"] for r in rows[i:i+32]], tokenizer, encoder, device) for i in range(0, len(rows), 32)])
    del tokenizer, encoder; gc.collect()
    np.savez_compressed(out / "local_features.npz", audio=audio, text=text)
    audio, text = torch.from_numpy(audio), torch.from_numpy(text)
    values = {}
    for name, key in (("text", "text_models"), ("fusion", "fusion_models")):
        values[name] = ensemble([load_head(root / path)[0] for path in config[key]], audio, text)
    values["selected_blend"] = .75 * values["text"] + .25 * values["fusion"]
    values["previous_v1_fusion"] = ensemble([load_head(a.baseline)[0]], audio, text)
    predicted = {key: value.argmax(1) for key, value in values.items()}
    truth, languages = np.array([r["label"] for r in rows]), np.array([r["language"] for r in rows])
    scores = {name: {"test": classification_metrics(truth, labels),
                    "by_language": {lang: classification_metrics(truth[languages == lang], labels[languages == lang]) for lang in sorted(set(languages))}}
              for name, labels in predicted.items()}
    report = {"models": scores, "primary_selected_minus_previous": comparison(truth, predicted["selected_blend"], predicted["previous_v1_fusion"], languages),
              "secondary_selected_minus_text": comparison(truth, predicted["selected_blend"], predicted["text"], languages),
              "model_selection": "None; all models and blend fixed before external data preparation", "environment": environment(), "limitations": card["limitations"]}
    dump_json(out / "metrics.json", report)
    dump_json(out / "predictions.json", [{"id": f"example_{i:05d}", "language": r["language"], "label": r["label"],
                                         "predicted": {key: int(value[i]) for key, value in predicted.items()}} for i, r in enumerate(rows)])
    for path, checksum in lock["model_weight_sha256"].items():
        if fingerprint(Path(path) / "head.safetensors") != checksum: raise ValueError("Model changed during evaluation")
    print(json.dumps({"accuracy": {k: v['test']['accuracy'] for k, v in scores.items()}, "primary": report["primary_selected_minus_previous"]}, indent=2), flush=True)


if __name__ == "__main__": main()
