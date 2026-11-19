"""Conservative normalized-target ASR adaptation, selected on generated dev WER."""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from .asr import attach_lora, cached_batch, predict, tokenize_target
from .common import RATE, add_noise, audio_array, dump_json, environment, fingerprint, read_manifest, seed_everything, split_summary
from .intent_features import AUDIO_MODEL, AUDIO_REVISION
from .metrics import asr_summary, paired_wer_bootstrap


def select_checkpoint(history):
    """The zero-step pretrained baseline participates and wins exact ties."""
    return min(history, key=lambda r: (r["validation_wer"], r["epoch"]))


def cache_training(base, processor, rows, directory, device):
    directory.mkdir(parents=True, exist_ok=True)
    normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
    paths = []
    base.eval()
    for i, row in enumerate(rows):
        signal = audio_array(row["absolute_path"])
        # Half clean, half modest noise augmentation, fixed before selection.
        if i % 2:
            signal = add_noise(signal, 15.0, 19 + i)
        inputs = processor.feature_extractor(signal, sampling_rate=RATE, return_tensors="pt")
        with torch.no_grad():
            hidden = base.get_encoder()(inputs.input_features.to(device)).last_hidden_state[0].cpu()
        labels = tokenize_target(processor, normalizer(row["text"]), base.config.decoder_start_token_id,
                                 base.config.max_target_positions)
        path = directory / f"item_{i:05d}.pt"
        torch.save({"hidden": hidden, "labels": labels}, path)
        paths.append(path)
        if (i + 1) % 48 == 0:
            print(f"Training cache {i + 1}/{len(rows)}", flush=True)
    return paths


def compact_predictions(rows):
    """Expose sufficient error counts for audit without republishing source text."""
    speakers = {s: f"speaker_{i:03d}" for i, s in enumerate(sorted({r["speaker_id"] for r in rows}))}
    return [{"id": f"example_{i:05d}", "speaker_id": speakers[r["speaker_id"]],
             **{k: r[k] for k in ("word_errors", "reference_words", "character_errors", "reference_characters", "duration_seconds")}}
            for i, r in enumerate(rows)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/librispeech-v2/manifest.jsonl")
    p.add_argument("--output", default="runs/asr-v2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.learning_rate <= 0: p.error("Invalid configuration")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()): p.error("Output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(19, args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_manifest(args.manifest)
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    run = {"config": vars(args), "seed": 19, "model": AUDIO_MODEL, "revision": AUDIO_REVISION,
           "manifest_sha256": fingerprint(args.manifest), "splits": split_summary(rows), "environment": environment(),
           "selection": "Lowest generated clean validation WER including unchanged baseline; earliest epoch wins ties",
           "training_targets": "Whisper EnglishTextNormalizer applied to uppercase source transcripts",
           "augmentation": "15 dB Gaussian noise for alternating training recordings; no augmentation for other half",
           "status": "running"}
    dump_json(out / "run_config.json", run)
    processor = WhisperProcessor.from_pretrained(AUDIO_MODEL, revision=AUDIO_REVISION, language="english", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(AUDIO_MODEL, revision=AUDIO_REVISION, attn_implementation="sdpa").to(device)
    base.generation_config.forced_decoder_ids = None; base.config.forced_decoder_ids = None
    cache = cache_training(base, processor, split["train"], out / "encoder_cache", device)
    model = attach_lora(base)
    model.peft_config["default"].revision = AUDIO_REVISION
    initial_state = copy.deepcopy(get_peft_model_state_dict(model))
    baseline_dev = predict(model, processor, split["validation"], device)
    history = [{"epoch": 0, "optimizer_steps": 0, "validation_wer": asr_summary(baseline_dev)["wer"]}]
    dump_json(out / "development_epoch_0.json", compact_predictions(baseline_dev))
    best_state, best_epoch, step = initial_state, 0, 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=.01)
    total_steps = args.epochs * ((len(cache) + args.batch_size - 1) // args.batch_size)
    warmup = max(1, int(.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, max(0., (total_steps - s) / max(1, total_steps - warmup))))
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(cache))); random.Random(19 + epoch).shuffle(order)
        loss_sum, count = 0., 0
        for start in range(0, len(order), args.batch_size):
            batch = cached_batch([cache[i] for i in order[start:start + args.batch_size]], device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step(); scheduler.step(); step += 1
            n = int((batch["labels"] != -100).sum()); loss_sum += loss.item() * n; count += n
            if step % 48 == 0: print(f"Epoch {epoch}, step {step}/{total_steps}", flush=True)
        dev = predict(model, processor, split["validation"], device)
        history.append({"epoch": epoch, "optimizer_steps": step, "train_token_loss": loss_sum / count,
                        "validation_wer": asr_summary(dev)["wer"], "elapsed_seconds": time.perf_counter() - started})
        dump_json(out / f"development_epoch_{epoch}.json", compact_predictions(dev))
        if select_checkpoint(history)["epoch"] == epoch:
            best_state, best_epoch = copy.deepcopy(get_peft_model_state_dict(model)), epoch
        dump_json(out / "training_history.json", history)
        print(history[-1], flush=True)
    set_peft_model_state_dict(model, best_state)
    model.save_pretrained(out / "adapter", safe_serialization=True)
    # Force reconstruction of exported adapter before scoring the untouched test.
    base = model.unload(); model = PeftModel.from_pretrained(base, out / "adapter").to(device)
    dump_json(out / "selection.json", {"epoch": best_epoch, "validation_wer": select_checkpoint(history)["validation_wer"],
                                       "baseline_selected": best_epoch == 0, "test_used": False})
    results = {}
    for name, snr in (("clean", None), ("gaussian_noise_10db", 10.)):
        with model.disable_adapter(): original = predict(model, processor, split["test"], device, noise_snr=snr, seed=19)
        selected = predict(model, processor, split["test"], device, noise_snr=snr, seed=19)
        original, selected = compact_predictions(original), compact_predictions(selected)
        dump_json(out / f"predictions_{name}_base.json", original)
        dump_json(out / f"predictions_{name}_selected.json", selected)
        results[name] = {"base": asr_summary(original), "selected": asr_summary(selected),
                         "paired_comparison": paired_wer_bootstrap(original, selected, 19)}
    dump_json(out / "metrics.json", {"model": AUDIO_MODEL, "revision": AUDIO_REVISION, "selected_epoch": best_epoch,
        "trainable_parameters": sum(p.numel() for p in best_state.values()), "optimizer_steps": step,
        "baseline_validation_wer": history[0]["validation_wer"], "selected_validation_wer": select_checkpoint(history)["validation_wer"],
        "results": results, "limitations": ["Small English audiobook experiment, not a conversational speech LLM.",
        "Fresh adaptation holdout speakers may have occurred in foundation pretraining.",
        "Synthetic Gaussian noise does not establish real-world noise robustness.", "One training seed; no causal attribution to one individual change."]})
    run["status"] = "completed"; dump_json(out / "run_config.json", run)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__": main()
