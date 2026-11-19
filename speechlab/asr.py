"""Fine-tune decoder LoRA on real speech; evaluate a locked test set once."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from .common import (RATE, add_noise, audio_array, dump_json, environment, fingerprint,
                     read_manifest, seed_everything, split_summary)
from .metrics import asr_summary, paired_wer_bootstrap, transcript_counts


def attach_lora(base, rank=8):
    # Generic PeftModel preserves Whisper's input_features interface. A text-oriented
    # SEQ_2_SEQ_LM wrapper can incorrectly forward input_ids to Whisper.
    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=.05, bias="none",
                        target_modules=r"model\.decoder\..*\.(q_proj|v_proj)")
    model = get_peft_model(base, config)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("decoder" not in n or "lora_" not in n for n in trainable):
        raise RuntimeError("Only decoder LoRA parameters may be trained")
    return model


def tokenize_target(processor, text, decoder_start_token_id, maximum=448):
    labels = processor.tokenizer(text).input_ids
    if labels and labels[0] == decoder_start_token_id:
        labels = labels[1:]
    if len(labels) > maximum:
        raise ValueError("Transcript exceeds decoder context; refusing silent truncation")
    return labels


def collate_cached(items, device):
    hidden = torch.stack([i["hidden"] for i in items]).to(device)
    labels = torch.full((len(items), max(len(i["labels"]) for i in items)), -100, dtype=torch.long)
    for j, item in enumerate(items):
        labels[j, :len(item["labels"])] = torch.tensor(item["labels"])
    return {"encoder_outputs": BaseModelOutput(last_hidden_state=hidden), "labels": labels.to(device)}


def cache_encoder(base, processor, rows, directory, device):
    directory.mkdir(parents=True, exist_ok=True)
    base.eval()
    paths = []
    for i, row in enumerate(rows):
        waveform = audio_array(row["absolute_path"])
        if len(waveform) > 30 * RATE:
            raise ValueError("Audio exceeds Whisper's 30-second window")
        inputs = processor.feature_extractor(waveform, sampling_rate=RATE, return_tensors="pt")
        with torch.no_grad():
            hidden = base.get_encoder()(inputs.input_features.to(device)).last_hidden_state[0].cpu()
        item = {"hidden": hidden, "labels": tokenize_target(processor, row["text"], base.config.decoder_start_token_id,
                                                            base.config.max_target_positions)}
        path = directory / f"{row['id']}.pt"
        torch.save(item, path)
        paths.append(path)
        if (i + 1) % 32 == 0 or i == len(rows) - 1:
            print(f"Cached frozen encoder: {i+1}/{len(rows)}", flush=True)
    return paths


def cached_batch(paths, device):
    return collate_cached([torch.load(p, weights_only=True, map_location="cpu") for p in paths], device)


def validation_loss(model, paths, batch_size, device):
    model.eval()
    weighted, tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = cached_batch(paths[i:i + batch_size], device)
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss")
            count = int((batch["labels"] != -100).sum())
            weighted += float(loss) * count
            tokens += count
    return weighted / tokens


def predict(model, processor, rows, device, max_new_tokens=128, noise_snr=None, seed=13):
    normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)
    model.eval()
    predictions = []
    for i, row in enumerate(rows):
        signal = audio_array(row["absolute_path"])
        if noise_snr is not None:
            signal = add_noise(signal, noise_snr, seed + i)
        inputs = processor.feature_extractor(signal, sampling_rate=RATE, return_tensors="pt", return_attention_mask=True)
        started = time.perf_counter()
        with torch.no_grad():
            tokens = model.generate(input_features=inputs.input_features.to(device),
                                    attention_mask=inputs.attention_mask.to(device),
                                    language="english", task="transcribe", do_sample=False,
                                    num_beams=1, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - started
        hypothesis = processor.batch_decode(tokens, skip_special_tokens=True)[0]
        reference, normalized = normalizer(row["text"]), normalizer(hypothesis)
        predictions.append({"id": row["id"], "speaker_id": row["speaker_id"],
                            "reference": row["text"], "prediction": hypothesis,
                            "normalized_reference": reference, "normalized_prediction": normalized,
                            "generation_seconds": elapsed, "duration_seconds": row["duration_seconds"],
                            # Whisper can strip EOS in postprocessing; do not infer
                            # generation truncation from the returned final token.
                            "generated_length": tokens.shape[-1],
                            **transcript_counts(reference, normalized)})
        if (i + 1) % 12 == 0 or i == len(rows) - 1:
            print(f"Transcribed {i+1}/{len(rows)}; noise SNR={noise_snr}", flush=True)
    return predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/librispeech-pilot/manifest.jsonl")
    p.add_argument("--output", default="runs/whisper-pilot")
    p.add_argument("--model", default="openai/whisper-tiny")
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--noise-snr", type=float, default=None)
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.rank, a.threads, a.max_new_tokens) < 1 or a.learning_rate <= 0:
        p.error("Training settings must be positive")
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        p.error("Output directory must be empty; use a new name to preserve earlier evidence")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(a.seed, a.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    rows = read_manifest(a.manifest)
    if any(r.get("language") != "en" for r in rows):
        raise ValueError("This experiment uses English prompts and normalization; supply English recordings")
    split = {s: [r for r in rows if r["split"] == s] for s in ["train", "validation", "test"]}
    revision = HfApi().model_info(a.model, revision=a.revision).sha
    run_config = {**vars(a), "resolved_revision": revision, "device": device,
                  "manifest_sha256": fingerprint(a.manifest), "splits": split_summary(rows),
                  "environment": environment(), "normalization": "Whisper EnglishTextNormalizer",
                  "selection": "lowest validation token loss among trained epochs; test never used for selection",
                  "status": "running"}
    dump_json(out / "run_config.json", run_config)
    processor = WhisperProcessor.from_pretrained(a.model, revision=revision, language="english", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(a.model, revision=revision, use_safetensors=True,
                                                           attn_implementation="sdpa").to(device)
    base.generation_config.forced_decoder_ids = None
    base.config.forced_decoder_ids = None
    cache = {s: cache_encoder(base, processor, split[s], out / "encoder_cache" / s, device)
             for s in ["train", "validation"]}
    model = attach_lora(base, a.rank)
    model.peft_config["default"].revision = revision
    params = {"base_parameters": sum(p.numel() for n, p in model.named_parameters() if "lora_" not in n),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    print(params, flush=True)
    baseline_loss = validation_loss(model, cache["validation"], a.batch_size, device)
    history, best_loss, best_epoch = [], float("inf"), None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.learning_rate)
    step = 0
    started = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train()
        indices = list(range(len(cache["train"])))
        random.Random(a.seed + epoch).shuffle(indices)
        epoch_loss, epoch_tokens = 0.0, 0
        for i in range(0, len(indices), a.batch_size):
            paths = [cache["train"][j] for j in indices[i:i + a.batch_size]]
            batch = cached_batch(paths, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            count = int((batch["labels"] != -100).sum())
            epoch_loss += float(loss.detach()) * count
            epoch_tokens += count
            step += 1
            if step % 16 == 0:
                print(f"Epoch {epoch}, step {step}, token loss {epoch_loss/epoch_tokens:.4f}", flush=True)
        val = validation_loss(model, cache["validation"], a.batch_size, device)
        history.append({"epoch": epoch, "optimizer_steps": step, "train_token_loss": epoch_loss / epoch_tokens,
                        "validation_token_loss": val, "elapsed_seconds": time.perf_counter() - started})
        print(history[-1], flush=True)
        dump_json(out / "training_history.json", history)
        if val < best_loss:
            best_loss, best_epoch = val, epoch
            model.save_pretrained(out / "adapter", safe_serialization=True)
            processor.save_pretrained(out / "adapter")
    # Reload the selected adapter: validates that exported artifacts are usable.
    frozen_base = model.unload()
    model = PeftModel.from_pretrained(frozen_base, out / "adapter").to(device)
    all_results = {}
    for condition, snr in [("clean", None)] + ([("gaussian_noise", a.noise_snr)] if a.noise_snr is not None else []):
        with model.disable_adapter():
            baseline = predict(model, processor, split["test"], device, a.max_new_tokens, snr, a.seed)
        adapted = predict(model, processor, split["test"], device, a.max_new_tokens, snr, a.seed)
        dump_json(out / f"predictions_{condition}_base.json", baseline)
        dump_json(out / f"predictions_{condition}_lora.json", adapted)
        all_results[condition] = {"base": asr_summary(baseline), "lora": asr_summary(adapted),
                                  "paired_comparison": paired_wer_bootstrap(baseline, adapted, a.seed), "noise_snr_db": snr}
    report = {"status": "completed", "scope": "small English LibriSpeech pilot, not a full-corpus benchmark",
              "base_model": a.model, "base_model_revision": revision, "parameters": params,
              "best_epoch": best_epoch, "baseline_validation_loss": baseline_loss,
              "best_validation_loss": best_loss, "optimizer_steps": step, "results": all_results,
              "limitations": ["No speech-LLM conversation, multilingual, or assistant-directed speech evaluation.",
                              "Foundation pretraining overlap is unknown; speakers are held out only from this adaptation.",
                              "Noise stress is additive Gaussian noise, not recorded environmental noise.",
                              "Bootstrap intervals reflect a small sampled set of speakers."]}
    dump_json(out / "metrics.json", report)
    run_config["status"] = "completed"
    dump_json(out / "run_config.json", run_config)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
