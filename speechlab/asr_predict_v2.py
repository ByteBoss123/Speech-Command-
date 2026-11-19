"""Transcribe with the ASR model selected by the v2 development protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .common import RATE, audio_array, seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio")
    p.add_argument("--selection", default="evidence/2026-11-19-v2/asr_selection.json")
    a = p.parse_args(); seed_everything(19, 3)
    source = Path(a.selection); config = json.loads(source.read_text())
    audio = audio_array(a.audio)
    if not 0 < len(audio) <= 30 * RATE: p.error("Audio must be at most 30 seconds")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(config["model"], revision=config["revision"], language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(config["model"], revision=config["revision"], attn_implementation="sdpa")
    if config["uses_adapter"]: model = PeftModel.from_pretrained(model, source.parent / config["adapter_path"])
    model = model.to(device).eval(); model.generation_config.forced_decoder_ids = None
    features = processor.feature_extractor(audio, sampling_rate=RATE, return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
        output = model.generate(input_features=features.input_features.to(device), attention_mask=features.attention_mask.to(device),
                                language="english", task="transcribe", do_sample=False, num_beams=1, max_new_tokens=128)
    print(json.dumps({"model": config["model"], "transcript": processor.batch_decode(output, skip_special_tokens=True)[0]}))


if __name__ == "__main__": main()
