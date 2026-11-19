"""Transcribe a local recording using an exported, revision-pinned LoRA adapter."""
import argparse
import json

import torch
from peft import PeftConfig, PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .common import RATE, audio_array, seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio")
    p.add_argument("--adapter", required=True)
    p.add_argument("--revision", default=None, help="Required if the adapter does not record its base revision")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    seed_everything()
    config = PeftConfig.from_pretrained(a.adapter)
    revision = a.revision or config.revision
    if revision is None:
        p.error("Supply the base-model revision from run_config.json with --revision")
    audio = audio_array(a.audio)
    if not 0 < len(audio) <= 30 * RATE:
        p.error("Supply a recording between 0 and 30 seconds")
    processor = WhisperProcessor.from_pretrained(config.base_model_name_or_path, revision=revision,
                                                 language="english", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(config.base_model_name_or_path, revision=revision,
                                                           use_safetensors=True)
    model = PeftModel.from_pretrained(base, a.adapter).to(a.device).eval()
    model.generation_config.forced_decoder_ids = None
    features = processor.feature_extractor(audio, sampling_rate=RATE, return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
        ids = model.generate(input_features=features.input_features.to(a.device),
                             attention_mask=features.attention_mask.to(a.device),
                             language="english", task="transcribe", max_new_tokens=256,
                             do_sample=False, num_beams=1)
    print(json.dumps({"transcript": processor.batch_decode(ids, skip_special_tokens=True)[0],
                      "base_model": config.base_model_name_or_path, "revision": revision}))


if __name__ == "__main__":
    main()
