# Siri Speech portfolio audit — November 19, 2026

The connected `ByteBoss123` account exposed 64 repositories. This audit selected the
most relevant speech, generative-model, and multimodal projects for file-level review.
It is not a correctness audit of every repository or a claim to have reproduced earlier runs.

| Repository | Evidence inspected | What it supports | What remains uncertain |
|---|---|---|---|
| [Legal-Case-Holding-Prediction-LoRA](https://github.com/ByteBoss123/Legal-Case-Holding-Prediction-LoRA) | README and `legal_holding_lora.ipynb` | Actual Qwen2.5-1.5B LoRA implementation: answer-token loss, 5,000 training examples, zero-shot and Legal-BERT comparisons | README reports 77.4% accuracy on 3,600 test items; committed notebook has empty outputs, and raw GPU results/adapter were not present in the inspected default-branch tree. Treat the score as repository-reported until the original results archive is checked. |
| [Medical-Record-Entity-Extraction-LoRA](https://github.com/ByteBoss123/Medical-Record-Entity-Extraction-LoRA) | README and `medrec_extraction_lora.ipynb` | Qwen2.5-1.5B LoRA implementation, answer-only loss, document splits, BioBERT/CRF comparisons, OCR stress tests | README reports 0.642 strict micro-F1; committed notebook outputs are empty and raw GPU metrics/adapter were not in the inspected default-branch tree. The CRF Kubernetes service is not an LLM serving benchmark. |
| [Speech-Command-](https://github.com/ByteBoss123/Speech-Command-) — original experiment | `train.py`, `evaluate.py`, `unseen_voice_eval.json`, README, file tree | PyTorch CNN training on synthetic isolated words; result file records 240/252 correct on two held-out synthesis voices | This is neither real-speech ASR nor foundation-model fine-tuning. Legacy files refer to `results/` while uploaded artifacts are at repository root. |
| [Multimodal-Creative-Asset-Search-Ranking-Engine](https://github.com/ByteBoss123/Multimodal-Creative-Asset-Search-Ranking-Engine) | README, `embedder.py`, file tree | Implemented LSA text backend and a CLIP text-embedding backend | Inspected embedder exposes no image-embedding method or model-training loop. README CLIP/multimodal claims and metrics alone do not establish multimodal model training. |
| [VLMHallucinationAudit](https://github.com/ByteBoss123/VLMHallucinationAudit) | README and file tree | Explicitly scoped text-only hallucination classification on VLM-generated reports | It does not jointly process image pixels and text or tune a VLM. |
| [Gen-AI-QA-System](https://github.com/ByteBoss123/Gen-AI-QA-System) | File tree | Retrieval pipeline, API, tests, and saved results are present | This limited inspection does not establish generative-model training. |

## Resulting implementation choice

The earlier resume-only assessment understated the available generative-model training
code. Creating another generic text LoRA notebook would add little. This branch adds:

1. **Real-speech generative ASR adaptation:** decoder LoRA on pretrained Whisper,
   speaker-disjoint LibriSpeech splits, baseline comparison, WER/CER, speaker-cluster
   bootstrap, and optional Gaussian-noise stress evaluation.
2. **Speaker verification evaluation:** a frozen WavLM X-vector model, same/different
   speaker trials, development-only threshold calibration, held-out FAR/FRR, and EER.

These are portfolio experiments. They do not demonstrate large-scale foundation-model
training, multilingual conversational LLMs, reinforcement learning, or assistant-directed
speech detection. LibriSpeech does not provide the labels needed to evaluate whether an
utterance addresses an assistant; inventing those labels would make that claim misleading.

## Recommended application project selection

Use one Qwen LoRA project, this real-speech adaptation experiment, and the speaker
verification experiment. Keep the two speech components under one repository link if
space is limited. Retain exact scale and dataset limitations in the interview explanation.
Recover the original Qwen GPU result archive before using its reported numerical gains
as independently verified evidence. Read and rerun the code before presenting it as work
you can explain in an interview; this extension was implemented with AI assistance.
