# V3 evaluation — rerun in progress

The interrupted November 19 workspace did not retain its unfinished runs. The
November 20 rerun keeps the same development/test separation and model choices.
ASR and speaker test metrics are pending. V2 evidence remains unchanged.

The frozen English-variety replication completed on 1,116 en-AU/en-GB recordings:
v1 scored 1,068 correct (95.70%); the unchanged selected v2 blend scored 1,078
(96.59%). Twenty errors were corrected and ten new errors introduced. Exact
two-sided McNemar p=0.098737; the gain remains uncertain. The nominal utterance
bootstrap interval is [0.00, 1.88] percentage points; the intent-cluster sensitivity
interval is [-0.37, 2.44] points. Neither interval models unknown speaker dependence.

An explicitly recorded extension evaluates all nine remaining language configs,
with every model still frozen. Both external comparisons will be reported
separately, using Bonferroni-adjusted primary p-values. No language is selected
based on its score, and the English result is not reinterpreted as confirmed.
See [`v3_multilingual_extension_protocol.json`](../experiments/v3_multilingual_extension_protocol.json).

Speaker development selected five-recording averaging at 0.474% FAR and 13% FRR,
versus the frozen v2 three-recording configuration's 1.947% FAR and 10% FRR.
Before test scores were inspected, a training-only covariance-normalization
extension was specified to try to reduce that rejection cost. It uses eight
recordings per training speaker and three fixed shrinkage candidates. The original
raw/centered methods remain candidates, and the final choice uses development
FRR under the same FAR cap. This is a regularized WCCN-inspired cosine backend,
not a reproduction of the original SVM/PCA paper.
See [`v3_speaker_extension_protocol.json`](../experiments/v3_speaker_extension_protocol.json)
and [the WCCN paper](https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.html).

- Train Whisper-base decoder LoRA on 768 recordings from 48 speakers. Select an
  epoch by generated validation WER, including the unchanged pretrained baseline.
  Evaluate on 150 probes from 30 speakers absent from every prior manifest.
- Compare five-recording WavLM enrollment, selected at a 0.5% development FAR cap,
  with frozen v2 one/three-recording thresholds on the same 4,500 fresh trials.
- Evaluate frozen v1/v2 intent models on all eligible unused en-AU/en-GB recordings,
  excluding earlier IDs, normalized transcripts and audio hashes. Use paired
  bootstrap and exact McNemar statistics; do not select models from this test.

Protocol: [`experiments/v3_protocol.json`](../experiments/v3_protocol.json).
Run the earlier data builders to reproduce exclusions, then:

```bash
python -m speechlab.prepare_v3
python -m speechlab.asr_v3
python -m speechlab.speaker_v3
python -m speechlab.speaker_normalized
python -m speechlab.intent_external
python -m speechlab.intent_external --data data/minds14-unseen-v3 --output runs/intent-unseen-v3 --languages cs-CZ de-DE it-IT ko-KR nl-NL pl-PL pt-PT ru-RU zh-CN --previous data/minds14-pilot/manifest.jsonl data/minds14-v2/manifest.jsonl data/minds14-external-v3/manifest.jsonl --protocol experiments/v3_multilingual_extension_protocol.json
python -m speechlab.bundle_v3
python -m speechlab.verify_v3
```

Outputs must be empty for fresh runs. The bundle exporter uses an explicit
allowlist; source transcripts, recording identifiers, audio and per-record
embeddings remain local. New results require interpretation within their sample:
LibriSpeech source-prefix sampling, synthetic ASR noise, dependent speaker trials,
and unavailable MInDS speaker identities remain limitations.
