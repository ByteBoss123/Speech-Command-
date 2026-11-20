# V3 evaluation — rerun in progress

The interrupted November 19 workspace did not retain its unfinished runs. The
November 20 rerun keeps the same development/test separation and model choices.
No completed v3 metric is claimed in this checkpoint. V2 evidence remains unchanged.

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
python -m speechlab.intent_external
python -m speechlab.bundle_v3
python -m speechlab.verify_v3
```

Outputs must be empty for fresh runs. The bundle exporter uses an explicit
allowlist; source transcripts, recording identifiers, audio and per-record
embeddings remain local. New results require interpretation within their sample:
LibriSpeech source-prefix sampling, synthetic ASR noise, dependent speaker trials,
and unavailable MInDS speaker identities remain limitations.
