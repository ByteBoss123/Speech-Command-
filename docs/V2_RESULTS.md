# What improved, and how to interpret it

The earlier results were narrow pilots: ASR adaptation regressed, speaker trials
shared chapters, and intent used 126 test examples from one seed. V2 improves the
experiments and measures changes on fresh holdouts. Old scores are retained.

## ASR

Training grew from 128 recordings / 16 speakers / 25.1 minutes to **384 / 48 /
84.2 minutes**. Learning rate fell from 0.0002 to **0.00005**, with warmup/decay.
Uppercase targets are normalized and alternating examples receive 15 dB Gaussian
noise. Changes were applied together; the gain cannot be assigned to one change.

Generated validation WER selects among the unchanged model and three epochs, with
the earlier model winning ties. There were 288 total updates; epoch 2, after 192
updates, won. Validation WER decreased from 7.59% to 7.51%. This replaces the old
selection by token loss, which had not tracked generated-transcript accuracy.

A pretrained Whisper-base candidate was added after development showed the first
adapter epoch had not beaten baseline. That candidate and its selection rule were
recorded before ASR test scoring. Its **4.80% validation WER** won; the model choice
was recorded while no test prediction export existed.

New test: 120 recordings / 20 speakers / 2,554 reference words. New validation/test
speakers exclude all earlier evaluation speakers.

| Model | Clean WER | 10 dB noise WER |
|---|---:|---:|
| Original pretrained tiny | 6.42% | 14.96% |
| Selected tiny LoRA | 5.91% | 14.80% |
| Selected capacity candidate: pretrained base | 4.66% | 9.51% |

LoRA gives a **7.9% relative reduction in clean word errors**. Its absolute difference
is −0.51 points, speaker-bootstrap CI [−1.02, −0.05]. The noisy difference is −0.16
points [−1.19, +0.75], so noise improvement from LoRA is not established. Selecting
Whisper-base reduces word errors **27.4% clean** and **36.4% under synthetic noise**;
its intervals exclude zero in both conditions.

The largest gain comes from a larger **pretrained** model. The tiny adapter remains
a real trained, saved and evaluated artifact. No fair latency/device benchmark was run.

## Speaker verification

V1 used same-chapter positives and a 5% calibration FAR target. V2 uses three
recordings for enrollment and three probes from a different chapter, at a stricter
**1% empirical validation FAR target**. Twenty fresh speakers yield 1,200 trials:
60 positive and 1,140 negative.

Methods fixed before testing: one raw enrollment embedding, a three-recording mean,
and training-cohort centering plus that mean. Centering uses 96 recordings from
48 training speakers; the encoder stays frozen. The uncentered mean wins on validation.

| Method | False accepts | False rejects | Descriptive EER |
|---|---:|---:|---:|
| Single raw enrollment | 8/1,140 = 0.70% | 22/60 = 36.67% | 5.18% |
| Selected three-recording mean | 11/1,140 = 0.96% | 6/60 = 10.00% | 3.03% |
| Centered mean, unselected | 0/1,140 = 0% | 17/60 = 28.33% | 4.96% |

Selected false rejects fall **72.7% relative**, while observed FAR rises 0.26 points.
The paired false-reject difference has a probe-speaker bootstrap interval
[−41.67, −13.33] points, conditional on the enrolled cohort. Shared enrollment
recordings introduce dependence that this interval does not remove.

These are harder trials, so a higher final FRR than the old easy pilot is not a
regression on the same task. Chapter separation does not establish different sessions,
microphones or foundation-pretraining speakers. There is no spoofing evaluation.

## Multilingual intent

V2: **1,135 train / 168 validation / 210 test**. Every test example excludes all
588 previously used records, normalized transcripts and source-audio hashes. The
earlier test is retired and may enter training. All three languages occur in training.

The encoders stay frozen. Audio, text and joint heads each use seeds 13, 29 and 47.
Checkpoints use validation macro F1, then validation loss. A fixed probability-blend
grid [0, .1, .25, .5, .75, 1] is evaluated on validation; .25 joint / .75 text wins.

| Model | Fresh-test accuracy | Macro F1 |
|---|---:|---:|
| Unchanged v1 fusion | 95.71% | 0.9571 |
| v2 audio ensemble | 83.33% | 0.8318 |
| v2 text ensemble | 97.14% | 0.9712 |
| v2 joint ensemble | 98.10% | 0.9809 |
| Selected blend | 97.62% | 0.9760 |

Selected blend by language: English 95.71%, Spanish 100%, French 97.14%, with only
70 examples per language. Individual joint seed accuracies are 97.62%, 98.10%,
98.10%. The final choice does not switch to the higher-scoring pure joint ensemble
after inspecting test results.

Gain over unchanged v1 is +1.90 points, descriptive paired CI [−0.48, +4.29]. It is
**not yet a reliable population-level gain**. The joint signal adds little over
text only. Training size and seed ensembling changed together; gains cannot be
attributed to either alone. Supplied transcripts are required, and missing speaker
IDs remain a material limitation. This is not generative multimodal LLM training.

## Verification

The README and v2 notebook contain commands. Public artifacts retain weights,
histories, validation choices, probabilities, error counts and anonymous speaker
groups. Full source manifests/audio and individual speaker embeddings remain local.
Pinned data builders regenerate them.

`python -m speechlab.verify_v2` recomputes scorecards, validation selections and
bootstrap comparisons. Nine intent heads were reloaded against the local cache;
per-seed/ensemble scores matched, and normalization matched training-only statistics.
Raw-input intent/speaker inference and selected ASR inference were checked. Notebook
schema and code cells were validated; equivalent CLI runs were executed.

The 28 offline tests also cover leakage rejection, baseline retention, capacity
ties, cross-chapter sampling, enrollment averaging, fusion rejection and exclusion
of source content from the public evaluation index. Earlier results are not overwritten.
