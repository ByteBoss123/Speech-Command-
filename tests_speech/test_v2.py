import numpy as np
import pytest

from speechlab.asr_v2 import compact_predictions, select_checkpoint
from speechlab.asr_capacity import choose_model
from speechlab.intent_data import text_key
from speechlab.intent_data_v2 import split_fresh
from speechlab.intent_v2 import select_fusion
from speechlab.prepare_v2 import choose_chapters
from speechlab.speaker_v2 import clustered_comparison, enrollment_scores


def test_checkpoint_selection_retains_base_when_updates_regress():
    history = [{"epoch": 0, "validation_wer": .10}, {"epoch": 1, "validation_wer": .13}, {"epoch": 2, "validation_wer": .10}]
    assert select_checkpoint(history)["epoch"] == 0
    history.append({"epoch": 3, "validation_wer": .09})
    assert select_checkpoint(history)["epoch"] == 3


def test_capacity_selection_uses_development_scores_and_prefers_small_ties():
    assert choose_model({"tiny_selected": .08, "base_pretrained": .05}) == "base_pretrained"
    assert choose_model({"tiny_selected": .05, "base_pretrained": .05}) == "tiny_selected"


def test_asr_export_keeps_errors_and_clusters_without_source_identifiers():
    row = {"id": "source-audio-identity", "speaker_id": "original-speaker", "reference": "source transcript",
           "prediction": "hypothesis", "word_errors": 2, "reference_words": 10,
           "character_errors": 3, "reference_characters": 40, "duration_seconds": 3.}
    result = compact_predictions([row, row])
    assert result[0]["word_errors"] == 2
    assert result[0]["speaker_id"] == result[1]["speaker_id"]
    assert not any(x in str(result) for x in (row["id"], row["speaker_id"], row["reference"], row["prediction"]))


def test_fresh_holdout_excludes_every_previous_split_and_exact_duplicates():
    rows = [{"id": f"id{i}", "language": "en-US", "label": 0, "text": f"request number {i}",
             "source_audio_sha256": f"audio{i}"} for i in range(12)]
    old = [{**rows[i], "split": split} for i, split in enumerate(("train", "validation", "test"))]
    rows += [{**rows[0], "id": "alias-text", "text": rows[0]["text"].upper() + "!", "source_audio_sha256": "another-audio"},
             {**rows[2], "id": "alias-audio", "text": "a different transcript"}]
    selected = split_fresh(rows, old, test_per_class=3, validation_per_class=1)
    assert selected == split_fresh(list(reversed(rows)), old, test_per_class=3, validation_per_class=1)
    test = [r for r in selected if r["split"] == "test"]
    assert len(test) == 3
    assert all(r["id"] not in {x["id"] for x in old} and text_key(r["text"]) not in {text_key(x["text"]) for x in old}
               and r["source_audio_sha256"] not in {x["source_audio_sha256"] for x in old} for r in test)
    assert next(r for r in selected if r["id"] == "id1")["split"] == "validation"
    assert next(r for r in selected if r["id"] == "id2")["split"] == "train"


def test_cross_chapter_sampling_refuses_single_chapter_speakers():
    groups = {"a": {"one": [1, 2, 3]}, "b": {"one": [1, 2, 3], "two": [4, 5, 6]}}
    assert choose_chapters(groups, 1, 3, 19)[0][0] == "b"
    with pytest.raises(ValueError, match="eligible fresh"):
        choose_chapters(groups, 2, 3, 19)


def speaker_fixture():
    rows, vectors = [], []
    for speaker in ["a", "b"]:
        for role, chapter in [("enroll", "one"), ("probe", "two")]:
            for i in range(3):
                rows.append({"speaker_id": speaker, "chapter_id": speaker + chapter, "verification_role": role})
                vectors.append([0., 1.] if speaker == "a" and role == "enroll" and i == 0 else [1., 0.] if speaker == "a" else [-1., 0.])
    return rows, np.array(vectors)


def test_multi_enrollment_uses_all_clips_and_trials_are_cross_chapter():
    rows, vectors = speaker_fixture()
    single = enrollment_scores(rows, vectors, "single_raw", np.zeros(2))
    multi = enrollment_scores(rows, vectors, "multi_raw", np.zeros(2))
    assert len(multi) == 12 and sum(r["label"] for r in multi) == 6
    assert multi[0]["score"] > single[0]["score"]
    assert all(r["cross_chapter_positive"] for r in multi if r["label"])
    result = clustered_comparison(single, single, .5, .5, samples=50)
    assert result["frr_difference_ci95"] == result["far_difference_ci95"] == [0., 0.]
    rows[3]["chapter_id"] = rows[0]["chapter_id"]
    with pytest.raises(ValueError, match="chapter leakage"):
        enrollment_scores(rows, vectors, "multi_raw", np.zeros(2))


def test_fusion_selection_can_reject_an_unhelpful_audio_branch():
    good = np.array([[.95, .05], [.05, .95]])
    bad = np.array([[.05, .95], [.95, .05]])
    selected, _ = select_fusion(good, bad, np.array([0, 1]))
    assert selected["alpha"] == 0
    selected, _ = select_fusion(bad, good, np.array([0, 1]))
    assert selected["alpha"] == 1
