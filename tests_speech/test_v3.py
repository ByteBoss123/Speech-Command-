import numpy as np
import pytest
from speechlab.intent_external import comparison, exclude_used
from speechlab.prepare_v3 import fresh_partition
from speechlab.speaker_v2 import enrollment_scores
from speechlab.speaker_v3 import compact_trials, select


def test_fresh_speakers_exclude_all_previous_and_keep_folds_disjoint():
    groups = {s: {"a": list(range(5)), "b": list(range(5))} for s in ("old", "one", "two", "three")}
    split = fresh_partition(groups, {"old"}, validation=1, test=2)
    val, test = [{s for s, _ in split[key]} for key in ("validation", "test")]
    assert not val & test and "old" not in val | test and len(val | test) == 3
    with pytest.raises(ValueError, match="eligible fresh"):
        fresh_partition(groups, {"old", "one"}, validation=1, test=2)


def test_external_holdout_excludes_id_text_audio_and_new_duplicates():
    old = [{"id": "old", "text": "Old request!", "source_audio_sha256": "old-audio"}]
    candidates = [
        {"id": "old", "text": "other", "source_audio_sha256": "x"},
        {"id": "a", "text": "OLD REQUEST", "source_audio_sha256": "y"},
        {"id": "b", "text": "new words", "source_audio_sha256": "old-audio"},
        {"id": "c", "text": "fresh example", "source_audio_sha256": "fresh"},
        {"id": "d", "text": "Fresh example!", "source_audio_sha256": "different"}]
    assert [r["id"] for r in exclude_used(candidates, old)] == ["c"]
    assert exclude_used(candidates[::-1], old) == exclude_used(candidates, old)


def test_paired_statistics_use_discordant_pairs_and_keep_null():
    truth = np.array([0] * 8 + [1] * 8)
    baseline = truth.copy(); baseline[:8] = 1
    result = comparison(truth, truth, baseline, ["en-AU"] * 8 + ["en-GB"] * 8)
    assert result["improved"] == 8 and result["regressed"] == 0
    assert result["exact_mcnemar_two_sided_p"] == pytest.approx(2 / 2**8)
    assert result["accuracy_difference"] == .5
    null = comparison(truth, baseline, baseline, ["en-AU"] * 16)
    assert null["ci95"] == [0., 0.] and null["exact_mcnemar_two_sided_p"] == 1.


def test_false_accept_budget_cannot_be_overridden_by_better_recall():
    calibration = {"five_raw": {"far": .006, "frr": .01}, "five_centered": {"far": .004, "frr": .2}}
    assert select(calibration) == "five_centered"
    calibration["five_centered"]["far"] = .007
    with pytest.raises(ValueError, match="budget"): select(calibration)


def test_fixed_three_enrollment_baseline_does_not_use_extra_recordings():
    rows = [{"speaker_id": "a", "chapter_id": "enroll", "verification_role": "enroll"} for _ in range(5)]
    rows += [{"speaker_id": "a", "chapter_id": "probe", "verification_role": "probe"}]
    vectors = np.array([[1., 0.]] * 3 + [[0., 1.]] * 2 + [[1., 0.]])
    three = enrollment_scores(rows, vectors, "multi_raw", np.zeros(2), 3)
    five = enrollment_scores(rows, vectors, "multi_raw", np.zeros(2), 5)
    assert three[0]["score"] == 1. and five[0]["score"] < 1.
    assert compact_trials({"three": three, "five": five})[0]["scores"]["three"] == 1.
    with pytest.raises(ValueError, match="Mismatched"): compact_trials({"three": three, "five": []})
    with pytest.raises(ValueError, match="Insufficient"): enrollment_scores(rows, vectors, "multi_raw", np.zeros(2), 6)
