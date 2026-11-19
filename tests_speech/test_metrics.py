import numpy as np
import pytest

from speechlab.metrics import (approximate_eer, asr_summary, calibrate_threshold,
                               make_trials, paired_wer_bootstrap, transcript_counts,
                               verification_rates)


def test_wer_is_corpus_weighted_and_can_exceed_one():
    a = {"id": "a", "speaker_id": "s1", **transcript_counts("one", "one two three")}
    b = {"id": "b", "speaker_id": "s2", **transcript_counts("a b c", "a b c")}
    assert asr_summary([a, b])["wer"] == .5
    assert asr_summary([a])["wer"] == 2


def test_paired_bootstrap_preserves_speakers_and_direction():
    base = [{"id": str(i), "speaker_id": str(i), **transcript_counts("one two", "one")} for i in range(3)]
    adapted = [{"id": str(i), "speaker_id": str(i), **transcript_counts("one two", "one two")} for i in range(3)]
    result = paired_wer_bootstrap(base, adapted)
    assert result["ci95"] == [-.5, -.5]
    assert result["resampling_unit"] == "speaker"
    with pytest.raises(ValueError, match="pairing"):
        paired_wer_bootstrap(base, adapted[::-1])


def test_threshold_handles_ties_and_reject_all():
    calibration = calibrate_threshold([1, 0, 1, 0], [.5] * 4, target_far=0)
    assert calibration["threshold"] > .5
    assert calibration["far"] == 0
    assert calibration["frr"] == 1


def test_threshold_selected_on_development_does_not_guarantee_test_far():
    c = calibrate_threshold([0, 0, 1, 1], [.1, .2, .8, .9], target_far=0)
    assert c["far"] == 0 and c["frr"] == 0
    rates = verification_rates([0, 1], [.95, .7], c["threshold"])
    assert rates["far"] == 1 and rates["frr"] == 1
    assert approximate_eer([0, 0, 1, 1], [.1, .2, .8, .9])["eer"] == 0


@pytest.mark.parametrize("labels,scores", [([1, 1], [.1, .2]), ([0, 1], [np.nan, .2]), ([0, 1], [.2])])
def test_invalid_verification_inputs_rejected(labels, scores):
    with pytest.raises(ValueError):
        calibrate_threshold(labels, scores)


def test_trials_are_unique_balanced_and_have_correct_labels():
    rows = [{"id": str(i), "speaker_id": str(i // 3)} for i in range(12)]
    trials = make_trials(rows)
    assert trials == make_trials(rows)
    assert len({(i, j) for i, j, _ in trials}) == len(trials)
    assert sum(t[2] for t in trials) == len(trials) // 2
    for i, j, label in trials:
        assert i != j and label == int(rows[i]["speaker_id"] == rows[j]["speaker_id"])
