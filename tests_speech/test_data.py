import io
import json

import numpy as np
import pytest
import soundfile as sf

from speechlab.common import add_noise, audio_array, fingerprint, read_manifest, validate_splits


def records():
    return [{"id": str(i), "speaker_id": str(i), "split": split, "audio_sha256": str(i), "text": "hello"}
            for i, split in enumerate(["train", "validation", "test"])]


def test_speaker_and_audio_leakage_rejected():
    rows = records()
    validate_splits(rows)
    rows[-1]["speaker_id"] = rows[0]["speaker_id"]
    with pytest.raises(ValueError, match="Speaker leakage"):
        validate_splits(rows)
    rows = records()
    rows[-1]["audio_sha256"] = rows[0]["audio_sha256"]
    with pytest.raises(ValueError, match="Duplicate audio"):
        validate_splits(rows)


def test_manifest_checks_actual_audio_bytes_and_paths(tmp_path):
    rows = records()
    for i, row in enumerate(rows):
        path = tmp_path / f"{i}.wav"
        sf.write(path, np.random.default_rng(i).normal(0, .1, 1600), 16000)
        row.update(path=path.name, audio_sha256=fingerprint(path))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(read_manifest(manifest)) == 3
    (tmp_path / "0.wav").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(manifest)
    rows[0]["path"] = "../outside.wav"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="escapes"):
        read_manifest(manifest)


def test_stereo_resampling():
    data = io.BytesIO()
    sf.write(data, np.ones((8000, 2), dtype=np.float32) * .2, 8000, format="WAV")
    data.seek(0)
    result = audio_array(data)
    assert result.shape == (16000,)
    assert result.dtype == np.float32


def test_noise_matches_requested_snr_and_is_reproducible():
    audio = np.random.default_rng(1).normal(size=16000).astype(np.float32)
    noisy = add_noise(audio, 10, 3)
    np.testing.assert_array_equal(noisy, add_noise(audio, 10, 3))
    snr = 10 * np.log10(np.mean(audio ** 2) / np.mean((noisy - audio) ** 2))
    assert snr == pytest.approx(10, abs=.001)
