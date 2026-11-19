import io

import numpy as np
import soundfile as sf

from speechlab import prepare


def test_preparation_enforces_quotas_filters_short_audio_and_closes_source(tmp_path, monkeypatch):
    closed = []

    def signal(seconds, seed):
        b = io.BytesIO()
        sf.write(b, np.random.default_rng(seed).normal(0, .1, int(seconds * 16000)), 16000, format="WAV")
        return b.getvalue()

    def source(*args):
        try:
            yield {"id": "short", "speaker_id": 0, "chapter_id": 0, "text": "hello", "audio": {"bytes": signal(.1, 0)}}
            for i in range(20):
                yield {"id": str(i), "speaker_id": i // 2, "chapter_id": i // 2, "text": "hello world",
                       "audio": {"bytes": signal(2.1, i + 1)}}
        finally:
            closed.append(True)

    monkeypatch.setattr(prepare, "source_records", source)
    rows, scanned = prepare.select_split("train.100", "train", "fixture", 2, 2, tmp_path, 100)
    assert len(rows) == 4 and scanned == 5
    assert {r["speaker_id"] for r in rows} == {"0", "1"}
    assert all(r["duration_seconds"] == 2.1 for r in rows)
    assert all((tmp_path / r["path"]).exists() for r in rows)
    assert closed == [True]
