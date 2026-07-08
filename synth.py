"""
synth.py

espeak-ng synthesis wrapper supporting voice selection (accent/gender) in
addition to rate/pitch, used to build genuine acoustic diversity into the
training data.
"""

import subprocess
from pathlib import Path


def synthesize(text: str, out_path: Path, voice: str, rate: int, pitch: int) -> Path:
    out_path = Path(out_path)
    raw_path = out_path.with_suffix(".raw.wav")

    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", str(rate), "-p", str(pitch), "-w", str(raw_path), text],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["sox", str(raw_path), "-r", "16000", "-c", "1", str(out_path)],
        check=True,
        capture_output=True,
    )
    raw_path.unlink(missing_ok=True)
    return out_path
