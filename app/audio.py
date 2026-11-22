"""AudioEngine: narration sync, ambience, SFX, ducking, loudness, subtitles.

Uses FFmpeg (required on Kaggle: `apt-get install ffmpeg` or imageio-ffmpeg).
All functions degrade with clear errors if ffmpeg is missing — never fake audio.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(cmd: list[str]) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return r.returncode == 0, (r.stderr or "")[-1500:]
    except Exception as e:
        return False, str(e)


def loudness_normalize(input_path: Path, output_path: Path, target_lufs: float = -16.0) -> tuple[bool, str]:
    """EBU R128 loudnorm to target LUFS (single pass), output 48 kHz AAC."""
    if not ffmpeg_available():
        return False, "ffmpeg not found. Install ffmpeg (Kaggle: apt-get install -y ffmpeg)."
    ok, err = _run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", "48000",  # loudnorm upsamples to 192 kHz internally; pin a standard rate
        "-c:a", "aac", "-b:a", "192k", str(output_path),
    ])
    return ok, err


def mix_narration_ambience(narration: Path, ambience: Path, output_path: Path,
                           ambience_volume: float = 0.15) -> tuple[bool, str]:
    """Mix narration (full) with low ambience bed (ducked)."""
    if not ffmpeg_available():
        return False, "ffmpeg not found."
    ok, err = _run([
        "ffmpeg", "-y", "-i", str(narration), "-i", str(ambience),
        "-filter_complex",
        f"[1:a]volume={ambience_volume}[amb];[0:a][amb]amix=inputs=2:duration=first:dropout_transition=0[a]",
        "-map", "[a]", "-c:a", "aac", "-b:a", "192k", str(output_path),
    ])
    return ok, err


def generate_ambience(output_path: Path, seconds: float, kind: str = "wind") -> tuple[bool, str]:
    """Synthesize a simple ambience bed with ffmpeg filters (no external files).

    kind: wind (filtered noise), drone (low sine), rain (noise + lowpass).
    This is a procedural bed, clearly not a recorded soundscape.
    """
    if not ffmpeg_available():
        return False, "ffmpeg not found."
    if kind == "drone":
        src = ["-f", "lavfi", "-i", "sine=frequency=55:duration=%f" % seconds]
        filt = "volume=0.08,afade=t=in:st=0:d=1,afade=t=out:st=%f:d=2" % max(0.1, seconds - 2)
    elif kind == "rain":
        src = ["-f", "lavfi", "-i", "anoisesrc=d=%f:c=brown:a=0.3" % seconds]
        filt = "lowpass=f=1200,volume=0.25,afade=t=in:st=0:d=1,afade=t=out:st=%f:d=2" % max(0.1, seconds - 2)
    else:  # wind
        src = ["-f", "lavfi", "-i", "anoisesrc=d=%f:c=pink:a=0.4" % seconds]
        filt = "lowpass=f=400,volume=0.2,afade=t=in:st=0:d=1,afade=t=out:st=%f:d=2" % max(0.1, seconds - 2)
    ok, err = _run(["ffmpeg", "-y", *src, "-af", filt, "-c:a", "aac", "-b:a", "128k", str(output_path)])
    return ok, err


def pad_audio_to_duration(input_path: Path, seconds: float, output_path: Path) -> tuple[bool, str]:
    """Exact-length WAV: pad with silence or trim to `seconds` (keeps speech untouched when shorter)."""
    if not ffmpeg_available():
        return False, "ffmpeg not found."
    return _run(["ffmpeg", "-y", "-i", str(input_path), "-af", f"apad,atrim=0:{seconds:.3f}",
                 "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(output_path)])


def concat_audio(parts: list[Path], output_path: Path) -> tuple[bool, str]:
    """Lossless concat of WAV parts in order."""
    if not ffmpeg_available():
        return False, "ffmpeg not found."
    if not parts:
        return False, "no audio parts"
    lst = output_path.parent / (output_path.stem + "_list.txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    return _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:a", "pcm_s16le", str(output_path)])


def audio_duration(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        r = subprocess.run(["ffmpeg", "-i", str(path)], capture_output=True, text=True)
        import re

        m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", r.stderr)
        return (int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])) if m else 0.0


def write_srt(scenes: list[dict], output_path: Path) -> Path:
    """Write SRT subtitles from scene narrations (timed by cumulative duration)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t = 0.0
    lines: list[str] = []
    for i, sc in enumerate(scenes, 1):
        dur = float(sc.get("duration_seconds", 5.0))
        start, end = t, t + dur
        fmt = lambda s: f"{int(s//3600):02d}:{int(s%3600//60):02d}:{int(s%60):02d},{int((s%1)*1000):03d}"
        lines.append(f"{i}\n{fmt(start)} --> {fmt(end)}\n{sc['narration']}\n")
        t = end
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
