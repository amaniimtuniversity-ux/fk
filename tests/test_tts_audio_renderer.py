"""Tests: TTS silent fallback (flagged), SRT, renderer guards (no fake outputs)."""
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio import write_srt
from app.renderer import VideoRenderer
from app.tts import HindiTTSService


def test_tts_silent_fallback_flagged(tmp_path):
    tts = HindiTTSService(backend="silent")
    out = tmp_path / "narr.wav"
    result = tts.synthesize_script("यह एक परीक्षण है। दूसरा वाक्य।", out)
    assert result.success is True
    assert result.is_silent_fallback is True
    assert out.exists() and out.stat().st_size > 44
    with wave.open(str(out)) as w:
        assert w.getnframes() > 0


def test_srt_timestamps():
    scenes = [
        {"narration": "पहला वाक्य।", "duration_seconds": 5.0},
        {"narration": "दूसरा वाक्य।", "duration_seconds": 5.0},
    ]
    out = Path("/tmp/test_sub.srt")
    write_srt(scenes, out)
    text = out.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:05,000" in text
    assert "00:00:05,000 --> 00:00:10,000" in text


def test_renderer_no_clips_error(tmp_path):
    r = VideoRenderer()
    ok, err = r.concat_clips([], tmp_path / "out.mp4")
    assert ok is False
    assert "No clips" in err


def test_renderer_probe_missing_ffmpeg(tmp_path):
    # probe must not crash when ffprobe is absent.
    r = VideoRenderer()
    result = r.probe(tmp_path / "nonexistent.mp4")
    assert "available" in result
