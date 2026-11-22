"""End-to-end pipeline test with a fake video engine (real ffmpeg clips, real rendering).

Exercises everything except the diffusion model: script -> scenes -> TTS -> per-scene
generation + checkpoints -> concat -> ambience/mix/loudnorm -> SRT -> mux -> quality -> manifest,
then resume (completed scenes skipped). Skipped when ffmpeg is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

ffmpeg = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not installed")

HINDI = (
    "रात के बारह बजे थे। पुराने हवेली में एक अजीब सी आवाज़ आई। "
    "रमेश ने धीरे से दरवाज़ा खोला। अंदर अंधेरा था और हवा ठंडी थी। "
    "अचानक पीछे से किसी ने उसका नाम पुकारा।"
)


class FakeEngine:
    """Produces genuine short video clips with ffmpeg (test pattern), honouring the request."""

    calls = 0

    def __init__(self, *a, **k):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    def generate(self, req):
        from app.engines.base import GenerationResult

        FakeEngine.calls += 1
        t0 = time.time()
        secs = max(0.5, req.num_frames / req.fps)
        req.output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-y", "-f", "lavfi", "-i",
               f"testsrc=size={req.width}x{req.height}:rate={req.fps}:duration={secs}",
               "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(req.output_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return GenerationResult(success=False, error=r.stderr[-300:])
        return GenerationResult(success=True, output_path=req.output_path, frames=req.num_frames,
                                width=req.width, height=req.height, fps=req.fps,
                                seconds=time.time() - t0)


def _run(tmp_path, monkeypatch, resume_id=None):
    import app.pipeline as pl

    monkeypatch.setattr(pl, "get_engine", lambda *a, **k: FakeEngine())
    # sandbox has no GPU: pretend the registry check passed (as it does on a T4)
    monkeypatch.setattr(pl.ModelCompatibilityChecker, "check",
                        lambda self, spec: {"model": spec.id, "compatible": True, "reasons": [],
                                            "status": spec.status})
    # tiny profile so the test is fast; sizes still flow through the real renderer
    monkeypatch.setitem(pl.PROFILES, "balanced", {**pl.PROFILES["balanced"], "width": 128, "height": 96,
                                                  "frames": 8, "steps": 1})
    events = []
    pipe = pl.Pipeline(data_dir=tmp_path / "data", progress=lambda *e: events.append(e))
    manifest = pipe.run(HINDI, engine_id="wan2.1", profile="balanced", style="horror",
                        tts_backend="silent", seed=1, resume_project_id=resume_id)
    return manifest, events


def test_pipeline_end_to_end_and_resume(tmp_path, monkeypatch):
    FakeEngine.calls = 0
    manifest, events = _run(tmp_path, monkeypatch)
    final = Path(manifest["final"])
    assert final.exists() and final.stat().st_size > 1000
    assert manifest["project_id"].startswith("proj_")
    assert len(manifest["scenes"]) >= 2
    assert all(r["success"] for r in manifest["scene_results"])
    assert manifest["upscaled"] is False
    # narration manifest is a dict in BOTH fresh and resume paths (regression for .get crash)
    stages = [e[0] for e in events]
    assert "narration" in stages and "render" in stages and "done" in stages
    n_scenes = len(manifest["scenes"])
    assert FakeEngine.calls == n_scenes
    # final mp4 has a video AND an audio stream
    probe = subprocess.run([ffmpeg, "-i", str(final)], capture_output=True, text=True).stderr
    assert "Video:" in probe and "Audio:" in probe
    # subtitles were written
    assert (final.parent / "subtitles.srt").read_text(encoding="utf-8").strip()
    # narration-driven timeline: final length == sum of per-scene segments ~= narration + pauses,
    # and every clip was time-fitted (never cut) to its own narration
    assert len(manifest["timeline"]) == n_scenes
    total = sum(t["segment_seconds"] for t in manifest["timeline"])
    assert abs(manifest["duration_seconds"] - total) < 0.5 + 0.02 * total
    assert manifest["duration_seconds"] >= manifest["narration"]["duration_seconds"] - 0.1
    for t in manifest["timeline"]:
        assert t["segment_seconds"] >= t["narration_seconds"] - 0.05
    assert manifest["generation_fps"] == 16 and manifest["output_fps"] == 24

    # --- resume: nothing regenerated, same script hash enforced ---
    FakeEngine.calls = 0
    m2, ev2 = _run(tmp_path, monkeypatch, resume_id=manifest["project_id"])
    assert FakeEngine.calls == 0, "resume must skip completed scenes"
    assert m2["script_hash"] == manifest["script_hash"]
    assert Path(m2["final"]).exists()


def test_resume_rejects_changed_script(tmp_path, monkeypatch):
    manifest, _ = _run(tmp_path, monkeypatch)
    import app.pipeline as pl

    monkeypatch.setattr(pl, "get_engine", lambda *a, **k: FakeEngine())
    pipe = pl.Pipeline(data_dir=tmp_path / "data")
    with pytest.raises(ValueError, match="hash mismatch"):
        pipe.run(HINDI + " बदलाव।", tts_backend="silent", resume_project_id=manifest["project_id"])
