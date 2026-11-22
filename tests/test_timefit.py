"""Time-fit: a genuine short clip is extended to the narration length with real frames only."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from app.renderer import VideoRenderer

ffmpeg = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not installed")


def _clip(path, seconds, fps=16):
    subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i", f"testsrc=size=160x96:rate={fps}:duration={seconds}",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", str(path)],
                   capture_output=True, check=True)


@pytest.mark.parametrize("clip_s,target_s,expect_pingpong", [(1.0, 2.0, False), (1.0, 4.0, True), (3.0, 2.5, False)])
def test_fit_clip_to_duration(tmp_path, clip_s, target_s, expect_pingpong):
    src = tmp_path / "src.mp4"
    _clip(src, clip_s)
    r = VideoRenderer(fps=24)
    ok, err, info = r.fit_clip_to_duration(src, target_s, tmp_path / "fit.mp4")
    assert ok, err
    assert info["pingpong"] == expect_pingpong
    out = r.duration(tmp_path / "fit.mp4")
    if clip_s >= target_s:
        assert abs(out - clip_s) < 0.15          # longer clip is kept whole, never cut
        assert info["slow_factor"] == 1.0
    else:
        assert abs(out - target_s) < 0.15        # extended exactly to the narration length
        assert info["slow_factor"] <= 2.5
