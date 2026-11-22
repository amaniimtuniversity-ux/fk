"""VideoRenderer: concat scenes + narration + ambience + subtitles -> MP4.

Final: H.264 / AAC, 16:9, 30fps where practical. Upscaling to 1080p is
labeled as upscale (never claimed as native generation).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(cmd: list[str], timeout: int = 1800) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stderr or "")[-2000:]
    except Exception as e:
        return False, str(e)


class VideoRenderer:
    def __init__(self, fps: int = 30, crf: int = 18, preset: str = "medium"):
        self.fps = fps
        self.crf = crf
        self.preset = preset

    def concat_clips(self, clips: list[Path], output_path: Path) -> tuple[bool, str]:
        if not clips:
            return False, "No clips to concat."
        if not ffmpeg_available():
            return False, "ffmpeg not found. Install ffmpeg."
        output_path.parent.mkdir(parents=True, exist_ok=True)
        list_file = output_path.parent / "concat_list.txt"
        list_file.write_text("".join(f"file '{c.resolve()}'\n" for c in clips), encoding="utf-8")
        ok, err = _run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c:v", "libx264", "-crf", str(self.crf), "-preset", self.preset,
            "-pix_fmt", "yuv420p", "-r", str(self.fps), str(output_path),
        ])
        return ok, err

    def mux_audio_subtitles(self, video: Path, audio: Path, srt: Optional[Path],
                            output_path: Path, grade: bool = True) -> tuple[bool, str]:
        """Mux video + audio (+ optional SRT as a selectable mov_text track).

        Subtitles are embedded as a soft track (not burned in) so the viewer can toggle them;
        the colour grade is applied to the video stream only.
        """
        if not ffmpeg_available():
            return False, "ffmpeg not found."
        has_srt = srt is not None and srt.exists() and srt.stat().st_size > 0
        cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(audio)]
        if has_srt:
            cmd += ["-i", str(srt)]
        if grade:
            cmd += ["-vf", "eq=contrast=1.05:saturation=1.1:brightness=0.01"]
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        if has_srt:
            cmd += ["-map", "2:s:0", "-c:s", "mov_text", "-metadata:s:s:0", "language=hin"]
        cmd += ["-c:v", "libx264", "-crf", str(self.crf), "-preset", self.preset, "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(output_path)]
        return _run(cmd)

    def upscale_1080p(self, video: Path, output_path: Path) -> tuple[bool, str]:
        """Lanczos upscale to 1920x1080. Labeled as upscale in manifest."""
        if not ffmpeg_available():
            return False, "ffmpeg not found."
        return _run([
            "ffmpeg", "-y", "-i", str(video),
            "-vf", "scale=1920:1080:flags=lanczos",
            "-c:v", "libx264", "-crf", str(self.crf), "-pix_fmt", "yuv420p",
            "-c:a", "copy", str(output_path),
        ])

    def probe(self, video: Path) -> dict[str, Any]:
        """ffprobe metadata (duration, resolution, fps)."""
        if shutil.which("ffprobe") is None:
            return {"available": False}
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(video)],
                capture_output=True, text=True, timeout=60,
            )
            import json

            data = json.loads(r.stdout)
            v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
            return {
                "available": True,
                "duration": float(data.get("format", {}).get("duration", 0)),
                "width": v.get("width"), "height": v.get("height"),
                "fps": v.get("r_frame_rate"), "codec": v.get("codec_name"),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
