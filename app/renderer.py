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
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output_path)]
        return _run(cmd)

    def duration(self, video: Path) -> float:
        """Duration in seconds via ffprobe (falls back to ffmpeg -i parse)."""
        d = self.probe(video).get("duration") if shutil.which("ffprobe") else None
        if d:
            return float(d)
        r = subprocess.run(["ffmpeg", "-i", str(video)], capture_output=True, text=True)
        import re

        m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", r.stderr)
        return (int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])) if m else 0.0

    def fit_clip_to_duration(self, clip: Path, target_seconds: float, output_path: Path,
                             max_slow_factor: float = 2.5) -> tuple[bool, str, dict]:
        """Make a GENUINE generated clip last exactly `target_seconds` (the narration length).

        Policy (quality rules: never a static image, never a fake zoom):
        - clip longer than target  -> keep it whole (audio is padded instead, see pipeline).
        - clip shorter             -> slow motion of the real frames (setpts), up to max_slow_factor;
                                      if still short, ping-pong (forward+reverse, all real frames)
                                      until long enough, then trim exactly.
        Output is re-timed to self.fps (frames duplicated as needed). Returns (ok, err, info).
        """
        if not ffmpeg_available():
            return False, "ffmpeg not found.", {}
        src_dur = self.duration(clip)
        info: dict = {"clip_seconds": round(src_dur, 3), "target_seconds": round(target_seconds, 3),
                      "slow_factor": 1.0, "pingpong": False}
        if src_dur <= 0:
            return False, f"could not read duration of {clip}", info
        if src_dur >= target_seconds - 0.02:
            ok, err = _run(["ffmpeg", "-y", "-i", str(clip), "-c:v", "libx264", "-crf", str(self.crf),
                            "-preset", self.preset, "-pix_fmt", "yuv420p", "-r", str(self.fps), "-an",
                            str(output_path)])
            info["out_seconds"] = round(self.duration(output_path), 3) if ok else 0
            return ok, err, info
        factor = target_seconds / src_dur
        filters = []
        if factor > max_slow_factor:
            # extend with real frames first: forward + reverse, repeated
            reps = int(factor / max_slow_factor) + 1
            seg = "[0:v]split=2[f][r];[r]reverse[rv];[f][rv]concat=n=2:v=1:a=0[pp]"
            if reps > 1:
                seg += f";[pp]loop=loop={reps - 1}:size=32767:start=0[pp]"
            filters.append(seg)
            base_dur = src_dur * 2 * reps
            factor = max(1.0, target_seconds / base_dur)
            info["pingpong"] = True
            vin = "[pp]"
        else:
            vin = "[0:v]"
        filters.append(f"{vin}setpts={factor:.5f}*PTS,fps={self.fps},trim=duration={target_seconds:.3f}[v]")
        info["slow_factor"] = round(factor, 3)
        ok, err = _run(["ffmpeg", "-y", "-i", str(clip), "-filter_complex", ";".join(filters), "-map", "[v]",
                        "-c:v", "libx264", "-crf", str(self.crf), "-preset", self.preset, "-pix_fmt", "yuv420p",
                        "-an", str(output_path)])
        info["out_seconds"] = round(self.duration(output_path), 3) if ok else 0
        return ok, err, info

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
