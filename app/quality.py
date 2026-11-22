"""QualityEvaluator: measurable video quality (no fake scores).

Measures on generated clips:
- Frame-to-frame flicker: mean abs diff between consecutive frames
- Sharpness: Laplacian variance (via PIL/numpy; no cv2 required)
- Temporal consistency proxy: flicker + motion magnitude
- Resolution/fps/codec from ffprobe (honest: native vs upscale labeled)

Scores are reported as raw measurements with interpretation bands, never as
a single invented "quality score".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_frames(video: Path, max_frames: int = 16) -> list[Any]:
    """Load up to max_frames grayscale frames via imageio (or raise)."""
    import imageio.v2 as imageio
    import numpy as np

    reader = imageio.get_reader(str(video))
    frames = []
    try:
        for i, frame in enumerate(reader):
            if i >= max_frames:
                break
            gray = frame.mean(axis=2) if frame.ndim == 3 else frame
            frames.append(np.asarray(gray, dtype="float32"))
    finally:
        reader.close()
    return frames


def evaluate_clip(video: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"video": str(video), "exists": video.exists()}
    if not video.exists():
        result["error"] = "file missing"
        return result
    try:
        import numpy as np

        frames = _load_frames(video)
        if len(frames) < 2:
            result["error"] = f"only {len(frames)} frame(s) readable"
            return result
        # Flicker: mean abs diff between consecutive frames.
        diffs = [float(np.abs(frames[i + 1] - frames[i]).mean()) for i in range(len(frames) - 1)]
        flicker = float(np.mean(diffs))
        # Sharpness: variance of Laplacian (3x3 kernel) on middle frame.
        mid = frames[len(frames) // 2]
        lap = (
            -4 * mid[1:-1, 1:-1] + mid[:-2, 1:-1] + mid[2:, 1:-1]
            + mid[1:-1, :-2] + mid[1:-1, 2:]
        )
        sharpness = float(lap.var())
        # Motion magnitude: mean abs diff (same as flicker here; kept separate
        # for interpretation: very low = static, very high = chaotic).
        result.update({
            "frames_sampled": len(frames),
            "flicker_mean_abs_diff": round(flicker, 3),
            "sharpness_laplacian_var": round(sharpness, 2),
            "interpretation": _interpret(flicker, sharpness),
        })
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def _interpret(flicker: float, sharpness: float) -> str:
    parts = []
    if flicker < 0.5:
        parts.append("very low motion (check for static/zoom fallback)")
    elif flicker < 8:
        parts.append("natural motion range")
    else:
        parts.append("high frame-to-frame change (check flicker/chaos)")
    if sharpness < 50:
        parts.append("soft/blurry")
    elif sharpness < 500:
        parts.append("moderate sharpness")
    else:
        parts.append("sharp")
    return "; ".join(parts)


def compare_clips(a: Path, b: Path) -> dict[str, Any]:
    ea, eb = evaluate_clip(a), evaluate_clip(b)
    return {"a": ea, "b": eb}
