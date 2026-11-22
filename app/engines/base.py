"""VideoGenerationEngine ABC + request/result types."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class GenerationRequest:
    prompt: str
    output_path: Path
    width: int = 832
    height: int = 480
    num_frames: int = 49
    fps: int = 24
    num_inference_steps: int = 30
    guidance_scale: float = 6.0
    seed: int = 42
    image_path: Optional[Path] = None  # for I2V engines
    negative_prompt: str = ""
    progress_callback: Optional[Callable[[int, int, str], None]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt, "output_path": str(self.output_path),
            "width": self.width, "height": self.height, "num_frames": self.num_frames,
            "fps": self.fps, "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale, "seed": self.seed,
            "image_path": str(self.image_path) if self.image_path else None,
        }


@dataclass
class GenerationResult:
    success: bool
    output_path: Optional[Path] = None
    frames: int = 0
    width: int = 0
    height: int = 0
    fps: int = 0
    seconds: float = 0.0
    peak_vram_bytes: int = 0
    peak_ram_bytes: int = 0
    engine: str = ""
    model: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "success": self.success, "output_path": str(self.output_path) if self.output_path else None,
            "frames": self.frames, "width": self.width, "height": self.height, "fps": self.fps,
            "seconds": round(self.seconds, 2), "peak_vram_bytes": self.peak_vram_bytes,
            "peak_ram_bytes": self.peak_ram_bytes, "engine": self.engine, "model": self.model,
            "error": self.error, "warnings": self.warnings, "metrics": self.metrics,
        }
        return d



def load_diffusers_pipeline(pipeline_cls, repo: str, dtype, prefer_variant: Optional[str] = "fp16",
                            **kwargs):
    """from_pretrained with a safe precision-variant fallback.

    Some repos ship dedicated `fp16` weight files (e.g. SVD-XT) — using them avoids downloading
    fp32. Others (e.g. Wan2.1-*-Diffusers) do not, and diffusers raises
    "no such modeling files are available". In that case we load the default files and let
    `torch_dtype` cast them to the requested dtype. Nothing is silently downgraded: the
    compute dtype is exactly what the precision plan asked for in both paths.
    """
    if prefer_variant:
        try:
            return pipeline_cls.from_pretrained(repo, torch_dtype=dtype, variant=prefer_variant, **kwargs)
        except (ValueError, OSError) as e:
            msg = str(e)
            if "variant" not in msg and "no such modeling files" not in msg:
                raise
    return pipeline_cls.from_pretrained(repo, torch_dtype=dtype, **kwargs)


def normalize_frames(frames) -> "np.ndarray":
    """Return uint8 array [F,H,W,3] from diffusers output (np float [0,1], uint8, torch, or PIL)."""
    import numpy as np

    if hasattr(frames, "detach"):  # torch tensor
        frames = frames.detach().float().cpu().numpy()
    if isinstance(frames, (list, tuple)):
        if len(frames) == 0:
            raise ValueError("Engine returned zero frames.")
        if hasattr(frames[0], "convert"):  # PIL images
            frames = [np.asarray(f.convert("RGB")) for f in frames]
        elif hasattr(frames[0], "detach"):
            frames = [f.detach().float().cpu().numpy() for f in frames]
        frames = np.stack([np.asarray(f) for f in frames])
    arr = np.asarray(frames)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] != 3 and arr.shape[1] == 3:  # [F,3,H,W] -> [F,H,W,3]
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected frame array shape {arr.shape}; expected [F,H,W,3].")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if np.isfinite(arr).all() and arr.min() >= -1.0 and arr.max() <= 1.0 and arr.min() < 0:
            arr = (arr + 1.0) / 2.0  # [-1,1] -> [0,1]
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return arr


def validate_frames(arr, where: str = "engine output") -> None:
    """Fail loudly on NaN/black/constant output — never ship a fake or broken clip.

    FP16 overflow in a BF16-trained transformer typically shows up as all-NaN latents which
    decode to uniform black or grey frames. Raise with a clear, actionable message.
    """
    import numpy as np

    raw = frames_raw_stats(arr)
    if raw["nan"]:
        raise RuntimeError(f"{where}: NaN/Inf in {raw['nan']} frame values — FP16 overflow. "
                           "Try profile 'max_quality' (fp32 VAE) or a different engine.")
    if raw["std"] < 2.0:
        raise RuntimeError(f"{where}: frames are uniform (std={raw['std']:.2f}, mean={raw['mean']:.1f}) — "
                           "black/blank output, likely FP16 overflow or a broken decode. Refusing to use it.")


def frames_raw_stats(arr) -> dict:
    import numpy as np

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        nan = int((~np.isfinite(a.astype(np.float32))).sum())
        a = np.nan_to_num(a.astype(np.float32))
    else:
        nan = 0
    a = a.astype(np.float32)
    return {"nan": nan, "std": float(a.std()), "mean": float(a.mean()), "frames": int(a.shape[0])}


class VideoGenerationEngine(ABC):
    engine_id: str = "base"
    model_id: str = ""
    supports_t2v: bool = True
    supports_i2v: bool = False

    def __init__(self, model_repo: str, precision_plan: Optional[dict] = None,
                 attention_plan: Optional[dict] = None, offload_mode: str = "sequential"):
        self.model_repo = model_repo
        self.precision_plan = precision_plan or {}
        self.attention_plan = attention_plan or {}
        self.offload_mode = offload_mode
        self._pipeline = None
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model pipeline (idempotent)."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one clip. Must produce REAL moving video (no static zoom)."""

    def unload(self) -> None:
        self._pipeline = None
        self._loaded = False
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _peak_memory(self) -> tuple[int, int]:
        vram = ram = 0
        try:
            import torch

            if torch.cuda.is_available():
                vram = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        try:
            import psutil

            ram = int(psutil.Process().memory_info().rss)
        except Exception:
            pass
        return vram, ram

    def _reset_peak(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _save_frames(self, frames, output_path: Path, fps: int) -> None:
        """Validate and save frames as a real H.264 MP4 at the exact generated resolution.

        - Accepts diffusers output (float [0,1] numpy, uint8, torch, or PIL).
        - Raises on NaN / uniform (black) output instead of writing a broken clip.
        - macro_block_size=1 so imageio never silently resizes to a multiple of 16.
        """
        import numpy as np

        arr = normalize_frames(frames)
        validate_frames(arr, where=f"{self.engine_id} output")
        if arr.shape[1] % 2 or arr.shape[2] % 2:  # yuv420p needs even dims; crop 1px, never rescale
            arr = arr[:, : arr.shape[1] - arr.shape[1] % 2, : arr.shape[2] - arr.shape[2] % 2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        for attempt in range(2):
            try:
                if attempt == 1:
                    import imageio_ffmpeg  # noqa: F401  (ensures a bundled ffmpeg binary exists)
                import imageio.v2 as imageio

                imageio.mimsave(str(output_path), list(arr), fps=fps, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=1)
                if output_path.stat().st_size < 1024:
                    raise RuntimeError("written MP4 is suspiciously small")
                return
            except Exception as e:  # try again with the bundled ffmpeg
                errors.append(f"{type(e).__name__}: {e}")
        raise RuntimeError("No MP4 writer available (need imageio + ffmpeg). Install imageio[ffmpeg]. "
                           f"Errors: {errors}")

    def _report(self, request: GenerationRequest, step: int, total: int, msg: str = "") -> None:
        if request.progress_callback is not None:
            try:
                request.progress_callback(step, total, msg)
            except Exception:
                pass
