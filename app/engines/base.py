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
        """Save numpy/PIL frames as MP4 via imageio or PIL fallback.

        frames: list of PIL images or numpy arrays [H,W,3] uint8.
        Uses imageio if available, else writes an animated GIF renamed .mp4
        is FORBIDDEN (fake video). Raises if no writer available.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio

            imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264", quality=8)
            return
        except Exception:
            pass
        try:
            import imageio_ffmpeg  # noqa

            import imageio.v2 as imageio

            imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264", quality=8)
            return
        except Exception as e:
            raise RuntimeError(
                "No MP4 writer available (need imageio + ffmpeg). "
                f"Install imageio[ffmpeg]. Original error: {e}"
            ) from e

    def _report(self, request: GenerationRequest, step: int, total: int, msg: str = "") -> None:
        if request.progress_callback is not None:
            try:
                request.progress_callback(step, total, msg)
            except Exception:
                pass
