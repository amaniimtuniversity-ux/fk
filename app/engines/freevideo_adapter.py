"""FreeVideo H3 adapter — EXPERIMENTAL on T4/P100. Optional engine only.

This adapter wraps the upstream FreeVideo engine (MiniMax H3 / VDN-H3) with
the T4/P100 compatibility patches (see patches/). It attempts:
- FP8 checkpoint -> FP16 dequant on CPU (streaming, memory-bounded)
- BF16 -> FP16 weight conversion
- SDPA attention fallback (cudnn/math) instead of FA2/FA4/Sage2
- torch.compile + Triton disabled on SM60; guarded on SM75
- Sequential layer offload (FreeVideo's own offload.py)

Honest status: on Kaggle T4/P100 this is EXPECTED to fail or be impractically
slow because the model needs ~66GB RAM for offload (Kaggle has ~29GB) and
~23GB download (Kaggle working dir is 20GB). The adapter exists so the
integration is preserved and testable on SM80+ hardware; it is never the
default on T4/P100. See AUDIT.md for the full root-cause analysis.
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import GenerationRequest, GenerationResult, VideoGenerationEngine


class FreeVideoAdapter(VideoGenerationEngine):
    engine_id = "freevideo-h3"
    model_id = "freevideo-h3"
    supports_t2v = True
    supports_i2v = True

    def __init__(self, *args, freevideo_repo: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.freevideo_repo = freevideo_repo

    def load(self) -> None:
        if self._loaded:
            return
        # Verify upstream repo + patches present.
        repo = self.freevideo_repo
        if repo is None or not (repo / "freevideo_engine" / "hardware.py").exists():
            raise RuntimeError(
                "FreeVideo repo not found. Clone https://github.com/FlashML-org/FreeVideo.git "
                "and set FREEVVIDEO_REPO, then apply patches/freevideo_t4_p100.patch."
            )
        # The actual generation path delegates to the upstream CLI/engine
        # subprocess (isolated process, same as upstream design). Importing
        # freevideo_engine in-process would pull ComfyUI/VDN deps; the
        # subprocess boundary is intentional and matches upstream.
        self._loaded = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = time.time()
        try:
            self.load()
            import subprocess
            import sys

            repo = self.freevideo_repo
            assert repo is not None
            cmd = [
                sys.executable, "-m", "freevideo_engine.cli", "generate",
                "--prompt", request.prompt,
                "--output", str(request.output_path),
                "--width", str(request.width), "--height", str(request.height),
                "--frames", str(request.num_frames),
                "--seed", str(request.seed),
            ]
            env = {"FREEVVIDEO_T4_P100": "1", "PYTHONPATH": str(repo)}
            proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=6 * 3600)
            if proc.returncode != 0 or not request.output_path.exists():
                return GenerationResult(
                    success=False, seconds=time.time() - t0, engine=self.engine_id, model=self.model_id,
                    error=f"FreeVideo subprocess failed (rc={proc.returncode}): {proc.stderr[-2000:]}",
                )
            vram, ram = self._peak_memory()
            return GenerationResult(
                success=True, output_path=request.output_path, frames=request.num_frames,
                width=request.width, height=request.height, fps=request.fps,
                seconds=time.time() - t0, peak_vram_bytes=vram, peak_ram_bytes=ram,
                engine=self.engine_id, model=self.model_id,
                warnings=["EXPERIMENTAL engine on T4/P100. Verify output quality manually."],
            )
        except Exception as e:
            import traceback

            return GenerationResult(
                success=False, seconds=time.time() - t0, engine=self.engine_id, model=self.model_id,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            )
