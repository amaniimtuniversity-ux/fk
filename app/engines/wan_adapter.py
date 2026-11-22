"""Wan2.1-T2V-1.3B adapter — PRIMARY production engine for Kaggle T4/P100.

- FP16 compute (native on T4/P100), FP32 sensitive layers via autocast policy
- SDPA attention (cudnn on T4, math on P100), chunked queries
- Sequential CPU offload + VAE tiling/slicing + attention slicing
- torch.compile DISABLED (Inductor/Triton needs SM80+; P100 has no Triton 3.x)
- Real text-to-video diffusion. No static-image fallback.
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import GenerationRequest, GenerationResult, VideoGenerationEngine


class WanAdapter(VideoGenerationEngine):
    engine_id = "wan2.1"
    model_id = "wan2.1-1.3b"
    supports_t2v = True
    supports_i2v = False

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from diffusers import WanPipeline

        dtype = torch.float16 if self.precision_plan.get("compute_dtype") == "float16" else torch.float32
        from app.engines.base import load_diffusers_pipeline

        self._pipeline = load_diffusers_pipeline(
            WanPipeline, self.model_repo, dtype,
            prefer_variant="fp16" if dtype == torch.float16 else None,
        )
        from app.offload import ModelOffloadManager

        ModelOffloadManager(mode=self.offload_mode).apply_to_pipeline(self._pipeline)
        # Explicitly disable torch.compile (SM80+ only).
        try:
            self._pipeline.transformer = torch.compile(self._pipeline.transformer, disable=True)
        except Exception:
            pass
        self._loaded = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = time.time()
        self._reset_peak()
        warnings: list[str] = []
        try:
            self.load()
            import torch

            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(request.seed)
            # Wan requires frames % 4 == 1 (temporal VAE compression).
            frames = request.num_frames
            if frames % 4 != 1:
                frames = frames - (frames % 4) + 1
                warnings.append(f"Frames adjusted to {frames} (Wan requires 4n+1).")
            self._report(request, 0, request.num_inference_steps, "sampling")
            result = self._pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or "blurry, low quality, distorted, watermark",
                height=request.height, width=request.width,
                num_frames=frames, num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale, generator=generator,
            )
            video = result.frames[0]  # [F,H,W,3] uint8 numpy
            self._report(request, request.num_inference_steps, request.num_inference_steps, "encoding")
            self._save_frames(list(video), request.output_path, request.fps)
            vram, ram = self._peak_memory()
            return GenerationResult(
                success=True, output_path=request.output_path, frames=len(video),
                width=request.width, height=request.height, fps=request.fps,
                seconds=time.time() - t0, peak_vram_bytes=vram, peak_ram_bytes=ram,
                engine=self.engine_id, model=self.model_id, warnings=warnings,
            )
        except Exception as e:
            import traceback

            return GenerationResult(
                success=False, seconds=time.time() - t0, engine=self.engine_id, model=self.model_id,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}", warnings=warnings,
            )
