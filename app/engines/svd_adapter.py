"""Stable Video Diffusion XT adapter — image-to-video fallback (lightest).

Requires a first-frame image. Used when the user provides an image or when
T2V engines OOM on P100. Still genuine motion generation (not Ken Burns).
"""
from __future__ import annotations

import time

from .base import GenerationRequest, GenerationResult, VideoGenerationEngine


class SVDAdapter(VideoGenerationEngine):
    engine_id = "svd"
    model_id = "svd-xt"
    supports_t2v = False
    supports_i2v = True

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from diffusers import StableVideoDiffusionPipeline

        self._pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.model_repo, torch_dtype=torch.float16, variant="fp16",
        )
        from app.offload import ModelOffloadManager

        ModelOffloadManager(mode=self.offload_mode).apply_to_pipeline(self._pipeline)
        self._loaded = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = time.time()
        self._reset_peak()
        try:
            if request.image_path is None:
                return GenerationResult(success=False, engine=self.engine_id, model=self.model_id,
                                        error="SVD requires image_path (first frame).")
            self.load()
            import torch
            from PIL import Image

            image = Image.open(request.image_path).convert("RGB")
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(request.seed)
            result = self._pipeline(
                image, decode_chunk_size=8, generator=generator,
                motion_bucket_id=180, noise_aug_strength=0.02,
            )
            video = result.frames[0]
            self._save_frames(list(video), request.output_path, request.fps)
            vram, ram = self._peak_memory()
            return GenerationResult(
                success=True, output_path=request.output_path, frames=len(video),
                width=video.shape[2], height=video.shape[1], fps=request.fps,
                seconds=time.time() - t0, peak_vram_bytes=vram, peak_ram_bytes=ram,
                engine=self.engine_id, model=self.model_id,
            )
        except Exception as e:
            import traceback

            return GenerationResult(
                success=False, seconds=time.time() - t0, engine=self.engine_id, model=self.model_id,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            )
