"""CogVideoX-2B adapter — secondary production engine (FP16-native model)."""
from __future__ import annotations

import time

from .base import GenerationRequest, GenerationResult, VideoGenerationEngine


class CogVideoXAdapter(VideoGenerationEngine):
    engine_id = "cogvideox"
    model_id = "cogvideox-2b"
    supports_t2v = True
    supports_i2v = False

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from diffusers import CogVideoXPipeline

        self._pipeline = CogVideoXPipeline.from_pretrained(self.model_repo, torch_dtype=torch.float16)
        from app.offload import ModelOffloadManager

        ModelOffloadManager(mode=self.offload_mode).apply_to_pipeline(self._pipeline)
        self._loaded = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = time.time()
        self._reset_peak()
        try:
            self.load()
            import torch

            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(request.seed)
            frames = request.num_frames
            if frames % 8 != 1:
                frames = frames - (frames % 8) + 1
            result = self._pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or "blurry, low quality",
                width=request.width, height=request.height, num_frames=frames,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale, generator=generator,
            )
            video = result.frames[0]
            self._save_frames(list(video), request.output_path, request.fps)
            vram, ram = self._peak_memory()
            return GenerationResult(
                success=True, output_path=request.output_path, frames=len(video),
                width=request.width, height=request.height, fps=request.fps,
                seconds=time.time() - t0, peak_vram_bytes=vram, peak_ram_bytes=ram,
                engine=self.engine_id, model=self.model_id,
            )
        except Exception as e:
            import traceback

            return GenerationResult(
                success=False, seconds=time.time() - t0, engine=self.engine_id, model=self.model_id,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            )
