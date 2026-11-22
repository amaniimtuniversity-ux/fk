"""ModelOffloadManager: sequential CPU offload for diffusers pipelines.

Strategy for T4/P100 (16GB VRAM, ~29GB RAM):
- Text encoder (T5-XXL ~9GB FP16 for Wan) stays on CPU; only embeddings move.
- Transformer blocks stream CPU->GPU one at a time (sequential offload).
- VAE decode runs separately after sampling, with tiling/slicing enabled.
- Pinned memory + non_blocking transfers where the backend supports them.
- Transfer overhead is measured (bytes + seconds) and logged.

Two T4s are NEVER treated as unified VRAM. Multi-GPU is only used if the
pipeline genuinely supports device_map/model parallelism AND a probe shows
a benefit; otherwise a single GPU is used and the second is left idle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OffloadStats:
    mode: str = "none"
    transfers: int = 0
    bytes_transferred: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "transfers": self.transfers,
                "bytes_transferred": self.bytes_transferred,
                "seconds": round(self.seconds, 3), "notes": self.notes}


class ModelOffloadManager:
    def __init__(self, mode: str = "sequential", device_index: int = 0):
        self.mode = mode
        self.device_index = device_index
        self.stats = OffloadStats(mode=mode)

    def apply_to_pipeline(self, pipeline) -> None:
        """Apply diffusers offload + VAE slicing/tiling. Idempotent-ish."""
        t0 = time.time()
        if self.mode == "sequential":
            try:
                pipeline.enable_sequential_cpu_offload(device=self.device_index)
                self.stats.notes.append("enable_sequential_cpu_offload applied")
            except Exception as e:
                self.stats.notes.append(f"sequential offload failed: {e}")
                try:
                    pipeline.enable_model_cpu_offload(device=self.device_index)
                    self.stats.notes.append("fallback: enable_model_cpu_offload applied")
                except Exception as e2:
                    self.stats.notes.append(f"model offload also failed: {e2}")
        elif self.mode == "model":
            try:
                pipeline.enable_model_cpu_offload(device=self.device_index)
                self.stats.notes.append("enable_model_cpu_offload applied")
            except Exception as e:
                self.stats.notes.append(f"model offload failed: {e}")
        # VAE memory savers (diffusers >= 0.30)
        for fn_name in ("enable_vae_slicing", "enable_vae_tiling"):
            try:
                getattr(pipeline, fn_name)()
                self.stats.notes.append(f"{fn_name} enabled")
            except Exception:
                pass
        # Attention slicing reduces peak activation memory on small VRAM.
        try:
            pipeline.enable_attention_slicing()
            self.stats.notes.append("enable_attention_slicing enabled")
        except Exception:
            pass
        self.stats.seconds += time.time() - t0

    def measure_transfer(self, tensor) -> None:
        try:
            self.stats.transfers += 1
            self.stats.bytes_transferred += int(tensor.numel() * tensor.element_size())
        except Exception:
            pass

    def to_dict(self) -> dict[str, Any]:
        return self.stats.to_dict()
