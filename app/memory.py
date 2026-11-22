"""MemoryManager: VRAM/RAM monitoring, GC boundaries, adaptive sizing.

Adaptive sizing NEVER silently reduces quality: every reduction emits a
warning that is surfaced in the UI and logged to the project manifest.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MemorySnapshot:
    timestamp: float
    vram_allocated_bytes: int = 0
    vram_reserved_bytes: int = 0
    vram_peak_bytes: int = 0
    ram_used_bytes: int = 0
    ram_available_bytes: int = 0
    label: str = ""


class MemoryManager:
    def __init__(self, vram_budget_bytes: Optional[int] = None):
        self.vram_budget_bytes = vram_budget_bytes
        self.snapshots: list[MemorySnapshot] = []
        self.warnings: list[str] = []

    def snapshot(self, label: str = "") -> MemorySnapshot:
        snap = MemorySnapshot(timestamp=time.time(), label=label)
        try:
            import torch

            if torch.cuda.is_available():
                snap.vram_allocated_bytes = int(torch.cuda.memory_allocated())
                snap.vram_reserved_bytes = int(torch.cuda.memory_reserved())
                snap.vram_peak_bytes = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        try:
            import psutil

            vm = psutil.virtual_memory()
            snap.ram_used_bytes = int(vm.used)
            snap.ram_available_bytes = int(vm.available)
        except Exception:
            pass
        self.snapshots.append(snap)
        return snap

    def cleanup(self, label: str = "") -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if label:
            self.snapshot(label)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def adapt_canvas(self, width: int, height: int, frames: int, vram_free_bytes: int) -> tuple[int, int, int, list[str]]:
        """Return (possibly reduced) canvas. Reductions are always reported.

        Heuristic: Wan2.1-1.3B FP16 needs ~9-13GB at 832x480x49 with offload.
        Below 10GB free, step down one tier at a time.
        """
        notes: list[str] = []
        free_gb = vram_free_bytes / 1e9
        if free_gb >= 10:
            return width, height, frames, notes
        tiers = [(832, 480, 49), (704, 400, 41), (640, 360, 33), (512, 288, 25)]
        for w, h, f in tiers:
            if w < width or h < height or f < frames:
                # First strictly-smaller tier: this is the reduction step.
                notes.append(
                    f"VRAM free {free_gb:.1f}GB < 10GB: canvas reduced "
                    f"{width}x{height}x{frames} -> {w}x{h}x{f}. "
                    "This REDUCES output resolution/frames. Increase VRAM or accept the reduction."
                )
                self.warn(notes[-1])
                return w, h, f, notes
            if (w, h, f) == (width, height, frames):
                continue  # requested tier itself is too big for free VRAM; step down
        notes.append(f"VRAM free {free_gb:.1f}GB critically low; keeping requested canvas (may OOM).")
        return width, height, frames, notes

    def peak_vram_bytes(self) -> int:
        return max((s.vram_peak_bytes for s in self.snapshots), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshots": [
                {"label": s.label, "vram_allocated": s.vram_allocated_bytes,
                 "vram_peak": s.vram_peak_bytes, "ram_used": s.ram_used_bytes}
                for s in self.snapshots
            ],
            "warnings": self.warnings,
            "peak_vram_bytes": self.peak_vram_bytes(),
        }
