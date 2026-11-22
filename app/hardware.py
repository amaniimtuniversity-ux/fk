"""HardwareDetector: GPU, compute capability, VRAM, RAM, disk, CUDA/PyTorch.

Never assumes T4x2 = unified VRAM. Each device is reported separately.
Works without torch (CPU-only sandbox) by degrading gracefully.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class GPUInfo:
    index: int
    name: str
    capability: tuple[int, int]
    vram_total_bytes: int
    vram_free_bytes: int
    uuid: str = ""


@dataclass
class HardwareReport:
    gpus: list[GPUInfo] = field(default_factory=list)
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    cpu_count: int = 0
    disk_free_bytes: int = 0
    disk_total_bytes: int = 0
    torch_version: str = ""
    cuda_version: str = ""
    cuda_available: bool = False
    platform: str = ""
    is_kaggle: bool = False
    kaggle_working_exists: bool = False
    internet_available: bool = False

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        return self.gpus[0] if self.gpus else None

    @property
    def capability(self) -> tuple[int, int]:
        g = self.primary_gpu
        return g.capability if g else (0, 0)

    @property
    def gpu_name(self) -> str:
        g = self.primary_gpu
        return g.name if g else "CPU-only"

    @property
    def vram_total_bytes(self) -> int:
        g = self.primary_gpu
        return g.vram_total_bytes if g else 0

    @property
    def architecture(self) -> str:
        cap = self.capability
        if cap == (7, 5):
            return "turing-t4"
        if cap == (6, 0):
            return "pascal-p100"
        if cap == (7, 0):
            return "volta"
        if cap == (7, 2):
            return "volta"
        if cap == (7, 5):
            return "turing"
        if (8, 0) <= cap < (8, 9):
            return "ampere"
        if cap == (8, 9):
            return "ada"
        if cap[0] == 9:
            return "hopper"
        if cap == (0, 0):
            return "cpu-only"
        return f"sm{cap[0]}{cap[1]}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["architecture"] = self.architecture
        d["gpu_name"] = self.gpu_name
        d["capability"] = list(self.capability)
        d["vram_total_bytes"] = self.vram_total_bytes
        return d


def _ram_bytes() -> tuple[int, int]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1]) * 1024
                return info.get("MemTotal", 0), info.get("MemAvailable", info.get("MemFree", 0))
        except Exception:
            return 0, 0


def _disk_bytes(path: str = "/") -> tuple[int, int]:
    try:
        usage = shutil.disk_usage(path)
        return int(usage.total), int(usage.free)
    except Exception:
        return 0, 0


def _nvidia_smi() -> list[GPUInfo]:
    """Parse nvidia-smi if available (fallback when torch is absent)."""
    gpus: list[GPUInfo] = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return gpus
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            idx, name, uuid, total_mib, free_mib = parts[:5]
            gpus.append(GPUInfo(
                index=int(idx), name=name, capability=(0, 0),
                vram_total_bytes=int(total_mib) * 1024 * 1024,
                vram_free_bytes=int(free_mib) * 1024 * 1024,
                uuid=uuid,
            ))
    except Exception:
        pass
    return gpus


def detect_hardware() -> HardwareReport:
    report = HardwareReport(
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        cpu_count=os.cpu_count() or 0,
        is_kaggle=Path("/kaggle").exists(),
        kaggle_working_exists=Path("/kaggle/working").exists(),
    )
    report.ram_total_bytes, report.ram_available_bytes = _ram_bytes()
    report.disk_total_bytes, report.disk_free_bytes = _disk_bytes("/kaggle/working" if report.kaggle_working_exists else "/")
    try:
        import socket

        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        report.internet_available = True
    except Exception:
        report.internet_available = False

    try:
        import torch

        report.torch_version = torch.__version__
        report.cuda_version = torch.version.cuda or ""
        report.cuda_available = bool(torch.cuda.is_available())
        if report.cuda_available:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                report.gpus.append(GPUInfo(
                    index=i,
                    name=props.name,
                    capability=(int(props.major), int(props.minor)),
                    vram_total_bytes=int(total),
                    vram_free_bytes=int(free),
                    uuid=getattr(props, "uuid", "") or "",
                ))
    except Exception:
        pass

    if not report.gpus:
        report.gpus = _nvidia_smi()
    return report


class HardwareDetector:
    """Stateful detector with caching + JSON persistence for session recovery."""

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path
        self._report: Optional[HardwareReport] = None

    def detect(self, force: bool = False) -> HardwareReport:
        if self._report is None or force:
            self._report = detect_hardware()
            if self.cache_path is not None:
                try:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self.cache_path.write_text(json.dumps(self._report.to_dict(), indent=2))
                except Exception:
                    pass
        return self._report

    def summary(self) -> str:
        r = self.detect()
        lines = [
            f"GPU: {r.gpu_name} SM{r.capability[0]}{r.capability[1]} ({r.architecture})",
            f"VRAM: {r.vram_total_bytes / 1e9:.1f} GB total, {len(r.gpus)} device(s) (separate, not unified)",
            f"RAM: {r.ram_total_bytes / 1e9:.1f} GB total, {r.ram_available_bytes / 1e9:.1f} GB available",
            f"Disk free: {r.disk_free_bytes / 1e9:.1f} GB",
            f"PyTorch: {r.torch_version or 'not installed'} CUDA: {r.cuda_version or 'n/a'}",
            f"Kaggle: {r.is_kaggle} working={r.kaggle_working_exists} internet={r.internet_available}",
        ]
        return "\n".join(lines)
