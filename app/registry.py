"""VideoModelRegistry + ModelCompatibilityChecker.

Engines are interchangeable. Selection priority (per requirements):
1. Genuine video generation  2. Visual quality  3. Temporal consistency
4. Kaggle compatibility  5. Memory efficiency  6. Speed

FreeVideo H3 is kept as an OPTIONAL experimental engine. It is NOT forced
into the production pipeline on T4/P100 because (see AUDIT.md):
- 99GB BF16 / 23GB FP8 download exceeds Kaggle disk/time budgets
- 33B-param transformer needs ~66GB RAM for CPU offload (Kaggle has ~29GB)
- Triton kernels + torch.compile require SM80+; P100 has no Triton 3.x
Production default on T4/P100: Wan2.1-T2V-1.3B (FP16, SDPA, sequential offload).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelSpec:
    id: str
    name: str
    hf_repo: str
    params_b: float
    download_gb: float
    vram_fp16_gb: float  # with offload + tiling, 480p
    ram_gb: float
    t2v: bool
    i2v: bool
    native_precision: str
    min_capability: tuple[int, int]
    license: str
    status: str  # "production" | "experimental" | "unsupported"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "hf_repo": self.hf_repo,
            "params_b": self.params_b, "download_gb": self.download_gb,
            "vram_fp16_gb": self.vram_fp16_gb, "ram_gb": self.ram_gb,
            "t2v": self.t2v, "i2v": self.i2v,
            "native_precision": self.native_precision,
            "min_capability": list(self.min_capability),
            "license": self.license, "status": self.status, "notes": self.notes,
        }


MODELS: dict[str, ModelSpec] = {
    "wan2.1-1.3b": ModelSpec(
        id="wan2.1-1.3b", name="Wan2.1 T2V 1.3B",
        hf_repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        params_b=1.3, download_gb=29.0, vram_fp16_gb=10.0, ram_gb=20.0,
        t2v=True, i2v=False, native_precision="bf16",
        min_capability=(6, 0), license="Apache-2.0",
        status="production",
        notes="Diffusers repo is 28.9GB on disk (UMT5-XXL text encoder stored fp32 = 22.7GB, transformer 5.7GB, "
              "VAE 0.5GB); loaded as FP16 (~15GB RAM) with sequential CPU offload. Official 8.19GB VRAM at 480p. "
              "Cache MUST live outside the 20GB /kaggle/working quota (use /kaggle/tmp). T2V only.",
    ),
    "cogvideox-2b": ModelSpec(
        id="cogvideox-2b", name="CogVideoX 2B",
        hf_repo="THUDM/CogVideoX-2b",
        params_b=2.0, download_gb=11.0, vram_fp16_gb=12.0, ram_gb=18.0,
        t2v=True, i2v=False, native_precision="fp16",
        min_capability=(6, 0), license="Apache-2.0",
        status="production",
        notes="Trained in FP16. 4GB min with diffusers opts; 12.5GB typical single-GPU. Good alternative.",
    ),
    "ltx-2b": ModelSpec(
        id="ltx-2b", name="LTX-Video 2B Distilled",
        hf_repo="Lightricks/LTX-Video",
        params_b=2.0, download_gb=8.0, vram_fp16_gb=12.0, ram_gb=18.0,
        t2v=True, i2v=True, native_precision="fp16",
        min_capability=(7, 5), license="Open RAIL-M",
        status="experimental",
        notes="Fast distilled model. Needs SM75+ in practice (Triton kernels). T4-only, not P100.",
    ),
    "svd-xt": ModelSpec(
        id="svd-xt", name="Stable Video Diffusion XT",
        hf_repo="stabilityai/stable-video-diffusion-xt",
        params_b=1.5, download_gb=4.5, vram_fp16_gb=8.0, ram_gb=14.0,
        t2v=False, i2v=True, native_precision="fp16",
        min_capability=(6, 0), license="Stability Community",
        status="production",
        notes="Image-to-video only. Lightest genuine-motion option; older quality. Good P100 fallback.",
    ),
    "freevideo-h3": ModelSpec(
        id="freevideo-h3", name="FreeVideo MiniMax H3 (VDN-H3)",
        hf_repo="OpenVDN/vdn-minimax-h3-edge",
        params_b=33.0, download_gb=23.0, vram_fp16_gb=99.0, ram_gb=66.0,
        t2v=True, i2v=True, native_precision="bf16/fp8",
        min_capability=(8, 0), license="MiniMax H3 Community",
        status="experimental",
        notes="EXPERIMENTAL on T4/P100: 99GB BF16 / 23GB FP8 download, ~66GB RAM offload need, SM80+ kernels. Kept optional; see AUDIT.md.",
    ),
    "wan2.2-5b": ModelSpec(
        id="wan2.2-5b", name="Wan2.2 TI2V 5B",
        hf_repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        params_b=5.0, download_gb=11.0, vram_fp16_gb=20.0, ram_gb=24.0,
        t2v=True, i2v=True, native_precision="fp16",
        min_capability=(7, 5), license="Apache-2.0",
        status="unsupported",
        notes="Needs ~20GB VRAM FP16. Exceeds single T4/P100 16GB even with offload at 720p; 480p borderline. Not default.",
    ),
    "hunyuan-13b": ModelSpec(
        id="hunyuan-13b", name="HunyuanVideo 13B",
        hf_repo="hunyuanvideo-community/HunyuanVideo",
        params_b=13.0, download_gb=26.0, vram_fp16_gb=31.0, ram_gb=32.0,
        t2v=True, i2v=False, native_precision="bf16",
        min_capability=(8, 0), license="Tencent Hunyuan Community",
        status="unsupported",
        notes="31-60GB VRAM. Cannot run on Kaggle T4/P100.",
    ),
}


class ModelCompatibilityChecker:
    def __init__(self, capability: tuple[int, int], vram_bytes: int, ram_bytes: int, disk_free_bytes: int,
                 cache_lookup=None, disk_path: str = ""):
        self.capability = capability
        self.vram_bytes = vram_bytes
        self.ram_bytes = ram_bytes
        self.disk_free_bytes = disk_free_bytes
        # callable(repo_id) -> bytes already in the HF cache; default = real filesystem walk
        from .kaggle_storage import cached_bytes as _cached

        self.cache_lookup = cache_lookup or _cached
        self.disk_path = disk_path

    def remaining_download_bytes(self, spec: ModelSpec) -> int:
        try:
            have = int(self.cache_lookup(spec.hf_repo))
        except Exception:
            have = 0
        return max(0, int(spec.download_gb * 1e9) - have)

    def check(self, spec: ModelSpec) -> dict[str, Any]:
        reasons: list[str] = []
        ok = True
        if self.capability < spec.min_capability:
            ok = False
            reasons.append(f"Needs SM{spec.min_capability[0]}{spec.min_capability[1]}+, have SM{self.capability[0]}{self.capability[1]}")
        if spec.vram_fp16_gb * 1e9 > self.vram_bytes * 1.15:  # 15% offload slack
            ok = False
            reasons.append(f"Needs ~{spec.vram_fp16_gb}GB VRAM (offloaded), have {self.vram_bytes/1e9:.1f}GB")
        if spec.ram_gb * 1e9 > self.ram_bytes:
            ok = False
            reasons.append(f"Needs ~{spec.ram_gb}GB RAM for offload, have {self.ram_bytes/1e9:.1f}GB")
        remaining = self.remaining_download_bytes(spec)
        if remaining > self.disk_free_bytes:
            ok = False
            where = f" at {self.disk_path}" if self.disk_path else ""
            reasons.append(f"Needs {remaining/1e9:.1f}GB more download (model {spec.download_gb}GB, "
                           f"{(spec.download_gb*1e9-remaining)/1e9:.1f}GB cached) but only "
                           f"{self.disk_free_bytes/1e9:.1f}GB free{where}")
        if spec.status == "unsupported":
            ok = False
            reasons.append("Marked unsupported for Kaggle T4/P100")
        return {"model": spec.id, "compatible": ok, "reasons": reasons, "status": spec.status}


class VideoModelRegistry:
    def __init__(self, checker: Optional[ModelCompatibilityChecker] = None):
        self.checker = checker
        self.models = dict(MODELS)

    def list_models(self) -> list[dict[str, Any]]:
        out = []
        for spec in self.models.values():
            d = spec.to_dict()
            if self.checker is not None:
                d["compatibility"] = self.checker.check(spec)
            out.append(d)
        return out

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self.models:
            raise KeyError(f"Unknown model: {model_id}. Available: {sorted(self.models)}")
        return self.models[model_id]

    def recommend(self, need_i2v: bool = False) -> ModelSpec:
        """Best production model for this hardware (quality-first)."""
        candidates = [m for m in self.models.values() if m.status == "production"]
        if need_i2v:
            candidates = [m for m in candidates if m.i2v] or candidates
        if self.checker is not None:
            compatible = [m for m in candidates if self.checker.check(m)["compatible"]]
            if compatible:
                candidates = compatible
        # Quality order: Wan 1.3B > CogVideoX 2B > SVD-XT
        order = {"wan2.1-1.3b": 0, "cogvideox-2b": 1, "svd-xt": 2}
        candidates.sort(key=lambda m: order.get(m.id, 99))
        return candidates[0]
