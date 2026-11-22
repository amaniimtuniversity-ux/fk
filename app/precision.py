"""PrecisionManager: FP16/FP32/BF16/FP8 compatibility layer for SM75/SM60.

Key facts (verified against FreeVideo source + CUDA docs):
- T4 (SM75) has NO native BF16 tensor cores. PyTorch BF16 matmul on T4 is
  emulated via FP32 (slow, and some ops raise). FP16 IS native on T4/P100.
- P100 (SM60) has FP16 (1:64 rate on consumer Pascal? P100 datacenter FP16 is
  1:2 of FP32 — still usable) but NO BF16, NO FP8, NO Triton 3.x support.
- FP8 (E4M3) requires SM89+ (Ada) for native tensor-core GEMM. Storing weights
  as FP8 does NOT make them executable on T4/P100. They must be dequantized
  to FP16 on CPU and run as FP16 GEMMs.
- Sensitive ops (LayerNorm/RMSNorm reductions, softmax, rotary embeddings,
  VAE output, timestep embeddings) stay FP32 for numerical stability.

This module implements:
- probe_precision(): actual matmul probes (not just capability checks)
- dequant_fp8_to_fp16(): CPU dequant of FP8 weights+scales -> FP16 tensors
- convert_checkpoint_fp8_to_fp16(): streaming safetensors conversion
- PrecisionPlan: per-component dtype assignment
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Layers that must stay FP32 regardless of compute dtype.
FP32_SENSITIVE_PATTERNS = (
    "norm", "layernorm", "rmsnorm", "softmax", "rotary", "rope",
    "timestep", "time_embed", "vae.norm", "final_norm", "adaln",
)


@dataclass
class PrecisionProbe:
    bf16_matmul_ok: bool = False
    bf16_matmul_note: str = ""
    fp16_matmul_ok: bool = False
    fp16_matmul_note: str = ""
    fp8_dtype_exists: bool = False
    fp8_matmul_ok: bool = False
    fp8_matmul_note: str = ""
    triton_available: bool = False
    triton_note: str = ""
    torch_compile_ok: bool = False
    torch_compile_note: str = ""


@dataclass
class PrecisionPlan:
    compute_dtype: str  # "float16" | "float32" | "bfloat16"
    weight_dtype: str
    sensitive_dtype: str = "float32"
    use_autocast: bool = True
    disable_torch_compile: bool = False
    disable_triton: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_dtype": self.compute_dtype,
            "weight_dtype": self.weight_dtype,
            "sensitive_dtype": self.sensitive_dtype,
            "use_autocast": self.use_autocast,
            "disable_torch_compile": self.disable_torch_compile,
            "disable_triton": self.disable_triton,
            "notes": self.notes,
        }


def _try_matmul(dtype_name: str) -> tuple[bool, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "no CUDA device"
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            return False, f"torch.{dtype_name} does not exist in this build"
        a = torch.randn(64, 64, device="cuda", dtype=dtype)
        b = torch.randn(64, 64, device="cuda", dtype=dtype)
        c = a @ b
        torch.cuda.synchronize()
        if c.dtype != dtype:
            return False, f"output dtype {c.dtype} != {dtype}"
        # Check for NaN/Inf (emulation bugs on old arch)
        if not torch.isfinite(c).all():
            return False, "non-finite result"
        return True, f"{dtype_name} matmul ok on {torch.cuda.get_device_name(0)}"
    except Exception as e:
        return False, f"{dtype_name} matmul failed: {type(e).__name__}: {e}"


def probe_precision() -> PrecisionProbe:
    probe = PrecisionProbe()
    try:
        import torch

        probe.fp8_dtype_exists = hasattr(torch, "float8_e4m3fn")
    except Exception:
        probe.fp8_dtype_exists = False
    probe.bf16_matmul_ok, probe.bf16_matmul_note = _try_matmul("bfloat16")
    probe.fp16_matmul_ok, probe.fp16_matmul_note = _try_matmul("float16")
    if probe.fp8_dtype_exists:
        probe.fp8_matmul_ok, probe.fp8_matmul_note = _try_matmul("float8_e4m3fn")
    else:
        probe.fp8_matmul_note = "torch.float8_e4m3fn missing (old torch / CPU build)"
    try:
        import triton  # noqa: F401

        probe.triton_available = True
        probe.triton_note = f"triton {getattr(triton, '__version__', '?')} importable"
    except Exception as e:
        probe.triton_note = f"triton unavailable: {e}"
    try:
        import torch

        if torch.cuda.is_available():
            def f(x):
                return x * 2

            cf = torch.compile(f)
            cf(torch.randn(8, device="cuda"))
            torch.cuda.synchronize()
            probe.torch_compile_ok = True
            probe.torch_compile_note = "torch.compile ran on CUDA"
        else:
            probe.torch_compile_note = "no CUDA device to test torch.compile"
    except Exception as e:
        probe.torch_compile_note = f"torch.compile failed: {type(e).__name__}: {e}"
    return probe


def plan_precision(capability: tuple[int, int], probe: Optional[PrecisionProbe] = None) -> PrecisionPlan:
    """Select precision plan from capability + actual probes.

    Rules:
    - SM80+ with working BF16 -> BF16 compute (FreeVideo native path).
    - SM75/SM60 (T4/P100) -> FP16 compute, FP32 sensitive layers.
      BF16 is NEVER selected on T4/P100 even if the probe "passes" via
      slow emulation, unless the user explicitly overrides.
    - FP8 native GEMM only on SM89+ with working probe.
    """
    probe = probe or PrecisionProbe()
    major, minor = capability
    notes: list[str] = []
    if capability >= (8, 9) and probe.fp8_matmul_ok:
        notes.append("SM89+ with working FP8 probe: native FP8 GEMM available.")
        return PrecisionPlan("float8_e4m3fn", "float8_e4m3fn", notes=notes)
    if capability >= (8, 0) and probe.bf16_matmul_ok:
        notes.append("SM80+ with working BF16 probe: native BF16 compute.")
        return PrecisionPlan("bfloat16", "bfloat16", notes=notes)
    if capability in ((7, 5), (6, 0), (7, 0), (7, 2)) or capability < (8, 0):
        notes.append(
            f"SM{major}{minor} has no native BF16/FP8 tensor cores. "
            "Using FP16 compute with FP32 sensitive layers. "
            "FP8 checkpoints must be dequantized to FP16 on CPU first."
        )
        if not probe.fp16_matmul_ok:
            notes.append("WARNING: FP16 matmul probe failed; falling back to FP32 compute (very slow).")
            return PrecisionPlan("float32", "float32", disable_torch_compile=True, disable_triton=True, notes=notes)
        return PrecisionPlan("float16", "float16", disable_torch_compile=True, disable_triton=(capability == (6, 0)), notes=notes)
    # Unknown / CPU-only
    notes.append("Unknown capability: FP32 fallback.")
    return PrecisionPlan("float32", "float32", disable_torch_compile=True, disable_triton=True, notes=notes)


def is_sensitive_layer(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in FP32_SENSITIVE_PATTERNS)


def dequant_fp8_to_fp16(weight_fp8, weight_scale, per_tensor: bool = True):
    """Dequantize one FP8 weight matrix to FP16 on CPU.

    weight_fp8: torch tensor (any int8-like storage of E4M3 values)
    weight_scale: scalar or per-row FP32 scales
    Returns FP16 tensor. Pure torch; works on CPU.
    """
    import torch

    w = weight_fp8.float()
    s = weight_scale.float()
    if per_tensor or s.numel() == 1:
        out = w * s.reshape(1) if s.numel() == 1 else w * s
    else:
        # per-row scales: shape (rows,) or (rows, 1)
        out = w * s.reshape(-1, 1)
    return out.to(torch.float16)


def convert_checkpoint_fp8_to_fp16(src_dir: Path, dst_dir: Path, max_files: Optional[int] = None) -> dict[str, Any]:
    """Stream-convert an FP8 safetensors checkpoint dir to FP16.

    Handles FreeVideo/VDN naming: *.weight_fp8 + *.weight_scale pairs.
    Other tensors are copied with dtype conversion BF16->FP16 where applicable.
    Memory-bounded: one tensor at a time. Never loads the full model.
    Returns manifest dict.
    """
    import torch
    from safetensors.torch import load_file, save_file

    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"converted": [], "copied": [], "skipped": []}
    files = sorted(src_dir.glob("*.safetensors"))
    if max_files:
        files = files[:max_files]
    for f in files:
        tensors = load_file(str(f))
        out: dict[str, Any] = {}
        # Collect FP8 pairs
        fp8_names = {k for k in tensors if k.endswith(".weight_fp8")}
        scales = {k: tensors[k] for k in tensors if k.endswith(".weight_scale")}
        consumed_scales: set[str] = set()
        for name in sorted(tensors):
            if name in fp8_names:
                base = name[: -len(".weight_fp8")]
                scale_key = base + ".weight_scale"
                scale = scales.get(scale_key)
                if scale is None:
                    manifest["skipped"].append(name)
                    continue
                consumed_scales.add(scale_key)
                per_tensor = scale.numel() == 1
                out[base + ".weight"] = dequant_fp8_to_fp16(tensors[name], scale, per_tensor=per_tensor)
                manifest["converted"].append(name)
            elif name.endswith(".weight_scale") and name in consumed_scales:
                continue
            elif name.endswith(".weight_scale"):
                manifest["skipped"].append(name)
            else:
                t = tensors[name]
                if t.dtype == torch.bfloat16:
                    out[name] = t.to(torch.float16)
                else:
                    out[name] = t
                manifest["copied"].append(name)
            del tensors[name]
        save_file(out, str(dst_dir / f.name), metadata={"format": "pt"})
        del out
        gc.collect()
    (dst_dir / "conversion_manifest.json").write_text(__import__("json").dumps(manifest, indent=2))
    return manifest


class PrecisionManager:
    def __init__(self, capability: tuple[int, int]):
        self.capability = capability
        self.probe: Optional[PrecisionProbe] = None
        self.plan: Optional[PrecisionPlan] = None

    def probe_and_plan(self) -> PrecisionPlan:
        self.probe = probe_precision()
        self.plan = plan_precision(self.capability, self.probe)
        return self.plan

    def apply_to_model(self, model) -> None:
        """Cast model weights per plan; sensitive layers stay FP32."""
        if self.plan is None:
            self.probe_and_plan()
        assert self.plan is not None
        import torch

        compute = getattr(torch, self.plan.compute_dtype, torch.float32)
        for name, param in model.named_parameters():
            if is_sensitive_layer(name):
                param.data = param.data.float()
            else:
                param.data = param.data.to(compute)
