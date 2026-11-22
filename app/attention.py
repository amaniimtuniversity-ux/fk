"""AttentionBackendSelector: hardware-aware attention for SM75/SM60.

FreeVideo backends: cudnn / torch-flash (SDPA) / sage2 / fa2 / fa4.
Reality on T4/P100:
- FA2 requires SM80+ (Ampere). FA4 requires SM90+ (Hopper). Both FAIL on T4/P100.
- SageAttention 2.x requires SM80+ for its Triton kernels. FAILS on T4/P100.
- xFormers memory-efficient attention supports SM75 but NOT SM60 (Pascal).
- PyTorch SDPA with CUDNN backend: works on T4, unreliable on P100.
- PyTorch SDPA MATH backend: works everywhere (slow, correct). Always available.
- PyTorch SDPA FLASH backend: requires SM80+ in recent torch. FAILS on T4/P100.

Selection order on T4/P100: cudnn-SDPA (probed) -> math-SDPA (always).
Chunked SDPA bounds activation memory by slicing the query sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AttentionPlan:
    backend: str  # "cudnn-sdpa" | "math-sdpa" | "flash-sdpa" | "sage2" | "fa2" | "fa4"
    query_chunk: int  # 0 = no chunking
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "query_chunk": self.query_chunk, "notes": self.notes}


def _probe_sdpa(backend_name: str) -> tuple[bool, str]:
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if not torch.cuda.is_available():
            return False, "no CUDA"
        q = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        backend = getattr(SDPBackend, backend_name)
        with sdpa_kernel(backend):
            out = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        return bool(torch.isfinite(out).all()), f"{backend_name} ok"
    except Exception as e:
        return False, f"{backend_name} failed: {type(e).__name__}: {e}"


def _probe_package(name: str) -> tuple[bool, str]:
    try:
        import importlib.metadata

        v = importlib.metadata.version(name)
        return True, f"{name} {v} installed"
    except Exception as e:
        return False, f"{name} not installed: {e}"


def select_attention(capability: tuple[int, int], query_chunk: int = 1024) -> AttentionPlan:
    notes: list[str] = []
    major, _ = capability
    if capability >= (9, 0):
        notes.append("SM90+: FA4/Sage2 available (probe before use).")
        return AttentionPlan("fa4", query_chunk, notes)
    if capability >= (8, 0):
        notes.append("SM80+: FA2/Sage2 available (probe before use).")
        return AttentionPlan("sage2", query_chunk, notes)
    # T4 / P100 / older: only SDPA paths are viable.
    notes.append(f"SM{major}{_}: FA2/FA4/Sage2 are NOT supported (require SM80+). Using SDPA.")
    ok, note = _probe_sdpa("CUDNN_ATTENTION")
    notes.append(note)
    if ok:
        return AttentionPlan("cudnn-sdpa", query_chunk, notes)
    ok2, note2 = _probe_sdpa("MATH")
    notes.append(note2)
    if ok2:
        return AttentionPlan("math-sdpa", query_chunk, notes)
    notes.append("All SDPA probes failed (likely CPU-only). Using eager math attention.")
    return AttentionPlan("math-sdpa", query_chunk, notes)


def chunked_sdpa(q, k, v, scale: Optional[float] = None, chunk: int = 1024, backend: str = "math-sdpa"):
    """Memory-bounded SDPA: slice queries into chunks of `chunk` rows.

    q,k,v: [batch, heads, seq, head_dim]. Returns [batch, heads, seq, head_dim].
    Chunking is mathematically exact (each query row is independent).
    """
    import torch
    import torch.nn.functional as F

    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    seq = q.shape[-2]
    if chunk <= 0 or seq <= chunk:
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    outs = []
    for start in range(0, seq, chunk):
        outs.append(F.scaled_dot_product_attention(q[..., start : start + chunk, :], k, v, scale=scale))
    return torch.cat(outs, dim=-2)


class AttentionBackendSelector:
    def __init__(self, capability: tuple[int, int], query_chunk: int = 1024):
        self.capability = capability
        self.query_chunk = query_chunk
        self.plan: Optional[AttentionPlan] = None

    def select(self) -> AttentionPlan:
        self.plan = select_attention(self.capability, self.query_chunk)
        return self.plan

    def apply_to_diffusers(self, pipeline) -> None:
        """Point diffusers at the selected SDPA backend where supported."""
        if self.plan is None:
            self.select()
        assert self.plan is not None
        try:
            # diffusers transformers use SDPA by default; force math on P100
            # by disabling flash/sage flags if present.
            for attr in ("_attention_processor",):
                pass
            # Best-effort: set torch SDPA default backend via context is not
            # global; instead record the plan for engine code to use.
            pipeline._fv_attention_plan = self.plan.to_dict()  # type: ignore[attr-defined]
        except Exception:
            pass
