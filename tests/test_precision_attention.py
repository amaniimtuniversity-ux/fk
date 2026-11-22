"""Tests: precision planning, FP8 dequant math, attention selection, chunked SDPA."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.attention import AttentionBackendSelector, chunked_sdpa, select_attention
from app.precision import PrecisionManager, dequant_fp8_to_fp16, is_sensitive_layer, plan_precision, PrecisionProbe


def test_t4_gets_fp16_not_bf16():
    probe = PrecisionProbe(fp16_matmul_ok=True, bf16_matmul_ok=True)  # even if BF16 "works" via emulation
    plan = plan_precision((7, 5), probe)
    assert plan.compute_dtype == "float16"
    assert plan.disable_torch_compile is True


def test_p100_gets_fp16_no_triton():
    probe = PrecisionProbe(fp16_matmul_ok=True)
    plan = plan_precision((6, 0), probe)
    assert plan.compute_dtype == "float16"
    assert plan.disable_triton is True


def test_sm80_gets_bf16():
    probe = PrecisionProbe(bf16_matmul_ok=True)
    plan = plan_precision((8, 0), probe)
    assert plan.compute_dtype == "bfloat16"


def test_fp16_failure_falls_back_fp32():
    probe = PrecisionProbe(fp16_matmul_ok=False)
    plan = plan_precision((7, 5), probe)
    assert plan.compute_dtype == "float32"


def test_sensitive_layers():
    assert is_sensitive_layer("transformer_blocks.0.norm1.weight") is True
    assert is_sensitive_layer("transformer_blocks.0.attn.q_proj.weight") is False


def test_dequant_per_tensor():
    torch = pytest.importorskip("torch")
    w = torch.ones(4, 8, dtype=torch.float32)
    s = torch.tensor(2.0)
    out = dequant_fp8_to_fp16(w, s, per_tensor=True)
    assert out.dtype == torch.float16
    assert float(out[0, 0]) == pytest.approx(2.0, abs=0.01)


def test_dequant_per_row():
    torch = pytest.importorskip("torch")
    w = torch.ones(4, 8, dtype=torch.float32)
    s = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = dequant_fp8_to_fp16(w, s, per_tensor=False)
    assert float(out[2, 0]) == pytest.approx(3.0, abs=0.01)


def test_attention_t4_uses_sdpa():
    plan = select_attention((7, 5))
    assert plan.backend in ("cudnn-sdpa", "math-sdpa")
    assert "fa2" not in plan.backend and "sage" not in plan.backend


def test_chunked_sdpa_matches_full():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    torch.manual_seed(0)
    q = torch.randn(1, 2, 64, 16)
    k = torch.randn(1, 2, 64, 16)
    v = torch.randn(1, 2, 64, 16)
    full = F.scaled_dot_product_attention(q, k, v)
    chunked = chunked_sdpa(q, k, v, chunk=16)
    assert torch.allclose(full, chunked, atol=1e-5)


def test_precision_manager_init():
    mgr = PrecisionManager((7, 5))
    assert mgr.capability == (7, 5)


class _UnpicklableUuid:
    """Mimics torch._C._CUuuid (torch>=2.5): not picklable, only str()-able."""

    def __reduce_ex__(self, protocol):
        raise TypeError("cannot pickle '_CUuuid' object")

    def __str__(self):
        return "GPU-deadbeef"


def test_hardware_report_to_dict_survives_unpicklable_torch_objects():
    import json
    from app.hardware import GPUInfo, HardwareReport

    report = HardwareReport(
        gpus=[GPUInfo(0, "Tesla T4", (7, 5), 15 * 2**30, 14 * 2**30, uuid=_UnpicklableUuid())],
        cuda_available=True,
    )
    d = report.to_dict()
    json.dumps(d)  # must be JSON-serialisable end to end
    assert d["gpus"][0]["uuid"] == "GPU-deadbeef"
    assert d["capability"] == [7, 5]
    assert d["gpu_name"] == "Tesla T4"
