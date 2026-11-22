"""Kernel probe: BF16/FP16/FP8 matmul, SDPA backends, Triton, torch.compile.

Run on Kaggle BEFORE any model download. Saves JSON report.
Usage: python scripts/probe_kernels.py [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.attention import _probe_sdpa
from app.hardware import HardwareDetector
from app.precision import probe_precision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kernel_probe.json")
    args = ap.parse_args()
    hw = HardwareDetector().detect()
    print(hw.summary() if hasattr(hw, "summary") else hw.to_dict())
    probe = probe_precision()
    sdpa = {}
    for name in ("CUDNN_ATTENTION", "FLASH_ATTENTION", "MATH"):
        ok, note = _probe_sdpa(name)
        sdpa[name] = {"ok": ok, "note": note}
        print(f"SDPA {name}: ok={ok} {note}")
    report = {
        "hardware": hw.to_dict(),
        "precision": {
            "bf16": {"ok": probe.bf16_matmul_ok, "note": probe.bf16_matmul_note},
            "fp16": {"ok": probe.fp16_matmul_ok, "note": probe.fp16_matmul_note},
            "fp8": {"ok": probe.fp8_matmul_ok, "note": probe.fp8_matmul_note},
            "triton": {"ok": probe.triton_available, "note": probe.triton_note},
            "torch_compile": {"ok": probe.torch_compile_ok, "note": probe.torch_compile_note},
        },
        "sdpa": sdpa,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
