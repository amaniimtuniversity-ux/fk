"""Fast runtime sanity check with actionable messages (run at server start and before a job).

Catches the environment failures seen on Kaggle *before* a 20-minute job dies on them:
- protobuf downgraded by a dependency -> transformers import fails -> every TTS backend fails
- diffusers without WanPipeline / transformers without VitsModel
- ffmpeg missing
"""
from __future__ import annotations

import importlib
import shutil


def runtime_problems() -> list[str]:
    problems: list[str] = []
    try:
        importlib.import_module("google.protobuf.runtime_version")
    except Exception as e:  # noqa: BLE001
        problems.append("protobuf is too old/broken for this environment "
                        f"({type(e).__name__}). Fix: pip install -U 'protobuf>=5.29,<7' then RESTART the kernel.")
    try:
        importlib.import_module("transformers.modeling_utils")
    except Exception as e:  # noqa: BLE001
        problems.append(f"transformers cannot be imported ({type(e).__name__}: {str(e)[:120]}). "
                        "Run scripts/setup_env.py (notebook Cell 2) and restart the kernel.")
    try:
        getattr(importlib.import_module("transformers"), "VitsModel")
    except Exception:  # noqa: BLE001
        problems.append("transformers.VitsModel missing -> MMS Hindi TTS unavailable (need transformers>=4.33).")
    try:
        getattr(importlib.import_module("diffusers"), "WanPipeline")
    except Exception:  # noqa: BLE001
        problems.append("diffusers.WanPipeline missing -> need diffusers>=0.33 (Cell 2 installs 0.33.1).")
    if shutil.which("ffmpeg") is None:
        problems.append("ffmpeg not found -> apt-get install -y ffmpeg (Cell 2 does this).")
    return problems
