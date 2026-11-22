"""Generate the Kaggle notebook (.ipynb) from the project sources.

Usage: python scripts/make_kaggle_notebook.py [--out notebooks/kaggle_freevideo_t4.ipynb]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def build() -> dict:
    cells = [
        md("# FreeVideo-Kaggle — Hindi Script to Video on T4/P100\n\n"
           "Production low-VRAM AI video pipeline. Primary engine: **Wan2.1-T2V-1.3B (FP16)**.\n"
           "FreeVideo H3 is included as an optional experimental engine (see AUDIT.md).\n\n"
           "**Steps:** 1) Install 2) Probe kernels 3) Launch web UI 4) Paste script → Generate → Download.\n"
           "Settings: Accelerator = GPU T4 x2 (or P100), Internet = ON, Save version = Save."),
        code("# Cell 1 — Install system + Python deps (CUDA 12.4 stack for T4/P100)\n"
             "!apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1; echo ffmpeg done\n"
             "!pip install -q torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124\n"
             "!pip install -q diffusers==0.32.2 transformers==4.46.3 accelerate==1.1.1 safetensors==0.4.5 huggingface_hub==0.26.2 einops==0.8.0 sentencepiece==0.2.0 parler-tts==0.2.1 soundfile==0.12.1 \"imageio[ffmpeg]==2.37.0\" flask psutil numpy pillow requests python-dotenv av\n"
             "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n"
             "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')\n"
             "print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'n/a')"),
        code("# Cell 2 — Clone project (or upload freevideo-kaggle/ as a Kaggle Dataset and unzip)\n"
             "!git clone https://github.com/FlashML-org/FreeVideo.git /kaggle/working/FreeVideo 2>&1 | tail -2\n"
             "# If you uploaded this project as a dataset, unzip it instead:\n"
             "# !unzip -q /kaggle/input/freevideo-kaggle/freevideo-kaggle.zip -d /kaggle/working/\n"
             "import os, sys; sys.path.insert(0, '/kaggle/working/freevideo-kaggle')\n"
             "os.chdir('/kaggle/working/freevideo-kaggle')\n"
             "!python patches/apply_patches.py /kaggle/working/FreeVideo\n"
             "print('project ready')"),
        code("# Cell 3 — Hardware + kernel probes (run BEFORE downloading any model)\n"
             "import sys; sys.path.insert(0, '/kaggle/working/freevideo-kaggle')\n"
             "from app.hardware import HardwareDetector\n"
             "print(HardwareDetector().detect().to_dict())\n"
             "!python scripts/probe_kernels.py --out /kaggle/working/kernel_probe.json\n"
             "!python scripts/benchmark.py --quick --out /kaggle/working/benchmark_quick.json"),
        code("# Cell 4 — Persistent HF cache (survives sessions via /kaggle/working)\n"
             "import os\n"
             "os.environ['HF_HOME'] = '/kaggle/working/.hf_cache'\n"
             "os.environ['HF_HUB_CACHE'] = '/kaggle/working/.hf_cache/hub'\n"
             "os.environ['FREEVVIDEO_REPO'] = '/kaggle/working/FreeVideo'\n"
             "print('HF cache ->', os.environ['HF_HOME'])"),
        code("# Cell 5 — Launch web UI (runs in background; open the preview URL)\n"
             "import sys, threading; sys.path.insert(0, '/kaggle/working/freevideo-kaggle')\n"
             "from app.server import run\n"
             "t = threading.Thread(target=run, kwargs={'host': '0.0.0.0', 'port': 5000}, daemon=True)\n"
             "t.start()\n"
             "print('Web UI starting on port 5000 — use the Kaggle preview link.')\n"
             "print('Paste your Hindi script, preview voice, then Start generation.')"),
        md("## Manual steps\n"
           "1. Open the web UI preview URL.\n"
           "2. Paste your Hindi/Hinglish script (stored verbatim).\n"
           "3. Select engine (Wan2.1 recommended), profile, style, TTS.\n"
           "4. Preview narration audio.\n"
           "5. Start generation → watch scene progress → preview scenes.\n"
           "6. Download final MP4 (also saved to /kaggle/working/outputs/).\n"
           "7. Optional: set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars for Telegram; "
           "YOUTUBE_CLIENT_SECRETS for YouTube upload.\n\n"
           "## Session recovery\n"
           "If the session ends mid-generation, re-run Cells 1–5 and call the API with "
           "`resume_project_id` (shown in the UI manifest) — completed scenes are skipped."),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "notebooks" / "kaggle_freevideo_t4.ipynb"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
