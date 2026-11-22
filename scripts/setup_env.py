"""Install + verify the runtime (used by Kaggle notebook Cell 1 AND testable locally).

Design rules learned the hard way on Kaggle:
- Never one giant `pip install a b c ...`: one bad pin (e.g. a version that does not exist)
  aborts the whole line and you silently run on whatever was preinstalled.
- Never pin torch: the Kaggle image ships a CUDA torch that works on T4; replacing it is slow
  and risky. (Locally, install torch yourself first — CPU is fine for tests.)
- transformers is pinned to 4.46.1 because parler-tts 0.2.3 (best Hindi TTS) requires exactly
  that; diffusers 0.33.1 is the first release with WanPipeline and works with it.
- parler-tts is installed with --no-deps so it cannot move torch/transformers; its codec dep
  (descript-audio-codec) is installed separately.
- Some of those deps DOWNGRADE protobuf, which breaks every `*_pb2` module compiled for
  protobuf>=5.26 ("cannot import name 'runtime_version' from 'google.protobuf'") and with it
  transformers itself -> both TTS backends die. So protobuf is re-pinned LAST and verified.

Usage:
    python scripts/setup_env.py            # install + verify, exit 1 on failure
    python scripts/setup_env.py --verify   # verify only (no installs)
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys

CORE = [
    "transformers==4.46.1", "diffusers==0.33.1", "accelerate>=1.1", "safetensors", "sentencepiece",
    "huggingface_hub>=0.26,<1.0", "soundfile", "indic-transliteration", "imageio[ffmpeg]", "flask", "psutil", "av", "einops", "ftfy",
    "numpy", "pillow", "requests", "python-dotenv", "scipy",
]
PARLER = ["--no-deps", "parler-tts==0.2.3"]
PARLER_DEPS = ["descript-audio-codec"]
PROTOBUF = ["--upgrade", "protobuf>=5.29,<7"]


def pip(*args: str) -> bool:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    ok = r.returncode == 0
    print(("OK    " if ok else "FAIL  ") + "pip install " + " ".join(args), flush=True)
    if not ok:
        tail = (r.stderr or r.stdout).strip().splitlines()
        print("      " + (tail[-1][:220] if tail else "?"), flush=True)
    return ok


def check(mod: str, attr: str | None, label: str) -> bool:
    try:
        m = importlib.import_module(mod)
        if attr:
            getattr(m, attr)
        ver = getattr(m, "__version__", "")
        print(f"READY   {label} {ver}".rstrip(), flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"MISSING {label}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return False


def verify() -> dict[str, bool]:
    importlib.invalidate_caches()
    res = {}
    res["protobuf"] = check("google.protobuf", None, "protobuf") and check("google.protobuf.runtime_version", None,
                                                                            "protobuf.runtime_version")
    res["torch"] = check("torch", "cuda", "torch")
    res["transformers"] = check("transformers", "AutoTokenizer", "transformers")
    res["transformers.modeling_utils"] = check("transformers.modeling_utils", "PreTrainedModel",
                                               "transformers.modeling_utils (needs working protobuf)")
    res["diffusers"] = check("diffusers", "WanPipeline", "diffusers.WanPipeline (Wan2.1)")
    res["mms"] = check("transformers", "VitsModel", "MMS Hindi TTS (VitsModel)")
    res["parler"] = check("parler_tts", "ParlerTTSForConditionalGeneration", "Indic Parler-TTS (optional)")
    res["soundfile"] = check("soundfile", None, "soundfile")
    res["indic_transliteration"] = check("indic_transliteration", None, "indic-transliteration (Hinglish->Devanagari for MMS TTS)")
    res["imageio"] = check("imageio", None, "imageio")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"GPU     {torch.cuda.get_device_name(0)} sm{''.join(map(str, torch.cuda.get_device_capability(0)))}")
        else:
            print("GPU     none (CPU only)")
    except Exception:
        pass
    return res


def required_ok(res: dict[str, bool]) -> bool:
    return all(res[k] for k in ("protobuf", "torch", "transformers", "transformers.modeling_utils",
                                "diffusers", "mms", "soundfile", "imageio", "indic_transliteration"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="verify only, no installs")
    ap.add_argument("--no-parler", action="store_true", help="skip optional Parler-TTS install")
    a = ap.parse_args()
    if not a.verify:
        pip(*CORE)
        if not a.no_parler:
            if pip(*PARLER):
                pip(*PARLER_DEPS)
        pip(*PROTOBUF)  # LAST, on purpose
    res = verify()
    ok = required_ok(res)
    print("\nRESULT:", "ALL REQUIRED READY" if ok else "REQUIRED COMPONENT MISSING — fix before running the pipeline")
    if not res.get("parler"):
        print("NOTE: Indic Parler-TTS unavailable -> MMS Hindi TTS will be used (recorded in the manifest).")
    if not ok and not a.verify:
        print("If protobuf was already imported in this kernel before the fix: Run -> Restart kernel, then re-run.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
