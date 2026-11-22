# FreeVideo-Kaggle — Production Low-VRAM AI Video on T4/P100

Paste a Hindi/Hinglish script → natural Hindi narration → genuine AI video scenes → final MP4
with subtitles, ambience, and loudness-normalized audio. Built for **Kaggle T4 (SM75) / P100 (SM60)**.

## Quick start (Kaggle)

Open `notebooks/kaggle_freevideo_t4.ipynb`, run Cells 1–5, open the **public tunnel link** printed by Cell 5 (Kaggle has no port preview), paste your script, generate. Or use Cell 6 to generate without the browser.

## Quick start (local)

```bash
pip install -r requirements.txt          # core (CPU-testable)
pip install -r requirements-kaggle.txt   # + CUDA stack (Kaggle)
python scripts/run_web.py                # http://0.0.0.0:5000
python -m pytest tests/ -v               # 23 tests
python scripts/benchmark.py --quick      # kernel probes
```

## Architecture

- **Primary engine:** Wan2.1-T2V-1.3B (FP16, SDPA, sequential offload) — best quality that fits T4/P100.
- **Secondary:** CogVideoX-2B (T2V), SVD-XT (I2V fallback).
- **Optional experimental:** FreeVideo MiniMax H3 (patched, SM80+; cannot fit Kaggle RAM/disk — see AUDIT.md).
- **TTS:** Indic Parler-TTS (primary, gated → `HF_TOKEN`), MMS Hindi (open fallback, recorded in manifest), silent only on explicit request.
- **Script:** verbatim + SHA-256; scene planner never rewrites; full-coverage verified.
- **Recovery:** scene-level checkpoints; resume skips completed scenes; HF cache in `/kaggle/working`.

## Docs

- `AUDIT.md` — FreeVideo source audit + SM80 root cause + model evaluation.
- `TECHNICAL_REPORT.md` — honest tested-vs-Kaggle report (read this first).
- `KAGGLE_GUIDE.md` — exact notebook steps + troubleshooting.
- `patches/` — upstream T4/P100 patches (verified to apply cleanly).

## License

New code: Apache-2.0. Upstream FreeVideo: Apache-2.0. Model weights: see each model's HF license.
No paid APIs. No data sent to third parties (Telegram/YouTube only when you configure them).
