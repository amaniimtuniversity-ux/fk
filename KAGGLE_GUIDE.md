# Kaggle Guide — Exact Steps

## Setup (one-time)

1. Kaggle → Notebooks → New Notebook. Settings: **Accelerator = GPU T4 x2** (or P100),
   **Internet = ON**, **Save version = Save**.
2. Upload `freevideo-kaggle/` as a **Dataset** (or `git clone` this project + FreeVideo).
3. Open `notebooks/kaggle_freevideo_t4.ipynb` and run Cells 1–5 in order.

## What each cell does

| Cell | Action | Time |
|---|---|---|
| 1 | Install ffmpeg + CUDA 12.4 torch + diffusers stack | ~5–10 min |
| 2 | Clone FreeVideo, apply T4/P100 patches | ~1 min |
| 3 | Hardware + kernel probes (no download yet) | ~1 min |
| 4 | Point HF cache at `/kaggle/working/.hf_cache` (persistent) | instant |
| 5 | Launch web UI on port 5000 | instant |

## Generate a video

1. Open the web UI preview URL.
2. **Paste your Hindi/Hinglish script** (stored verbatim, SHA-256 recorded).
3. Select: Engine (**Wan2.1 1.3B** recommended), Profile (**Balanced**), Style, TTS.
4. Click **Preview narration** — listen and verify pronunciation.
5. Click **Start generation** — watch scene progress (each scene is checkpointed).
6. Preview scenes, then **Download final MP4** (also in `/kaggle/working/outputs/`).

## Session recovery

If the session ends mid-generation: re-run Cells 1–5, then POST `/api/generate` with the same
script + `resume_project_id` (from the previous manifest). Completed scenes are skipped.

## Benchmarks (run once, attach results)

```bash
python scripts/benchmark.py --quick   # probes only
python scripts/benchmark.py --full    # + short Wan generation (downloads ~5.5 GB)
```

## Telegram / YouTube (optional)

```bash
export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
export YOUTUBE_CLIENT_SECRETS=client_secrets.json  # from Google Cloud Console
```

Then use `/api/send-telegram` / `/api/upload-youtube`. Secrets are env-only, never in code.

## Model cache as a Kaggle Dataset (optional, avoids re-download)

1. After first download, save `/kaggle/working/.hf_cache` as a private Dataset.
2. In later sessions, attach the Dataset and set `HF_HOME` to its path.
3. `KaggleStorageManager.verify_cached()` confirms the snapshot before reuse.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA_ARCH_UNSUPPORTED` | Expected without env. Production path (Wan) does not need FreeVideo env. |
| OOM during generation | Switch profile to Low VRAM / CPU-Assisted. Check `benchmark_results.json`. |
| TTS gated repo error | Accept terms at huggingface.co/ai4bharat/indic-parler-tts once. |
| Silent narration | TTS backend failed — check `narration_manifest.json` error. Do not ship silent videos. |
| ffmpeg missing | Re-run Cell 1 (`apt-get install ffmpeg`). |
