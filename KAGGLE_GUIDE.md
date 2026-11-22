# Kaggle Guide — Exact Steps

## Setup (one-time)

1. Kaggle → Notebooks → New Notebook. Settings: **Accelerator = GPU T4 x2** (or P100),
   **Internet = ON**, **Save version = Save**.
2. Get this project onto Kaggle — **pick one**:
   - **(A) GitHub (recommended):** push `freevideo-kaggle` to your own GitHub repo, then in
     Cell 2 set `PROJECT_REPO = 'https://github.com/<you>/freevideo-kaggle.git'`.
   - **(B) Kaggle Dataset:** Notebook → *Add Input* → *Upload* → *New Dataset*; upload the
     `freevideo-kaggle` folder or a zip of it. Cell 2 auto-detects it under `/kaggle/input/`.
     Leave `PROJECT_REPO` unchanged.
3. Open `notebooks/kaggle_freevideo_t4.ipynb` and run Cells 1–5 in order.

## What each cell does

| Cell | Action | Time |
|---|---|---|
| 1 | Install ffmpeg + CUDA 12.4 torch + diffusers stack | ~5–10 min |
| 2 | Fetch this project (GitHub or Dataset), clone FreeVideo, apply T4/P100 patches | ~1 min |
| 3 | Hardware + kernel probes (no download yet) | ~1 min |
| 4 | Point HF cache at `/kaggle/working/.hf_cache` (persistent) | instant |
| 5 | Launch web UI + public Cloudflare tunnel link | ~20 s |
| 6 | (Optional) generate directly from notebook, no browser | per script |

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

- **Web UI link does not open / `127.0.0.1:5000` refuses:** Kaggle has **no port preview**
  (unlike Colab). Only the `https://....trycloudflare.com` URL printed by Cell 5 works, and only
  while the session is alive. If it still fails, re-run Cell 5 (gets a fresh URL) or use Cell 6.

- **`FileNotFoundError: /kaggle/working/freevideo-kaggle`** or **`PROJECT NOT FOUND`** in Cell 2:
  the project itself was never fetched. Set `PROJECT_REPO` to your GitHub URL *or* attach the
  project as a Dataset input (see Setup step 2), then re-run Cell 2.

| Symptom | Fix |
|---|---|
| `CUDA_ARCH_UNSUPPORTED` | Expected without env. Production path (Wan) does not need FreeVideo env. |
| OOM during generation | Switch profile to Low VRAM / CPU-Assisted. Check `benchmark_results.json`. |
| TTS gated repo error | Accept terms at huggingface.co/ai4bharat/indic-parler-tts once. |
| Silent narration | TTS backend failed — check `narration_manifest.json` error. Do not ship silent videos. |
| ffmpeg missing | Re-run Cell 1 (`apt-get install ffmpeg`). |
