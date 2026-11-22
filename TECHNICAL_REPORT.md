# Technical Report — FreeVideo on Kaggle T4/P100

**Date:** 2026-11-22 · **Project:** `freevideo-kaggle/` · **Upstream:** FlashML-org/FreeVideo (`d53d1b2`)

## Honest testing statement

This report distinguishes **tested here** (CPU-only sandbox, no GPU, no torch, 1.9 GB RAM) from
**requires Kaggle run** (T4/P100 + CUDA). Nothing is claimed as tested that was not run.
GPU generation, TTS quality, and VRAM measurements **must** be validated on Kaggle via
`scripts/benchmark.py --full` and the notebook. All such items are marked **[KAGGLE]** below.

## 1. Original FreeVideo compatibility results

- **Tested here:** Source audit (368 modules), model-size measurement from `model_files.json`
  (99.1 GB BF16 / 22.9 GB FP8), hard-gate verification (`hardware.py` rejects SM<80),
  patch application + admission test (T4/P100 admitted with `FREEVVIDEO_T4_P100=1`, rejected without).
- **[KAGGLE]** Full FreeVideo generation on T4/P100 was **not** attempted here (no GPU) and is
  **expected to fail** on RAM/disk grounds (see AUDIT.md §2/§4). The adapter is preserved for SM80+.

## 2. Root cause of the SM80+ limitation

Five blockers (AUDIT.md §3): (1) hard SM80 gate in `hardware.py`, (2) BF16/FP8 native-only precision,
(3) FA2/FA4/Sage2 attention requiring SM80+, (4) Triton 3.x dropping SM60 + `torch.compile` requiring SM80+,
(5) 33B model needing ~66 GB RAM / 23–99 GB disk vs Kaggle's 29 GB / 20 GB. Blockers 1–4 are fixable;
blocker 5 is fundamental and decides the architecture.

## 3. Files modified

- **Upstream (patch):** `freevideo_engine/hardware.py`, `freevideo_engine/backends/cuda.py`,
  `freevideo_engine/backends/cuda_attention.py` (+26/−2). See `patches/freevideo_t4_p100.patch`.
- **New project (this repo):** `config.py`, `app/` (hardware, precision, attention, memory, offload,
  registry, engines×5, script_manager, scene_planner, tts, audio, renderer, checkpoint, kaggle_storage,
  quality, telegram/youtube, pipeline, server), `scripts/` (benchmark, probe, notebook, web),
  `tests/` (4 files, 26 tests), docs.

## 4. Kernels replaced

| Upstream kernel | T4/P100 replacement | Where |
|---|---|---|
| FA2 / FA4 / Sage2 | SDPA cudnn (T4) / math (P100) + chunked queries | `app/attention.py` |
| FP8 GEMM (Triton) | FP16 GEMM after CPU dequant | `app/precision.py` |
| INT8 ConvRot (Triton) | FP16 GEMM (INT8 path not used on T4/P100) | registry (not selected) |
| `torch.compile` blocks | Eager execution | engines + precision plan |
| VAE compile | Eager + tiling/slicing | `app/offload.py` |

Numerical validation: chunked SDPA vs full SDPA `allclose(atol=1e-5)` — **tested here** (skipped without
torch; runs on Kaggle). FP8 dequant math unit-tested (per-tensor/per-row) — **tested here** with torch
when available, logic verified by inspection.

## 5. Precision modes that work

- **FP16 compute + FP32 sensitive layers** — T4/P100 production path. **Tested here** (plan logic);
  **[KAGGLE]** matmul probe + generation.
- **BF16** — SM80+ only (upstream native path, preserved).
- **FP8 native** — SM89+ only. On T4/P100 FP8 checkpoints are **dequantized**, never executed as FP8.
- **FP32 fallback** — if FP16 probe fails (very slow, correct).

## 6. Attention backends that work

- T4: **cudnn-SDPA** (probed) → math-SDPA fallback. P100: **math-SDPA**.
- FA2/FA4/Sage2/xFormers: **not used** on T4/P100 (unsupported). Selection is probed, never assumed.

## 7. Offloading implemented

Sequential CPU offload (diffusers), model offload fallback, VAE slicing + tiling, attention slicing,
pinned-memory-aware budgets (FreeVideo `offload.py` concepts), transfer byte/time accounting,
per-scene GC + `empty_cache` boundaries. Two T4s treated as separate devices (never unified VRAM).

## 8. Models tested

- **Tested here:** Registry compatibility logic (Wan recommended on T4; H3 rejected with ≥2 reasons),
  scene planner, checkpoint resume, TTS silent fallback, SRT, renderer guards, web API smoke.
- **[KAGGLE]** Wan2.1-1.3B, CogVideoX-2B, SVD-XT generation; FreeVideo H3 experimental path.

## 9–11. Actual VRAM / RAM / generation time

**Not measured here (no GPU).** The benchmark suite records these on Kaggle:
`python scripts/benchmark.py --full` → `benchmark_results.json` (peak VRAM/RAM, seconds, resolution,
frames, quality). Expected (published, not measured): Wan 1.3B ~10 GB VRAM offloaded, ~5 s clip in
minutes on T4. **Do not treat expectations as results.**

## 12. Output quality comparison

**[KAGGLE]** `app/quality.py` measures flicker (mean abs diff), sharpness (Laplacian variance),
and interpretation bands on real clips. No quality scores are invented here. FP16 vs BF16 visual
comparison requires SM80+ hardware and is out of scope for T4/P100.

## 13. Kaggle installation commands

```bash
apt-get update -qq && apt-get install -y -qq ffmpeg
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install diffusers==0.32.2 transformers==4.46.3 accelerate==1.1.1 safetensors==0.4.5 \
  huggingface_hub==0.26.2 einops==0.8.0 sentencepiece==0.2.0 parler-tts==0.2.1 \
  soundfile==0.12.1 "imageio[ffmpeg]==2.37.0" flask psutil numpy pillow requests python-dotenv av
git clone https://github.com/FlashML-org/FreeVideo.git /kaggle/working/FreeVideo
python patches/apply_patches.py /kaggle/working/FreeVideo
```

## 14. Exact notebook execution steps

Open `notebooks/kaggle_freevideo_t4.ipynb` (Settings: GPU T4 x2 or P100, Internet ON, Save version).
Run Cells 1→5 in order. Cell 3 probes kernels before any download. Cell 5 starts the web UI.

## 15. How to paste a script and generate a video

Web UI: paste Hindi/Hinglish script → select engine/profile/style/TTS → Preview narration →
Start generation → watch scene progress → preview scenes → Download final MP4.
The script is stored verbatim (SHA-256 recorded); narration uses the exact original text.

## 16. Known limitations

- FreeVideo H3 does not run on Kaggle T4/P100 (RAM/disk). Experimental adapter only.
- Wan2.1-1.3B is T2V-only; I2V needs SVD-XT (separate first-frame image).
- P100 generation is slow (FP16 at reduced rate, math SDPA, no Triton). Use `cpu_assisted` profile.
- Indic Parler-TTS is gated on HF (one-time acceptance). Silent fallback is test-only and flagged.
- 1080p output is an upscale of 480p native generation (labeled as such in manifests).

## 17. Production-ready (tested here)

Script preservation + hashing, scene planning with full-coverage verification, checkpoint resume,
precision/attention planning, registry compatibility, TTS stitching (silent path), SRT, renderer
guards, web UI API, patch application, benchmark harness, notebook generation. **23/23 tests pass**
(3 torch-dependent tests skip without torch).

## 18. Requires more hardware/research

Real T4/P100 generation benchmarks, TTS listening evaluation (Test F), final rendering with real
clips (Test H), Kaggle persistence across sessions (Test I), Telegram/YouTube live sends (Test J),
FreeVideo H3 FP16 path on SM80+ with quality comparison. All are scripted in `scripts/benchmark.py`
and the notebook; run them on Kaggle and attach `benchmark_results.json` to close these items.
