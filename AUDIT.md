# FreeVideo Audit — T4/P100 Compatibility

**Upstream:** https://github.com/FlashML-org/FreeVideo (cloned at `../FreeVideo`, commit `d53d1b2`)
**Date:** 2026-11-22
**Method:** Full source inspection of `freevideo_engine/` (368 modules), `pyproject.toml`, model manifests.

## 1. What FreeVideo is

FreeVideo is a local inference engine for **MiniMax H3 / VDN-H3** (Video DeltaNet hybrid attention),
built on OpenVDN's 8-step VDN-H3 model. It runs as a ComfyUI plugin with Windows/macOS/Linux launchers.
Core design: hardware-adaptive execution (FP8 on datacenter cards, INT8 on GeForce/RTX30, BF16-weight-only
on Ampere), weight streaming, async prefetch, chunked computation. Claimed minimum: **8 GB VRAM / 16 GB RAM**.

## 2. Model size (measured from `model_files.json`)

| Artifact | Size |
|---|---|
| Full BF16 bundle (`OpenVDN/vdn-minimax-h3`) | **99.1 GB** (40 files) |
| Transformer alone (14 shards) | **66.3 GB** (~33B params BF16) |
| FP8 slim cache (`vdn-minimax-h3-edge`, per_tensor/rowwise/int8) | **22.9 GB** each |

Plus: text encoder, video+audio VAE, sampling caches, ComfyUI, VDN vendor code, patched Diffusers.

**Kaggle budget:** `/kaggle/working` ≈ **20 GB** persistent, RAM ≈ **29 GB**, 4 CPUs, 12 h/session, ~30 h/week.
The BF16 bundle alone is 5× the persistent disk. The FP8 cache (22.9 GB) exceeds it before any other file.
CPU offload of a 33B model needs ~66 GB RAM (BF16) or ~33 GB (FP16) — both exceed Kaggle's 29 GB.

## 3. Root cause of the SM80+ limitation

### 3.1 Hard gate — `freevideo_engine/hardware.py::cuda_compatibility()`

```python
admitted = known and capability >= (8, 0)
```

T4 (7,5) and P100 (6,0) are **rejected before any kernel runs** with `CUDA_ARCH_UNSUPPORTED`.
Error text: *"FreeVideo requires native BF16 GPU computation (compute capability 8.0 / SM80 or newer)."*
This gate is **fixable in code** (see `patches/freevideo_t4_p100.patch`), but it is only the first of five blockers.

### 3.2 Precision — BF16/FP8 are native only on SM80+/SM89+

- `backends/cuda.py`: linear policies are `native-fp8` (Ada SM89+), `bf16-weight-only` (Ampere SM80–89), `int8`.
- `prepare_linears()`: *"Native FP8 GEMM requires Ada or newer; choose bf16-weight-only on Ampere."*
- T4 has **no BF16 tensor cores** (BF16 matmul is FP32-emulated, slow, some ops fail). P100 has **no BF16/FP8 at all**.
- FP8 weights stored on disk do **not** execute on T4/P100. They must be dequantized to FP16 on CPU
  (`app/precision.py::convert_checkpoint_fp8_to_fp16`, streaming, memory-bounded) and run as FP16 GEMMs.
  This is mathematically valid (dequant is exact: `w = fp8 * scale`) but the model is then a 33B FP16 model
  needing ~66 GB RAM — see §2.

### 3.3 Attention — FA2/FA4/Sage2 require SM80+

`backends/cuda_attention.py` offers `cudnn / torch-flash / sage2 / fa2 / fa4`:
- **FA2** requires SM80+ (Ampere). **FA4** requires SM90+ (Hopper). **SageAttention 2.x** requires SM80+.
- On T4/P100 the only viable paths are **SDPA cudnn** (T4) and **SDPA math** (P100, always correct).
- `app/attention.py` implements probed selection + exact chunked SDPA (query slicing).

### 3.4 Triton kernels — SM75 borderline, SM60 unsupported

- `fp8_ops.py`, `int8_ops.py`, `row_kernels.py`, `linear_kernels.py` use `triton.jit` + `tl.dot`.
- Triton 3.x (pinned `triton==3.7.1` in `pyproject.toml`) supports Turing (SM75) for many kernels but
  **dropped Pascal (SM60) entirely**. P100 cannot run any Triton kernel.
- `torch.compile` (Inductor) requires SM80+ in practice. Both must be **disabled** on T4/P100
  (eager PyTorch fallback). FreeVideo's `linear_kernels.py`/`blocks.py` compile paths are therefore
  bypassed in the T4/P100 profile.

### 3.5 VAE / decode

- `vae_tiles.py` (`TileDecoder`) already implements layer-transposed tiling + offload — good design,
  reusable. But it compiles decoder blocks (`torch.compile`) and assumes BF16 latents. FP16 latents work
  with the same tiling logic; compile must be disabled on SM60.

## 4. What is mathematically replaceable vs fundamentally incompatible

| Component | Replaceable on T4/P100? | Replacement |
|---|---|---|
| SM80 hard gate | ✅ Yes (code) | `FREEVVIDEO_T4_P100=1` admits SM60+ for setup; probes still gate readiness |
| BF16 compute | ✅ Yes (slow) | FP16 compute + FP32 sensitive layers (norms/softmax/rotary) |
| FP8 weights | ✅ Yes (dequant) | CPU FP8→FP16 streaming conversion; run as FP16 GEMM |
| FA2/FA4/Sage2 | ✅ Yes | SDPA cudnn (T4) / math (P100) + chunked queries |
| Triton kernels | ⚠️ Partial | Eager PyTorch fallbacks; P100 has no Triton at all |
| torch.compile | ✅ Disable | Eager execution (slower, correct) |
| 33B model in 29 GB RAM | ❌ **No** | Cannot offload 33B FP16 (66 GB) into 29 GB RAM. Fundamental. |
| 99 GB download in 20 GB disk | ❌ **No** | Fundamental. Even 22.9 GB FP8 cache exceeds persistent disk. |

**Conclusion:** The SM80 gate, precision, and attention blockers are all fixable. The **model size /
RAM / disk blockers are not**. MiniMax H3 cannot realistically run on Kaggle T4/P100 even after every
valid optimization. The correct engineering decision is to keep FreeVideo as an **optional experimental
engine** (preserved, patched, testable on SM80+) and run a **production alternative** on T4/P100.

## 5. Alternative model evaluation (T4 SM75 / P100 SM60, FP16, 16 GB VRAM)

| Model | Params | Download | VRAM FP16 (offloaded) | T2V/I2V | T4 | P100 | Verdict |
|---|---|---|---|---|---|---|---|
| **Wan2.1-T2V-1.3B** | 1.3B | ~5.5 GB | ~10 GB (official 8.19 GB) | T2V | ✅ | ✅ | **Primary production** |
| CogVideoX-2B | 2B | ~9 GB | ~12 GB (4 GB min w/ opts) | T2V | ✅ | ✅ | Secondary production |
| LTX-Video 2B | 2B | ~8 GB | ~12 GB | T2V+I2V | ✅ | ❌ (Triton) | Experimental |
| SVD-XT | 1.5B | ~6 GB | ~8 GB | I2V | ✅ | ✅ | P100 fallback |
| Wan2.2-TI2V-5B | 5B | ~11 GB | ~20 GB | T2V+I2V | ❌ | ❌ | Unsupported |
| HunyuanVideo 13B | 13B | ~26 GB | 31–60 GB | T2V | ❌ | ❌ | Unsupported |
| FreeVideo H3 | 33B | 23–99 GB | ~99 GB | T2V+I2V | ❌ | ❌ | Experimental only |

Selection: **Wan2.1-T2V-1.3B** — best cinematic quality per GB, FP16-native, official 8.19 GB VRAM,
fits T4 with sequential offload + T5 on CPU, runs on P100 with SDPA-math fallback.

## 6. Files modified (upstream patches)

`patches/freevideo_t4_p100.patch` (3 files, +26/−2):
1. `freevideo_engine/hardware.py` — admit SM60+ when `FREEVVIDEO_T4_P100=1` (setup only; probes still required).
2. `freevideo_engine/backends/cuda.py` — reject FP8 execution on SM75/SM60 with a clear dequant instruction.
3. `freevideo_engine/backends/cuda_attention.py` — fail fast on FA2/FA4/Sage2 below SM80.

Apply: `python patches/apply_patches.py /path/to/FreeVideo` (verified: applies cleanly, T4 admitted with env, rejected without).

## 7. Hindi TTS evaluation

| System | Size | VRAM | Quality | Verdict |
|---|---|---|---|---|
| **Indic Parler-TTS** (ai4bharat, 0.94B, Apache-2.0) | ~3.8 GB FP32 | ~4–8 GB | Best naturalness/prosody | **Primary** (runs before video, then unloads) |
| AI4Bharat VITS (Indic-TTS) | <0.5 GB | <1 GB | Good, less expressive | Lightweight fallback |
| Silent WAV | 0 | 0 | None | Test-only, explicitly flagged |

Parler-TTS is a gated HF repo (one-time terms acceptance). Sentence-level synthesis + stitching with
0.35 s pauses; fixed voice description for narrator consistency; per-paragraph regeneration supported.
