"""HindiTTSService: natural Hindi narration.

Backends (in preference order):
1. Indic Parler-TTS (ai4bharat/indic-parler-tts, 0.94B, Apache-2.0) — best
   naturalness/prosody. ~3.8GB FP32 weights; runs on T4 (separate from video
   VRAM since TTS runs first, then unloads) or CPU (slow). Gated HF repo:
   user must accept terms once on HF website.
2. Meta MMS-TTS Hindi (facebook/mms-tts-hin, VITS, CC-BY-NC 4.0, ~145MB, NOT gated) — real
   Hindi speech, used automatically when Parler is unavailable (gated/no token). The manifest
   records backend="mms-hin" so this is never hidden.
3. Silent WAV — ONLY when the caller explicitly asks for backend="silent" (tests/offline smoke).
   It is NEVER used as an automatic fallback: a real TTS failure fails the pipeline.

Features: sentence-level synthesis, stitching with pauses, voice preview,
per-paragraph regeneration, consistent narrator (fixed voice description).
Original script is never modified (tts_safe_copy is used for synthesis).
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .script_manager import split_sentences, tts_safe_copy


@dataclass
class TTSResult:
    success: bool
    audio_path: Optional[Path] = None
    duration_seconds: float = 0.0
    backend: str = ""
    sample_rate: int = 24000
    error: str = ""
    is_silent_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success, "audio_path": str(self.audio_path) if self.audio_path else None,
            "duration_seconds": round(self.duration_seconds, 2), "backend": self.backend,
            "sample_rate": self.sample_rate, "error": self.error,
            "is_silent_fallback": self.is_silent_fallback,
        }


MMS_HINDI_REPO = "facebook/mms-tts-hin"


def hf_token() -> Optional[str]:
    """HF token from env (HF_TOKEN / HUGGINGFACE_HUB_TOKEN) or Kaggle Secrets (HF_TOKEN). None if absent."""
    import os

    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    try:  # Kaggle: Add-ons -> Secrets -> HF_TOKEN
        from kaggle_secrets import UserSecretsClient  # type: ignore

        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
            return tok
    except Exception:
        pass
    return None


def _write_silent_wav(path: Path, seconds: float, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n)


class HindiTTSService:
    def __init__(self, backend: str = "indic-parler", model: str = "ai4bharat/indic-parler-tts",
                 voice_description: str = "", sample_rate: int = 24000, device: str = "auto",
                 allow_fallback: bool = True):
        self.backend = backend
        self.allow_fallback = allow_fallback
        self.model = model
        self.voice_description = voice_description or (
            "Divya's voice is expressive and warm in delivery, with a very close recording "
            "that almost has no background noise."
        )
        self.sample_rate = sample_rate
        self.device = device
        self._model = None
        self._tokenizer = None
        self._desc_tokenizer = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load(self) -> None:
        if self._model is not None:
            return
        token = hf_token()
        if self.backend == "indic-parler":
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            device = self._resolve_device()
            self._model = ParlerTTSForConditionalGeneration.from_pretrained(self.model, token=token).to(device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model, token=token)
            self._desc_tokenizer = AutoTokenizer.from_pretrained(self._model.config.text_encoder._name_or_path,
                                                                 token=token)
            self._device = device
        elif self.backend == "mms-hin":
            from transformers import AutoTokenizer, VitsModel

            device = self._resolve_device()
            repo = self.model if "mms-tts" in self.model else MMS_HINDI_REPO
            self._model = VitsModel.from_pretrained(repo, token=token).to(device)
            self._tokenizer = AutoTokenizer.from_pretrained(repo, token=token)
            self._device = device
        elif self.backend == "silent":
            self._model = "silent"
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

    def synthesize_sentence(self, sentence: str) -> tuple[Any, int]:
        """Return (numpy float32 array, sample_rate)."""
        self.load()
        if self.backend == "silent":
            import numpy as np

            return np.zeros(int(len(sentence.split()) * 0.4 * self.sample_rate), dtype="float32"), self.sample_rate
        import torch

        device = getattr(self, "_device", "cpu")
        text = tts_safe_copy(sentence)
        if self.backend == "mms-hin":
            inputs = self._tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                wav = self._model(**inputs).waveform
            return wav.squeeze().float().cpu().numpy(), int(self._model.config.sampling_rate)
        desc = self._desc_tokenizer(self.voice_description, return_tensors="pt").to(device)
        prompt = self._tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            generation = self._model.generate(
                input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask,
            )
        audio = generation.cpu().numpy().squeeze()
        sr = int(getattr(self._model.config, "sampling_rate", self.sample_rate))
        return audio, sr

    def synthesize_script(self, script: str, output_path: Path,
                          pause_seconds: float = 0.35) -> TTSResult:
        """Sentence-level synthesis + stitching. Returns stitched WAV."""
        import time

        t0 = time.time()
        sentences = split_sentences(script)
        if not sentences:
            return TTSResult(success=False, error="No sentences in script.")
        try:
            import numpy as np
            import soundfile as sf

            chunks: list[Any] = []
            sr = self.sample_rate
            for s in sentences:
                audio, sr = self.synthesize_sentence(s)
                chunks.append(np.asarray(audio, dtype="float32"))
                pause = np.zeros(int(pause_seconds * sr), dtype="float32")
                chunks.append(pause)
            full = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), full, sr)
            return TTSResult(
                success=True, audio_path=output_path,
                duration_seconds=len(full) / sr, backend=self.backend,
                sample_rate=sr, is_silent_fallback=(self.backend == "silent"),
            )
        except Exception as e:
            first = f"{self.backend} failed: {type(e).__name__}: {str(e)[:300]}"
            if self.backend == "indic-parler" and self.allow_fallback:
                # Real speech fallback (open model). Explicitly recorded in the result/manifest.
                hint = ""
                if "gated" in str(e).lower() or "401" in str(e) or "restricted" in str(e).lower():
                    hint = (" | Indic Parler-TTS is a GATED repo: accept the license at "
                            "https://huggingface.co/ai4bharat/indic-parler-tts and set HF_TOKEN "
                            "(Kaggle: Add-ons -> Secrets -> HF_TOKEN).")
                self.unload()
                fb = HindiTTSService(backend="mms-hin", model=MMS_HINDI_REPO, sample_rate=self.sample_rate,
                                     device=self.device, allow_fallback=False)
                res = fb.synthesize_script(script, output_path, pause_seconds)
                fb.unload()
                res.error = (first + hint + " -> used mms-hin fallback" + (f" | {res.error}" if res.error else ""))
                return res
            return TTSResult(success=False, backend=self.backend, error=first)

    def preview(self, text: str, output_path: Path) -> TTSResult:
        return self.synthesize_script(text, output_path)

    def unload(self) -> None:
        self._model = None
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
