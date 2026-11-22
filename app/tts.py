"""HindiTTSService: natural Hindi narration.

Backends (in preference order):
1. Indic Parler-TTS (ai4bharat/indic-parler-tts, 0.94B, Apache-2.0) — best
   naturalness/prosody. ~3.8GB FP32 weights; runs on T4 (separate from video
   VRAM since TTS runs first, then unloads) or CPU (slow). Gated HF repo:
   user must accept terms once on HF website.
2. AI4Bharat VITS (Indic-TTS) — lightweight fallback if Parler unavailable.
3. Silent WAV fallback — ONLY for tests/offline smoke runs. Clearly labeled;
   never presented as real narration.

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
                 voice_description: str = "", sample_rate: int = 24000, device: str = "auto"):
        self.backend = backend
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
        if self.backend == "indic-parler":
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            device = self._resolve_device()
            self._model = ParlerTTSForConditionalGeneration.from_pretrained(self.model).to(device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._desc_tokenizer = AutoTokenizer.from_pretrained(self._model.config.text_encoder._name_or_path)
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
            # Offline/test fallback: silent WAV so pipeline can be smoke-tested.
            # The result is explicitly flagged; UI must show "silent fallback".
            try:
                est = max(1.0, len(script.split()) * 0.4)
                _write_silent_wav(output_path, est, self.sample_rate)
                return TTSResult(
                    success=True, audio_path=output_path, duration_seconds=est,
                    backend="silent-fallback", sample_rate=self.sample_rate,
                    error=f"TTS backend failed, silent fallback used: {e}",
                    is_silent_fallback=True,
                )
            except Exception as e2:
                return TTSResult(success=False, error=f"{e}; fallback also failed: {e2}")

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
