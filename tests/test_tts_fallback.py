"""TTS policy: gated Parler -> real MMS Hindi fallback; silent NEVER used unless explicitly requested."""
from __future__ import annotations

import numpy as np
import pytest

from app import tts as tts_mod
from app.tts import HindiTTSService, TTSResult


class _GatedError(Exception):
    pass


def test_parler_gated_falls_back_to_mms_and_records_it(tmp_path, monkeypatch):
    calls = []

    def fake_load(self):
        if self._model is not None:
            return
        calls.append(self.backend)
        if self.backend == "indic-parler":
            raise _GatedError("401 Client Error. Cannot access gated repo ... restricted")
        self._model = "fake"

    def fake_sentence(self, sentence):
        self.load()
        return np.random.uniform(-0.3, 0.3, 16000).astype("float32"), 16000

    monkeypatch.setattr(HindiTTSService, "load", fake_load)
    monkeypatch.setattr(HindiTTSService, "synthesize_sentence", fake_sentence)
    svc = HindiTTSService(backend="indic-parler")
    res = svc.synthesize_script("नमस्ते दुनिया। यह एक परीक्षण है।", tmp_path / "n.wav")
    assert res.success and res.backend == "mms-hin" and not res.is_silent_fallback
    assert "gated" in res.error.lower() and "HF_TOKEN" in res.error
    assert (tmp_path / "n.wav").stat().st_size > 1000
    assert calls == ["indic-parler", "mms-hin"]


def test_no_silent_fallback_when_everything_fails(tmp_path, monkeypatch):
    def boom(self):
        raise RuntimeError("no network")

    monkeypatch.setattr(HindiTTSService, "load", boom)
    res = HindiTTSService(backend="indic-parler").synthesize_script("नमस्ते।", tmp_path / "n.wav")
    assert not res.success and not res.is_silent_fallback
    assert not (tmp_path / "n.wav").exists()
    res2 = HindiTTSService(backend="mms-hin").synthesize_script("नमस्ते।", tmp_path / "m.wav")
    assert not res2.success and not (tmp_path / "m.wav").exists()


def test_silent_only_when_explicit(tmp_path):
    res = HindiTTSService(backend="silent").synthesize_script("नमस्ते दुनिया।", tmp_path / "s.wav")
    assert res.success and res.is_silent_fallback and res.backend == "silent"


def test_pipeline_refuses_silent_when_real_tts_requested(tmp_path, monkeypatch):
    import app.pipeline as pl

    def fake_synth(self, script, out, pause_seconds=0.35):
        tts_mod._write_silent_wav(out, 1.0, 24000)
        return TTSResult(success=True, audio_path=out, duration_seconds=1.0, backend="silent-fallback",
                         is_silent_fallback=True, error="x")

    monkeypatch.setattr(HindiTTSService, "synthesize_script", fake_synth)
    pipe = pl.Pipeline(data_dir=tmp_path / "data")
    with pytest.raises(RuntimeError, match="SILENT"):
        pipe.run("नमस्ते दुनिया। यह एक परीक्षण है।", tts_backend="indic-parler")


def test_hf_token_from_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_abc")
    assert tts_mod.hf_token() == "hf_abc"
