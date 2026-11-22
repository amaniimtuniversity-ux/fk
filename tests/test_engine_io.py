"""Engine I/O: diffusers loader variant fallback, frame normalisation/validation, MP4 writing."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from app.engines.base import (frames_raw_stats, load_diffusers_pipeline, normalize_frames,
                              validate_frames)

DIFFUSERS_MSG = ("You are trying to load model files of the `variant=fp16`, "
                 "but no such modeling files are available.")


class _FakePipeline:
    calls: list[dict] = []
    has_fp16 = False

    @classmethod
    def from_pretrained(cls, repo, **kw):
        cls.calls.append(kw)
        if kw.get("variant") == "fp16" and not cls.has_fp16:
            raise ValueError(DIFFUSERS_MSG)
        return ("pipe", repo, kw.get("torch_dtype"), kw.get("variant"))


def test_loader_falls_back_when_fp16_variant_missing():
    _FakePipeline.calls.clear(); _FakePipeline.has_fp16 = False
    out = load_diffusers_pipeline(_FakePipeline, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "f16")
    assert out == ("pipe", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "f16", None)
    assert [c.get("variant") for c in _FakePipeline.calls] == ["fp16", None]


def test_loader_uses_fp16_variant_when_available():
    _FakePipeline.calls.clear(); _FakePipeline.has_fp16 = True
    out = load_diffusers_pipeline(_FakePipeline, "stabilityai/svd-xt", "f16")
    assert out[3] == "fp16" and len(_FakePipeline.calls) == 1


def test_loader_reraises_unrelated_errors():
    class Broken:
        @staticmethod
        def from_pretrained(repo, **kw):
            raise ValueError("CUDA out of memory")

    with pytest.raises(ValueError, match="out of memory"):
        load_diffusers_pipeline(Broken, "x", "f16")


def _gradient(f=4, h=16, w=24):
    y = np.linspace(0, 1, h)[:, None, None]
    x = np.linspace(0, 1, w)[None, :, None]
    base = np.concatenate([y + 0 * x, x + 0 * y, 0.5 * (x + y)], axis=-1)
    return np.stack([np.clip(base + i * 0.05, 0, 1) for i in range(f)]).astype(np.float32)


def test_normalize_float01_and_uint8_and_chw():
    g = _gradient()
    a = normalize_frames(g)
    assert a.dtype == np.uint8 and a.shape == (4, 16, 24, 3) and a.max() == 255
    assert np.array_equal(normalize_frames(a), a)                  # uint8 passthrough
    assert normalize_frames(np.transpose(g, (0, 3, 1, 2))).shape == (4, 16, 24, 3)  # [F,3,H,W]
    assert normalize_frames(g[None]).shape == (4, 16, 24, 3)        # [1,F,H,W,3]
    assert normalize_frames(list(g)).shape == (4, 16, 24, 3)        # list of frames


def test_normalize_pil_and_torch_like():
    from PIL import Image

    g = (_gradient() * 255).astype(np.uint8)
    pil = [Image.fromarray(f) for f in g]
    assert np.array_equal(normalize_frames(pil), g)

    class T:  # minimal torch-tensor-like object
        def __init__(self, a): self.a = a
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return self.a

    assert normalize_frames(T(_gradient())).shape == (4, 16, 24, 3)


def test_validate_rejects_nan_and_black():
    g = _gradient()
    validate_frames(normalize_frames(g))  # ok
    bad = g.copy(); bad[1, 2, 3, 0] = np.nan
    with pytest.raises(RuntimeError, match="NaN"):
        validate_frames(bad)
    with pytest.raises(RuntimeError, match="uniform"):
        validate_frames(np.zeros((4, 16, 24, 3), dtype=np.uint8))
    assert frames_raw_stats(np.zeros((2, 4, 4, 3), np.uint8))["std"] == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_save_frames_keeps_exact_resolution(tmp_path):
    pytest.importorskip("imageio")
    from app.engines.wan_adapter import WanAdapter

    eng = WanAdapter("repo", {"compute_dtype": "float16"}, {"backend": "sdpa"}, "sequential")
    out = tmp_path / "clip.mp4"
    eng._save_frames(_gradient(f=9, h=96, w=160), out, fps=8)   # 96x160 not multiples of 16
    assert out.exists() and out.stat().st_size > 1024
    info = subprocess.run(["ffmpeg", "-i", str(out)], capture_output=True, text=True).stderr
    assert "160x96" in info, info   # NOT silently resized to 160x96->160x96 macro-block multiple
    assert "h264" in info
