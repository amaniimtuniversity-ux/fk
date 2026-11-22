"""Tests: checkpoint resume (Test G), registry compatibility, memory adapt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.checkpoint import CheckpointManager
from app.memory import MemoryManager
from app.registry import ModelCompatibilityChecker, VideoModelRegistry


def test_checkpoint_resume_skips_completed(tmp_path):
    ckpt = CheckpointManager(tmp_path / "proj")
    ckpt.scene_path(0).write_bytes(b"fake-mp4-data")
    ckpt.save_scene(0, {"success": True})
    assert ckpt.is_scene_complete(0) is True
    assert ckpt.is_scene_complete(1) is False
    assert ckpt.completed_scenes(3) == [0]


def test_checkpoint_partial_cleanup(tmp_path):
    ckpt = CheckpointManager(tmp_path / "proj")
    ckpt.scene_path(0).write_bytes(b"")  # zero-byte partial
    assert ckpt.cleanup_partial() == 1
    assert not ckpt.scene_path(0).exists()


def test_registry_recommends_wan_on_t4():
    # Kaggle T4: 15GB VRAM, 29GB RAM, model cache on /kaggle/tmp (~60GB free)
    checker = ModelCompatibilityChecker((7, 5), 15 * 10**9, 29 * 10**9, 60 * 10**9, cache_lookup=lambda r: 0)
    reg = VideoModelRegistry(checker)
    assert reg.recommend().id == "wan2.1-1.3b"


def test_registry_rejects_wan_when_cache_on_20gb_working_quota():
    # Wan2.1-Diffusers is 28.9GB on disk: it must never be downloaded into the 20GB /kaggle/working
    checker = ModelCompatibilityChecker((7, 5), 15 * 10**9, 29 * 10**9, 20 * 10**9, cache_lookup=lambda r: 0)
    reg = VideoModelRegistry(checker)
    res = checker.check(reg.get("wan2.1-1.3b"))
    assert not res["compatible"] and any("more download" in r for r in res["reasons"])


def test_h3_incompatible_on_kaggle():
    checker = ModelCompatibilityChecker((7, 5), 16 * 10**9, 29 * 10**9, 20 * 10**9)
    reg = VideoModelRegistry(checker)
    h3 = reg.get("freevideo-h3")
    result = checker.check(h3)
    assert result["compatible"] is False
    assert len(result["reasons"]) >= 2  # capability + RAM/disk


def test_memory_adapt_warns_on_reduction():
    mem = MemoryManager()
    w, h, f, notes = mem.adapt_canvas(832, 480, 49, vram_free_bytes=5 * 10**9)
    assert (w, h, f) != (832, 480, 49)
    assert len(notes) >= 1 and len(mem.warnings) >= 1


def test_memory_adapt_no_change_when_room():
    mem = MemoryManager()
    w, h, f, notes = mem.adapt_canvas(832, 480, 49, vram_free_bytes=14 * 10**9)
    assert (w, h, f) == (832, 480, 49)
    assert notes == []
