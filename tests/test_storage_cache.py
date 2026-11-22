"""HF cache accounting + cache-dir selection (the '29GB exceeds free disk' false rejection)."""
from __future__ import annotations

import os
from pathlib import Path

from app.kaggle_storage import cached_bytes, choose_cache_dir, hf_hub_cache_dir, repo_cache_dir
from app.registry import ModelCompatibilityChecker, VideoModelRegistry


def _fake_cache(hub: Path, repo: str, complete_mb: int, partial_mb: int = 0) -> None:
    blobs = repo_cache_dir(repo, hub) / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "aaa").write_bytes(b"\0" * complete_mb * 1024 * 1024)
    if partial_mb:
        (blobs / "bbb.incomplete").write_bytes(b"\0" * partial_mb * 1024 * 1024)


def test_cached_bytes_counts_complete_blobs_only(tmp_path):
    hub = tmp_path / "hub"
    _fake_cache(hub, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", complete_mb=3, partial_mb=2)
    assert cached_bytes("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", hub) == 3 * 1024 * 1024
    assert cached_bytes("not/cached", hub) == 0


def test_checker_subtracts_cached_bytes_from_disk_need():
    reg = VideoModelRegistry(ModelCompatibilityChecker((7, 5), 15e9, 29e9, 10**18))
    wan = reg.get("wan2.1-1.3b")
    # nothing cached, only 20GB free -> rejected (29GB needed), message names the path
    c = ModelCompatibilityChecker((7, 5), 15e9, 29e9, 20e9, cache_lookup=lambda r: 0, disk_path="/kaggle/working")
    r = c.check(wan)
    assert not r["compatible"] and "/kaggle/working" in r["reasons"][0]
    # fully cached already -> 0 more needed -> compatible even with 1GB free
    c = ModelCompatibilityChecker((7, 5), 15e9, 29e9, 1e9, cache_lookup=lambda r: 29e9)
    assert c.check(wan)["compatible"]
    # 20GB cached, 10GB free -> needs 9GB more -> compatible
    c = ModelCompatibilityChecker((7, 5), 15e9, 29e9, 10e9, cache_lookup=lambda r: 20e9)
    assert c.check(wan)["compatible"]


def test_choose_cache_dir_keeps_current_when_enough_space(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    chosen, free, notes = choose_cache_dir(1, apply_env=True)
    assert chosen == tmp_path / "hub" and free > 0
    assert not any("relocating" in n for n in notes)
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "hub")


def test_choose_cache_dir_relocates_when_impossible(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    # ask for more than any disk has -> picks the candidate with the MOST free space and says so
    chosen, free, notes = choose_cache_dir(10**18, apply_env=False)
    assert chosen in {Path(p) for p in [tmp_path / "hub", "/kaggle/tmp/.hf_cache/hub", "/tmp/.hf_cache/hub",
                                         Path.cwd() / ".hf_cache" / "hub"]}
    assert hf_hub_cache_dir() == tmp_path / "hub"  # apply_env=False leaves env untouched
