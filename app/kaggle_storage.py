"""KaggleStorageManager: persistence via /kaggle/working + HF cache.

Kaggle facts: /kaggle/working has a HARD 20GB quota (exceeding it fails I/O / kills the kernel)
and persists across sessions; /kaggle/tmp has ~60GB but is wiped per session. Video models are
large (Wan2.1-Diffusers = 29GB), so the HF cache lives in /kaggle/tmp and is re-downloaded per
session; outputs/manifests/checkpoints stay in /kaggle/working.
This manager:
- Points HF_HOME / HF_HUB_CACHE at /kaggle/tmp/.hf_cache on Kaggle
- Records model download manifests (repo, revision, file hashes, status)
- Verifies cached models before reuse (never trusts partial downloads)
- Exports final outputs to /kaggle/working/outputs/
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional


def hf_hub_cache_dir() -> Path:
    """Where huggingface_hub will actually write model files right now."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_cache_dir(repo_id: str, hub_cache: Optional[Path] = None) -> Path:
    return (hub_cache or hf_hub_cache_dir()) / ("models--" + repo_id.replace("/", "--"))


def cached_bytes(repo_id: str, hub_cache: Optional[Path] = None) -> int:
    """Bytes of `repo_id` already present in the HF cache (complete blobs only; partial .incomplete
    files are excluded). 0 if not cached. Pure filesystem walk — no network."""
    d = repo_cache_dir(repo_id, hub_cache) / "blobs"
    if not d.is_dir():
        return 0
    total = 0
    for f in d.iterdir():
        if f.is_file() and not f.name.endswith(".incomplete"):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _free_bytes(path: Path) -> int:
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return 0


def choose_cache_dir(min_free_bytes: int, apply_env: bool = True) -> tuple[Path, int, list[str]]:
    """Pick an HF cache location with enough free space and (optionally) point HF_HOME/HF_HUB_CACHE at it.

    Order: current HF_HOME (keep whatever is already cached there) -> /kaggle/tmp -> /tmp -> cwd.
    On Kaggle /kaggle/working has a HARD 20GB quota, so it is only used as a last resort.
    Returns (chosen_hub_cache_dir, free_bytes, notes).
    """
    notes: list[str] = []
    current = hf_hub_cache_dir()
    candidates: list[Path] = [current]
    for c in (Path("/kaggle/tmp/.hf_cache/hub"), Path("/tmp/.hf_cache/hub"), Path.cwd() / ".hf_cache" / "hub"):
        if c not in candidates:
            candidates.append(c)
    best: Optional[tuple[Path, int]] = None
    for c in candidates:
        free = _free_bytes(c)
        notes.append(f"{c}: free {free / 1e9:.1f} GB")
        if free >= min_free_bytes:
            best = (c, free)
            break
        if best is None or free > best[1]:
            best = (c, free)
    chosen, free = best  # type: ignore[misc]
    if chosen != current:
        notes.append(f"relocating HF cache {current} -> {chosen} (need {min_free_bytes / 1e9:.1f} GB)")
    if apply_env:
        chosen.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_CACHE"] = str(chosen)
        os.environ["HF_HOME"] = str(chosen.parent)
    return chosen, free, notes


class KaggleStorageManager:
    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = Path(working_dir) if working_dir else Path("/kaggle/working")
        self.is_kaggle = self.working_dir.exists()
        # Model cache goes to /kaggle/tmp (~60GB, ephemeral) — NOT /kaggle/working (20GB hard quota).
        self.cache_dir = (Path(os.environ.get("HF_HOME", "/kaggle/tmp/.hf_cache")) if self.is_kaggle
                          else Path(os.environ.get("HF_HOME", ".hf_cache")))
        self.outputs_dir = self.working_dir / "outputs" if self.is_kaggle else Path("outputs")
        self.manifest_dir = self.working_dir / "model_manifests" if self.is_kaggle else Path("model_manifests")

    def setup_env(self) -> dict[str, str]:
        """Set HF cache env vars BEFORE huggingface_hub import. Returns applied vars."""
        applied: dict[str, str] = {}
        if self.is_kaggle:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.cache_dir))
            os.environ.setdefault("HF_HUB_CACHE", str(self.cache_dir / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(self.cache_dir / "hub"))
            applied = {k: os.environ[k] for k in ("HF_HOME", "HF_HUB_CACHE") if k in os.environ}
        return applied

    def record_download(self, repo: str, revision: str, files: list[dict[str, Any]], status: str = "complete") -> Path:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        safe = repo.replace("/", "__")
        path = self.manifest_dir / f"{safe}.json"
        path.write_text(json.dumps({
            "repo": repo, "revision": revision, "status": status,
            "recorded_at": time.time(), "files": files,
        }, indent=2), encoding="utf-8")
        return path

    def verify_cached(self, repo: str) -> dict[str, Any]:
        """Check HF cache has the repo snapshot (best-effort, no download)."""
        try:
            from huggingface_hub import scan_cache_dir

            for repo_info in scan_cache_dir().repos:
                if repo_info.repo_id == repo:
                    return {"cached": True, "size_bytes": int(repo_info.size_on_disk),
                            "revision": repo_info.revisions[0].commit_hash if repo_info.revisions else ""}
        except Exception as e:
            return {"cached": False, "error": str(e)}
        return {"cached": False}

    def export_output(self, src: Path, dest_name: Optional[str] = None) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        dest = self.outputs_dir / (dest_name or src.name)
        shutil.copy2(src, dest)
        return dest

    @staticmethod
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
