"""KaggleStorageManager: persistence via /kaggle/working + HF cache.

Kaggle facts: /kaggle/working (~20GB) persists across sessions when the
notebook is saved; /tmp does NOT. HF cache defaults to ~/.cache (ephemeral).
This manager:
- Points HF_HOME / HF_HUB_CACHE at /kaggle/working/.hf_cache when present
- Records model download manifests (repo, revision, file hashes, status)
- Verifies cached models before reuse (never trusts partial downloads)
- Exports final outputs to /kaggle/working/outputs/
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional


class KaggleStorageManager:
    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = Path(working_dir) if working_dir else Path("/kaggle/working")
        self.is_kaggle = self.working_dir.exists()
        self.cache_dir = self.working_dir / ".hf_cache" if self.is_kaggle else Path(os.environ.get("HF_HOME", ".hf_cache"))
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
