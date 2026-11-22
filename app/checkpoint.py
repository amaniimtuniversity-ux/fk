"""CheckpointManager: scene/render-level checkpoints + resume.

Layout per project:
  projects/<id>/
    project.json, script_original.txt, scene_plan.json
    narration.wav, narration_manifest.json
    scenes/scene_000.mp4 ... + scene_000.json (per-scene manifest)
    final.mp4, final_manifest.json
    job.json (status, progress, timestamps)

Interruption recovery: completed scenes (file exists + manifest valid) are
skipped on resume. Partial files are deleted, never trusted.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckpointManager:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_dir = self.project_dir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

    # -- job state --
    def set_status(self, status: str, extra: dict | None = None) -> None:
        job_path = self.project_dir / "job.json"
        job: dict[str, Any] = {}
        if job_path.exists():
            try:
                job = json.loads(job_path.read_text(encoding="utf-8"))
            except Exception:
                job = {}
        job.update({"status": status, "updated_at": time.time(), **(extra or {})})
        job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_status(self) -> dict[str, Any]:
        p = self.project_dir / "job.json"
        if not p.exists():
            return {"status": "new"}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "corrupt"}

    # -- scene checkpoints --
    def scene_path(self, index: int) -> Path:
        return self.scenes_dir / f"scene_{index:03d}.mp4"

    def scene_manifest_path(self, index: int) -> Path:
        return self.scenes_dir / f"scene_{index:03d}.json"

    def is_scene_complete(self, index: int) -> bool:
        mp4 = self.scene_path(index)
        man = self.scene_manifest_path(index)
        if not mp4.exists() or not man.exists():
            return False
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
            if not data.get("success"):
                return False
            if mp4.stat().st_size == 0:
                return False
            # Verify hash if recorded.
            if data.get("sha256") and _sha256_file(mp4) != data["sha256"]:
                return False
            return True
        except Exception:
            return False

    def save_scene(self, index: int, result_dict: dict[str, Any]) -> None:
        mp4 = self.scene_path(index)
        if mp4.exists() and mp4.stat().st_size > 0:
            result_dict = dict(result_dict)
            result_dict["sha256"] = _sha256_file(mp4)
        self.scene_manifest_path(index).write_text(
            json.dumps(result_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def completed_scenes(self, total: int) -> list[int]:
        return [i for i in range(total) if self.is_scene_complete(i)]

    def cleanup_partial(self) -> int:
        """Delete zero-byte / manifest-less partial scene files. Returns count."""
        removed = 0
        for mp4 in self.scenes_dir.glob("scene_*.mp4"):
            idx = int(mp4.stem.split("_")[1])
            if mp4.stat().st_size == 0 or not self.is_scene_complete(idx):
                # Only delete if manifest says failed/missing AND file is partial.
                # Completed scenes are never deleted here.
                if not self.is_scene_complete(idx):
                    try:
                        mp4.unlink()
                        removed += 1
                    except Exception:
                        pass
        return removed

    def save_json(self, name: str, data: dict[str, Any]) -> Path:
        p = self.project_dir / name
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def load_json(self, name: str) -> Optional[dict[str, Any]]:
        p = self.project_dir / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
