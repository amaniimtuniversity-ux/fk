"""ScriptManager: the pasted script is the source of truth.

- Original script stored VERBATIM (never rewritten/shortened/summarized).
- SHA-256 hash recorded; narration must correspond to the exact original.
- Each video gets its own project ID, script hash, scene plan, output dir.
- Pronunciation fixes (e.g. adding spaces) are applied to a COPY for TTS
  only; the original is never mutated.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScriptProject:
    project_id: str
    script_original: str
    script_hash: str
    created_at: str
    language: str = "hi"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "script_hash": self.script_hash,
            "created_at": self.created_at,
            "language": self.language,
            "char_count": len(self.script_original),
            "word_count": len(self.script_original.split()),
            "metadata": self.metadata,
        }


def hash_script(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_sentences(text: str) -> list[str]:
    """Split Hindi/Hinglish script into narration sentences (punctuation-aware).

    Handles Devanagari danda (।), ., ?, !. Never drops text: joining the
    result with spaces reproduces the input modulo whitespace.
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[।.?!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def tts_safe_copy(text: str) -> str:
    """Copy for TTS with minimal pronunciation aids. Original untouched.

    - Collapses excessive whitespace
    - Ensures sentence-final punctuation for natural pauses
    Does NOT change words, story, or meaning.
    """
    t = re.sub(r"\s+", " ", text).strip()
    if t and t[-1] not in "।.?!":
        t += "।"
    return t


class ScriptManager:
    def __init__(self, projects_dir: Path):
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def create_project(self, script: str, language: str = "hi", metadata: dict | None = None) -> ScriptProject:
        if not script or not script.strip():
            raise ValueError("Script is empty. Paste your Hindi/Hinglish script first.")
        import datetime

        project_id = f"proj_{uuid.uuid4().hex[:12]}"
        project = ScriptProject(
            project_id=project_id,
            script_original=script,  # verbatim
            script_hash=hash_script(script),
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            language=language,
            metadata=metadata or {},
        )
        pdir = self.projects_dir / project_id
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "script_original.txt").write_text(script, encoding="utf-8")
        (pdir / "project.json").write_text(json.dumps(project.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return project

    def load_project(self, project_id: str) -> ScriptProject:
        pdir = self.projects_dir / project_id
        meta = json.loads((pdir / "project.json").read_text(encoding="utf-8"))
        script = (pdir / "script_original.txt").read_text(encoding="utf-8")
        if hash_script(script) != meta["script_hash"]:
            raise ValueError(f"Script hash mismatch for {project_id}: file was modified.")
        return ScriptProject(
            project_id=project_id, script_original=script,
            script_hash=meta["script_hash"], created_at=meta["created_at"],
            language=meta.get("language", "hi"), metadata=meta.get("metadata", {}),
        )

    def verify_narration(self, project: ScriptProject, narration_text: str) -> bool:
        """Check narration corresponds to the original (whitespace-insensitive)."""
        norm = lambda s: re.sub(r"\s+", " ", s).strip()
        return norm(narration_text) == norm(project.script_original)
