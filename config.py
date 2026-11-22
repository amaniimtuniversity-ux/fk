"""Centralized configuration. All secrets via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # Storage
    data_dir: Path = field(default_factory=lambda: Path(_env("FV_DATA_DIR", str(ROOT / "data"))))
    kaggle_working: Path = field(default_factory=lambda: Path("/kaggle/working"))
    # Kaggle: /kaggle/working has a hard 20GB quota (exceeding it kills the kernel) while /kaggle/tmp has
    # ~60GB ephemeral space. Wan2.1-Diffusers alone is 29GB, so the HF cache must live in /kaggle/tmp.
    hf_cache_dir: Path = field(default_factory=lambda: Path(_env("HF_HOME", "/kaggle/tmp/.hf_cache" if Path("/kaggle/working").exists() else str(ROOT / ".hf_cache"))))
    model_cache_dir: Path = field(default_factory=lambda: Path(_env("FV_MODEL_CACHE", str(ROOT / "models"))))
    freevideo_repo: Path = field(default_factory=lambda: Path(_env("FREEVVIDEO_REPO", str(ROOT.parent / "FreeVideo"))))

    # Generation defaults (never silently reduced — see pipeline warnings)
    default_engine: str = field(default_factory=lambda: _env("FV_ENGINE", "wan2.1-1.3b"))
    default_profile: str = field(default_factory=lambda: _env("FV_PROFILE", "balanced"))
    default_width: int = field(default_factory=lambda: _env_int("FV_WIDTH", 832))
    default_height: int = field(default_factory=lambda: _env_int("FV_HEIGHT", 480))
    default_frames: int = field(default_factory=lambda: _env_int("FV_FRAMES", 49))
    default_fps: int = field(default_factory=lambda: _env_int("FV_FPS", 24))
    default_steps: int = field(default_factory=lambda: _env_int("FV_STEPS", 30))
    default_guidance: float = field(default_factory=lambda: float(_env("FV_GUIDANCE", "6.0")))
    default_seed: int = field(default_factory=lambda: _env_int("FV_SEED", 42))

    # TTS
    tts_backend: str = field(default_factory=lambda: _env("FV_TTS_BACKEND", "indic-parler"))
    tts_model: str = field(default_factory=lambda: _env("FV_TTS_MODEL", "ai4bharat/indic-parler-tts"))
    tts_voice_description: str = field(
        default_factory=lambda: _env(
            "FV_TTS_VOICE",
            "Divya's voice is expressive and warm in delivery, with a very close recording that almost has no background noise.",
        )
    )
    tts_sample_rate: int = field(default_factory=lambda: _env_int("FV_TTS_SR", 24000))

    # Server
    host: str = field(default_factory=lambda: _env("FV_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("FV_PORT", 5000))

    # Secrets (never hardcoded)
    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))
    youtube_client_secrets: str = field(default_factory=lambda: _env("YOUTUBE_CLIENT_SECRETS", "client_secrets.json"))
    youtube_token: str = field(default_factory=lambda: _env("YOUTUBE_TOKEN", "youtube_token.json"))

    # Behavior
    allow_paid_apis: bool = field(default_factory=lambda: _env_bool("FV_ALLOW_PAID_APIS", False))
    max_scenes: int = field(default_factory=lambda: _env_int("FV_MAX_SCENES", 24))
    scene_target_seconds: float = field(default_factory=lambda: float(_env("FV_SCENE_SECONDS", "5.0")))

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.model_cache_dir, self.hf_cache_dir):
            p.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()

# Generation profiles: (width, height, frames, steps, offload, vae_tiling)
PROFILES = {
    "max_quality": {"width": 832, "height": 480, "frames": 81, "steps": 40, "offload": "sequential", "vae_tiling": True, "note": "Best settings that fit T4 16GB with sequential offload. Slower."},
    "balanced": {"width": 832, "height": 480, "frames": 49, "steps": 30, "offload": "sequential", "vae_tiling": True, "note": "Default. ~5s clips at 24fps."},
    "low_vram": {"width": 640, "height": 360, "frames": 33, "steps": 25, "offload": "sequential", "vae_tiling": True, "note": "Smaller canvas for P100 / tight VRAM. User is warned; quality reduced by resolution only."},
    "cpu_assisted": {"width": 640, "height": 360, "frames": 25, "steps": 20, "offload": "sequential", "vae_tiling": True, "note": "Most conservative. Slowest, most reliable on P100."},
}
