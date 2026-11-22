"""TelegramService: send finished video via Telegram Bot API.

Secrets via env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID. Never hardcoded.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class TelegramService:
    def __init__(self, bot_token: str = "", chat_id: str = ""):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_video(self, video: Path, caption: str = "") -> dict[str, Any]:
        if not self.configured:
            return {"success": False, "error": "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set."}
        if not video.exists():
            return {"success": False, "error": f"Video not found: {video}"}
        try:
            import requests

            url = f"https://api.telegram.org/bot{self.bot_token}/sendVideo"
            with open(video, "rb") as f:
                r = requests.post(
                    url,
                    data={"chat_id": self.chat_id, "caption": caption[:1024]},
                    files={"video": (video.name, f, "video/mp4")},
                    timeout=600,
                )
            data = r.json()
            return {"success": bool(data.get("ok")), "response": data}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
