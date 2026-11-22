"""YouTubeService: upload finished video via YouTube Data API v3.

Requires OAuth client secrets (client_secrets.json) + token refresh.
Secrets via env: YOUTUBE_CLIENT_SECRETS, YOUTUBE_TOKEN. Never hardcoded.
Upload is resumable via google-api-python-client when available.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class YouTubeService:
    def __init__(self, client_secrets: str = "", token_path: str = ""):
        self.client_secrets = client_secrets or os.environ.get("YOUTUBE_CLIENT_SECRETS", "client_secrets.json")
        self.token_path = token_path or os.environ.get("YOUTUBE_TOKEN", "youtube_token.json")

    @property
    def configured(self) -> bool:
        return Path(self.client_secrets).exists()

    def upload(self, video: Path, title: str, description: str = "",
               privacy: str = "unlisted", tags: list[str] | None = None) -> dict[str, Any]:
        if not self.configured:
            return {"success": False, "error": f"Client secrets not found: {self.client_secrets}. See KAGGLE_GUIDE.md."}
        if not video.exists():
            return {"success": False, "error": f"Video not found: {video}"}
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            scopes = ["https://www.googleapis.com/auth/youtube.upload"]
            creds = None
            if Path(self.token_path).exists():
                creds = Credentials.from_authorized_user_file(self.token_path, scopes)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets, scopes)
                    creds = flow.run_local_server(port=0)
                Path(self.token_path).write_text(creds.to_json(), encoding="utf-8")
            youtube = build("youtube", "v3", credentials=creds)
            body = {
                "snippet": {"title": title[:100], "description": description[:5000],
                            "tags": tags or [], "categoryId": "22"},
                "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
            }
            media = MediaFileUpload(str(video), mimetype="video/mp4", resumable=True)
            req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
            resp = None
            while resp is None:
                status, resp = req.next_chunk()
            return {"success": True, "video_id": resp.get("id"), "response": resp}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
