"""Expose the local web UI through a free Cloudflare quick tunnel (no account needed).

Kaggle (unlike Colab) has NO built-in port preview, so a server on 127.0.0.1:5000 is
unreachable from your browser. This starts `cloudflared tunnel --url http://127.0.0.1:PORT`
and prints the public https://*.trycloudflare.com URL.

Usage (notebook):
    from scripts.tunnel import start_tunnel
    url = start_tunnel(5000)

Usage (CLI):
    python scripts/tunnel.py --port 5000
"""
from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_PROC: subprocess.Popen | None = None
_LOG: list[str] = []


def ensure_cloudflared(dest: Path | None = None) -> Path:
    dest = dest or Path(os.environ.get("CLOUDFLARED_BIN", "/kaggle/working/cloudflared"
                                        if Path("/kaggle/working").is_dir() else "/tmp/cloudflared"))
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        print(f"Downloading cloudflared -> {dest}")
        urllib.request.urlretrieve(CF_URL, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def _pump(proc: subprocess.Popen) -> None:
    for line in proc.stdout:  # type: ignore[union-attr]
        _LOG.append(line.rstrip())
        if len(_LOG) > 500:
            del _LOG[:250]


def start_tunnel(port: int = 5000, timeout: float = 60.0) -> str:
    """Start a quick tunnel and return its public URL. Idempotent per process."""
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        for line in _LOG:
            m = _URL_RE.search(line)
            if m:
                return m.group(0)
    binary = ensure_cloudflared()
    _LOG.clear()
    _PROC = subprocess.Popen(
        [str(binary), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    threading.Thread(target=_pump, args=(_PROC,), daemon=True).start()
    deadline = time.time() + timeout
    while time.time() < deadline:
        for line in _LOG:
            m = _URL_RE.search(line)
            if m:
                return m.group(0)
        if _PROC.poll() is not None:
            break
        time.sleep(0.5)
    tail = "\n".join(_LOG[-15:])
    raise RuntimeError(f"cloudflared did not report a URL within {timeout}s. Log tail:\n{tail}")


def stop_tunnel() -> None:
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        _PROC.terminate()
    _PROC = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    url = start_tunnel(args.port)
    print("PUBLIC URL:", url)
    try:
        while _PROC is not None and _PROC.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_tunnel()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
