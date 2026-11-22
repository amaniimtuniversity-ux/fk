"""Launch the web UI locally or on Kaggle."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SETTINGS
from app.server import run

if __name__ == "__main__":
    print(f"FreeVideo-Kaggle web UI on http://{SETTINGS.host}:{SETTINGS.port}")
    run(SETTINGS.host, SETTINGS.port)
