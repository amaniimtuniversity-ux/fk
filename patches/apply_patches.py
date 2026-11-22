"""Apply FreeVideo T4/P100 patches to a fresh clone.

Usage:
  python patches/apply_patches.py /path/to/FreeVideo
  FREEVVIDEO_T4_P100=1 python -m freevideo_engine.cli generate ...
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PATCH = Path(__file__).with_name("freevideo_t4_p100.patch")


def main() -> int:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("FreeVideo")
    if not (repo / "freevideo_engine" / "hardware.py").exists():
        print(f"ERROR: {repo} does not look like a FreeVideo clone.")
        return 1
    r = subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=str(repo), capture_output=True, text=True)
    if r.returncode != 0:
        print("Patch check failed (already applied or upstream changed):")
        print(r.stderr)
        # Try reverse-check to detect already-applied.
        r2 = subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=str(repo), capture_output=True, text=True)
        if r2.returncode == 0:
            print("Patch is already applied. Nothing to do.")
            return 0
        return 1
    r = subprocess.run(["git", "apply", str(PATCH)], cwd=str(repo), capture_output=True, text=True)
    print(r.stdout, r.stderr)
    print("Patches applied. Set FREEVVIDEO_T4_P100=1 for the FP16 fallback path.")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
