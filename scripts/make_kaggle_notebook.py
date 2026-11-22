"""Generate the Kaggle notebook (.ipynb) from the project sources.

Usage: python scripts/make_kaggle_notebook.py [--out notebooks/kaggle_freevideo_t4.ipynb]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


CELL1 = '# Cell 1 — Install deps (robust: each step independent; Kaggle\'s preinstalled CUDA torch is used as-is)\n# Run this FIRST in a fresh kernel. If any line below says FAIL/MISSING, stop and fix before Cell 2.\nimport subprocess, sys, importlib\ndef pip(*args):\n    r = subprocess.run([sys.executable, \'-m\', \'pip\', \'install\', \'-q\', *args], capture_output=True, text=True)\n    print((\'OK  \' if r.returncode == 0 else \'FAIL\') + \'  pip install \' + \' \'.join(args))\n    if r.returncode != 0: print(\'     \', (r.stderr or r.stdout).strip().splitlines()[-1][:200])\n    return r.returncode == 0\nsubprocess.run(\'apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1\', shell=True); print(\'OK    ffmpeg\')\nimport torch; print(f\'torch {torch.__version__} | cuda {torch.cuda.is_available()} | \'\n                    f\'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"} | \'\n                    f\'sm{"".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else "-"}\')\n# Core video stack. transformers is pinned to 4.46.1 because Parler-TTS (best Hindi TTS) requires exactly that;\n# diffusers 0.33.1 (first release with WanPipeline) works with it.\npip(\'transformers==4.46.1\', \'diffusers==0.33.1\', \'accelerate>=1.1\', \'safetensors\', \'sentencepiece\', \'protobuf\',\n    \'huggingface_hub>=0.26\', \'soundfile\', \'imageio[ffmpeg]\', \'flask\', \'psutil\', \'av\', \'einops\', \'ftfy\')\n# Parler-TTS (optional; gated model). --no-deps so it cannot drag torch/transformers around; DAC codec separately.\nparler_ok = pip(\'--no-deps\', \'parler-tts==0.2.3\') and pip(\'descript-audio-codec\')\n# protobuf: some of the above downgrade it, which breaks Kaggle\'s preinstalled *_pb2 modules\n# ("cannot import name \'runtime_version\' from \'google.protobuf\'") and with them transformers itself.\npip(\'--upgrade\', \'protobuf>=5.29,<7\')\nimportlib.invalidate_caches()\ntry:\n    import google.protobuf; from google.protobuf import runtime_version  # noqa: F401\n    print(\'OK    protobuf\', google.protobuf.__version__)\nexcept Exception as e:\n    print(\'FAIL  protobuf still broken ->\', e, \'\\n      Run: !pip install -U "protobuf>=5.29,<7"  then Run -> Restart kernel\')\nimport transformers, diffusers; print(f\'transformers {transformers.__version__} | diffusers {diffusers.__version__}\')\nfor mod, label in ((\'diffusers.WanPipeline\', \'Wan2.1 pipeline\'), (\'transformers.VitsModel\', \'MMS Hindi TTS\'),\n                   (\'parler_tts.ParlerTTSForConditionalGeneration\', \'Indic Parler-TTS\')):\n    m, a = mod.rsplit(\'.\', 1)\n    try: getattr(importlib.import_module(m), a); print(f\'READY  {label}\')\n    except Exception as e: print(f\'MISSING {label}: {type(e).__name__}: {str(e)[:120]}\')\nprint(\'If "Indic Parler-TTS" is MISSING the pipeline uses MMS Hindi automatically (recorded in the manifest).\')'

CELL4 = '# Cell 4 — Model cache on the BIG disk + pre-download models (visible progress)\n# /kaggle/working has a hard 20 GB quota (exceeding it kills the kernel). /kaggle/tmp lives on the\n# large root disk (hundreds of GB+, wiped per session). Wan2.1-Diffusers is 28.9 GB -> MUST go to /kaggle/tmp.\nimport os, shutil\nos.makedirs(\'/kaggle/tmp/.hf_cache/hub\', exist_ok=True)\nos.environ[\'HF_HOME\'] = \'/kaggle/tmp/.hf_cache\'\nos.environ[\'HF_HUB_CACHE\'] = \'/kaggle/tmp/.hf_cache/hub\'\nos.environ[\'HF_HUB_ENABLE_HF_TRANSFER\'] = \'0\'\nos.environ[\'FREEVVIDEO_REPO\'] = \'/kaggle/working/FreeVideo\'\nfor p in (\'/kaggle/working\', \'/kaggle/tmp\'):\n    u = shutil.disk_usage(p); print(f\'{p:16s} free {u.free/1e9:7.1f} GB\')\nprint(\'HF cache ->\', os.environ[\'HF_HOME\'])\n\n# --- HF token: needed ONLY for the gated Indic Parler-TTS (Wan2.1 and MMS Hindi are open) ---\n# 1) click "Agree" once at https://huggingface.co/ai4bharat/indic-parler-tts\n# 2) https://huggingface.co/settings/tokens -> New token (Read)\n# 3) Kaggle notebook: Add-ons -> Secrets -> name HF_TOKEN -> paste -> enable for this notebook\ntry:\n    from kaggle_secrets import UserSecretsClient\n    os.environ[\'HF_TOKEN\'] = UserSecretsClient().get_secret(\'HF_TOKEN\'); print(\'HF_TOKEN: found in Kaggle Secrets\')\nexcept Exception:\n    print(\'HF_TOKEN: not set -> Indic Parler-TTS (gated) skipped; MMS Hindi TTS (open) will be used\')\n\nPREDOWNLOAD = True           # False = skip (web UI downloads on first use, without visible progress)\nVIDEO_MODEL = \'Wan-AI/Wan2.1-T2V-1.3B-Diffusers\'   # 28.9 GB, open\nTTS_PARLER  = \'ai4bharat/indic-parler-tts\'          # 3.8 GB, GATED (best quality)\nTTS_MMS     = \'facebook/mms-tts-hin\'                # 0.15 GB, open (fallback)\nif PREDOWNLOAD:\n    from huggingface_hub import snapshot_download\n    need_gb = 29 + 4 + 3\n    if shutil.disk_usage(\'/kaggle/tmp\').free / 1e9 < need_gb:\n        raise SystemExit(f\'Not enough free space in /kaggle/tmp (need ~{need_gb} GB). Restart the session.\')\n    def _dl(repo, optional=False):\n        print(f\'\\n>>> downloading {repo} ...\', flush=True)\n        try:\n            path = snapshot_download(repo, cache_dir=os.environ[\'HF_HUB_CACHE\'], max_workers=8,\n                                     token=os.environ.get(\'HF_TOKEN\'))\n        except Exception as e:\n            if not optional:\n                raise\n            print(f\'    SKIPPED ({type(e).__name__}): {str(e)[:160]}\')\n            print(\'    -> accept the license on the HF page + add the HF_TOKEN secret to enable it. Using MMS Hindi.\')\n            return None\n        size = sum(f.stat().st_size for f in __import__(\'pathlib\').Path(path).rglob(\'*\') if f.is_file())\n        print(f\'    ready: {path}  ({size/1e9:.1f} GB)\')\n        return path\n    _dl(TTS_MMS)\n    _dl(TTS_PARLER, optional=True)\n    _dl(VIDEO_MODEL)\n    u = shutil.disk_usage(\'/kaggle/tmp\'); print(f\'\\n/kaggle/tmp free after download: {u.free/1e9:.1f} GB\')'

CELL5 = "# Cell 5 — Launch web UI + PUBLIC link (Kaggle has no port preview; we use a free Cloudflare quick tunnel)\nimport sys, threading, time, urllib.request\nsys.path.insert(0, '/kaggle/working/freevideo-kaggle')\nfrom app.server import run\nfrom scripts.tunnel import start_tunnel\nif not globals().get('_ui_thread'):\n    _ui_thread = threading.Thread(target=run, kwargs={'host': '0.0.0.0', 'port': 5000}, daemon=True)\n    _ui_thread.start(); time.sleep(2)\nurl = start_tunnel(5000)\nfor _ in range(20):  # wait for the tunnel DNS to go live (~10-20 s)\n    try:\n        urllib.request.urlopen(url + '/api/health', timeout=10); break\n    except Exception:\n        time.sleep(3)\nprint('\\n' + '=' * 70)\nprint('OPEN THIS IN YOUR BROWSER  ->', url)\nprint('=' * 70)\nprint('Link stays alive only while this notebook session is running.')\nprint('Paste your Hindi script, preview voice, then Start generation.')"

CELL6 = '# Cell 6 — (OPTIONAL, no browser needed) Generate directly from the notebook\n# Use this if the tunnel link is blocked for you. Paste your script between the triple quotes.\nSCRIPT = """\nयहाँ अपनी हिंदी स्क्रिप्ट पेस्ट करें।\n"""\nENGINE, PROFILE, STYLE, TTS = \'wan2.1\', \'balanced\', \'horror\', \'indic-parler\'\nRESUME_PROJECT_ID = None   # set to a previous project_id to resume after a session cut\n\nimport sys; sys.path.insert(0, \'/kaggle/working/freevideo-kaggle\')\nfrom app.pipeline import Pipeline\ndef _progress(stage, done, total, msg=\'\'): print(f\'[{stage}] {done}/{total} {msg}\', flush=True)\nif SCRIPT.strip() and \'अपनी हिंदी स्क्रिप्ट पेस्ट\' not in SCRIPT:\n    manifest = Pipeline(progress=_progress).run(SCRIPT.strip(), engine_id=ENGINE, profile=PROFILE,\n                                               style=STYLE, tts_backend=TTS, resume_project_id=RESUME_PROJECT_ID)\n    print(\'project_id:\', manifest[\'project_id\']); print(\'FINAL MP4:\', manifest[\'final\'])\n    print(\'seconds:\', manifest[\'seconds\'], \'| scenes:\', len(manifest[\'scenes\']))\n    from IPython.display import Video, display; display(Video(manifest[\'final\'], embed=True, width=640))\nelse:\n    print(\'Paste your script into SCRIPT above, then re-run this cell.\')'


def build() -> dict:
    cells = [
        md("# FreeVideo-Kaggle — Hindi Script to Video on T4/P100\n\n"
           "Production low-VRAM AI video pipeline. Primary engine: **Wan2.1-T2V-1.3B (FP16)**.\n"
           "FreeVideo H3 is included as an optional experimental engine (see AUDIT.md).\n\n"
           "**Steps:** 1) Install 2) Probe kernels 3) Launch web UI + public link 4) Paste script → Generate → Download. "
           "Cell 6 = generate without the browser UI.\n"
           "Settings: Accelerator = GPU T4 x2 (or P100), Internet = ON, Save version = Save."),
        code(CELL1),
        code("# Cell 2 — Get the project + upstream FreeVideo\n"
             "# >>> SET THIS to your own GitHub repo (the one you pushed freevideo-kaggle to) <<<\n"
             "PROJECT_REPO = 'https://github.com/YOUR_USER/freevideo-kaggle.git'\n"
             "# Alternative: upload the project as a Kaggle Dataset (zip or folder) and leave PROJECT_REPO as-is;\n"
             "# the cell auto-detects any /kaggle/input/*/ containing app/server.py or a freevideo-kaggle*.zip.\n"
             "import os, sys, glob, shutil, subprocess, zipfile\n"
             "WORK = '/kaggle/working'; PROJ = f'{WORK}/freevideo-kaggle'; UP = f'{WORK}/FreeVideo'\n"
             "\n"
             "def _have(p): return os.path.isfile(os.path.join(p, 'app', 'server.py'))\n"
             "\n"
             "if not _have(PROJ):\n"
             "    shutil.rmtree(PROJ, ignore_errors=True)\n"
             "    if 'YOUR_USER' not in PROJECT_REPO:\n"
             "        print('Cloning project from', PROJECT_REPO)\n"
             "        subprocess.run(['git', 'clone', '--depth', '1', PROJECT_REPO, PROJ], check=False)\n"
             "    if not _have(PROJ):\n"
             "        for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):\n"
             "            with zipfile.ZipFile(z) as zf:\n"
             "                names = zf.namelist()\n"
             "                hit = [n for n in names if n.endswith('app/server.py')]\n"
             "                if hit:\n"
             "                    print('Unzipping dataset', z); zf.extractall(f'{WORK}/_unz')\n"
             "                    src = os.path.join(f'{WORK}/_unz', hit[0][:-len('app/server.py')])\n"
             "                    shutil.copytree(src, PROJ, dirs_exist_ok=True); break\n"
             "    if not _have(PROJ):\n"
             "        for d in glob.glob('/kaggle/input/**/app/server.py', recursive=True):\n"
             "            src = os.path.dirname(os.path.dirname(d))\n"
             "            print('Copying dataset folder', src); shutil.copytree(src, PROJ, dirs_exist_ok=True); break\n"
             "if not _have(PROJ):\n"
             "    raise SystemExit('PROJECT NOT FOUND. Either (a) push freevideo-kaggle to GitHub and set PROJECT_REPO above, '\n"
             "                     'or (b) Add Input -> Upload -> New Dataset with the freevideo-kaggle folder/zip, then re-run this cell.')\n"
             "\n"
             "if not os.path.isdir(f'{UP}/freevideo_engine'):\n"
             "    shutil.rmtree(UP, ignore_errors=True)\n"
             "    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/FlashML-org/FreeVideo.git', UP], check=True)\n"
             "sys.path.insert(0, PROJ); os.chdir(PROJ)\n"
             "!python patches/apply_patches.py /kaggle/working/FreeVideo\n"
             "print('project ready at', PROJ)"),
        code("# Cell 3 — Hardware + kernel probes (run BEFORE downloading any model)\n"
             "import sys; sys.path.insert(0, '/kaggle/working/freevideo-kaggle')\n"
             "from app.hardware import HardwareDetector\n"
             "print(HardwareDetector().detect().to_dict())\n"
             "!python scripts/probe_kernels.py --out /kaggle/working/kernel_probe.json\n"
             "!python scripts/benchmark.py --quick --out /kaggle/working/benchmark_quick.json"),
        code(CELL4),
        code(CELL5),
        code(CELL6),
        md("## Manual steps\n"
           "1. Open the public `https://....trycloudflare.com` link printed by Cell 5 (Kaggle has no built-in port preview).\n"
           "2. Paste your Hindi/Hinglish script (stored verbatim).\n"
           "3. Select engine (Wan2.1 recommended), profile, style, TTS.\n"
           "4. Preview narration audio.\n"
           "5. Start generation → watch scene progress → preview scenes.\n"
           "6. Download final MP4 (also saved to /kaggle/working/outputs/).\n"
           "7. Optional: set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars for Telegram; "
           "YOUTUBE_CLIENT_SECRETS for YouTube upload.\n\n"
           "## Session recovery\n"
           "If the session ends mid-generation, re-run Cells 1–5, then either call the API with "
           "`resume_project_id` or set `RESUME_PROJECT_ID` in Cell 6 — completed scenes are skipped."),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "notebooks" / "kaggle_freevideo_t4.ipynb"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
