"""Flask web UI: paste script -> settings -> voice preview -> generate -> download.

Runs on Kaggle (0.0.0.0:5000) and locally. Single-page UI, no build step.
All generation runs in a background thread; progress via polling API.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file

from config import PROFILES, SETTINGS

app = Flask(__name__)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _job_public(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if k != "thread"}


@app.route("/")
def index() -> Response:
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/health")
def health():
    from app.envcheck import runtime_problems

    _problems = runtime_problems()
    from app.hardware import HardwareDetector

    return jsonify({"runtime_problems": _problems, "runtime_ok": not _problems, "ok": True, "hardware": HardwareDetector().detect().to_dict()})


@app.route("/api/profiles")
def profiles():
    return jsonify(PROFILES)


@app.route("/api/models")
def models():
    from app.hardware import HardwareDetector
    from app.registry import ModelCompatibilityChecker, VideoModelRegistry

    r = HardwareDetector().detect()
    checker = ModelCompatibilityChecker(r.capability, r.vram_total_bytes, r.ram_total_bytes, r.disk_free_bytes)
    return jsonify(VideoModelRegistry(checker).list_models())


@app.route("/api/preview-voice", methods=["POST"])
def preview_voice():
    data = request.get_json(force=True)
    text = (data.get("text") or "यह एक परीक्षण है।").strip()
    backend = data.get("backend", "indic-parler")
    from app.tts import HindiTTSService

    out = SETTINGS.data_dir / "previews" / f"preview_{int(time.time())}.wav"
    tts = HindiTTSService(backend=backend, model=SETTINGS.tts_model,
                          voice_description=SETTINGS.tts_voice_description)
    result = tts.synthesize_script(text, out)
    tts.unload()
    if not result.success or result.audio_path is None:
        return jsonify({"success": False, "error": result.error}), 400
    return send_file(str(result.audio_path), mimetype="audio/wav",
                    as_attachment=False, download_name="preview.wav")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    script = data.get("script", "")
    from app.envcheck import runtime_problems

    _pb = runtime_problems()
    if _pb:
        return jsonify({"success": False, "error": "Runtime not ready: " + " | ".join(_pb)}), 503
    if not script.strip():
        return jsonify({"success": False, "error": "Script is empty."}), 400
    job_id = f"job_{int(time.time() * 1000)}"
    job: dict[str, Any] = {
        "id": job_id, "status": "queued", "stage": "", "done": 0, "total": 0,
        "message": "", "manifest": None, "error": "", "created_at": time.time(),
    }
    with _jobs_lock:
        _jobs[job_id] = job

    def progress(stage: str, done: int, total: int, msg: str) -> None:
        with _jobs_lock:
            _jobs[job_id].update({"status": "running", "stage": stage, "done": done,
                                  "total": total, "message": msg})

    def worker() -> None:
        try:
            from app.pipeline import Pipeline

            pipe = Pipeline(progress=progress)
            manifest = pipe.run(
                script,
                engine_id=data.get("engine", "wan2.1"),
                profile=data.get("profile", "balanced"),
                tts_backend=data.get("tts_backend", "indic-parler"),
                style=data.get("style", "horror"),
                seed=int(data.get("seed", 42)),
                resume_project_id=data.get("resume_project_id") or None,
            )
            with _jobs_lock:
                _jobs[job_id].update({"status": "complete", "manifest": manifest})
        except Exception as e:
            import traceback

            with _jobs_lock:
                _jobs[job_id].update({"status": "failed", "error": f"{e}\n{traceback.format_exc(limit=5)}"})

    t = threading.Thread(target=worker, daemon=True)
    job["thread"] = t
    t.start()
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/job/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "unknown job"}), 404
    return jsonify(_job_public(job))


@app.route("/api/download/<project_id>")
def download(project_id: str):
    final = SETTINGS.data_dir / "projects" / project_id / "final.mp4"
    if not final.exists():
        return jsonify({"success": False, "error": "final video not ready"}), 404
    return send_file(str(final), mimetype="video/mp4", as_attachment=True, download_name=f"{project_id}.mp4")


@app.route("/api/scene/<project_id>/<int:index>")
def scene_preview(project_id: str, index: int):
    clip = SETTINGS.data_dir / "projects" / project_id / "scenes" / f"scene_{index:03d}.mp4"
    if not clip.exists():
        return jsonify({"success": False, "error": "scene not ready"}), 404
    return send_file(str(clip), mimetype="video/mp4", as_attachment=False)


@app.route("/api/send-telegram", methods=["POST"])
def send_telegram():
    data = request.get_json(force=True)
    from app.telegram_service import TelegramService

    svc = TelegramService()
    video = Path(data.get("video", ""))
    return jsonify(svc.send_video(video, data.get("caption", "")))


@app.route("/api/upload-youtube", methods=["POST"])
def upload_youtube():
    data = request.get_json(force=True)
    from app.youtube_service import YouTubeService

    svc = YouTubeService()
    return jsonify(svc.upload(Path(data.get("video", "")), data.get("title", "Video"),
                              data.get("description", ""), data.get("privacy", "unlisted")))


INDEX_HTML = r"""<!doctype html>
<html lang="hi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FreeVideo-Kaggle — Hindi Script to Video</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.5}
textarea{width:100%;min-height:160px} .row{display:flex;gap:1rem;flex-wrap:wrap} .card{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}
button{padding:.5rem 1rem;cursor:pointer} video{max-width:100%} .ok{color:green}.err{color:red}</style></head>
<body>
<h1>FreeVideo-Kaggle — Hindi Script to Video</h1>
<p>Your pasted script is the source of truth. It is stored verbatim and narrated exactly. Nothing is rewritten.</p>
<div class="card"><h2>1. Paste your Hindi/Hinglish script</h2>
<textarea id="script" placeholder="अपना script यहाँ paste करें..."></textarea></div>
<div class="card"><h2>2. Settings</h2><div class="row">
<label>Engine <select id="engine"><option value="wan2.1">Wan2.1 1.3B (recommended T4/P100)</option>
<option value="cogvideox">CogVideoX 2B</option><option value="svd">SVD-XT (image-to-video)</option>
<option value="freevideo-h3">FreeVideo H3 (experimental)</option></select></label>
<label>Profile <select id="profile"><option value="balanced">Balanced</option><option value="max_quality">Max Quality</option>
<option value="low_vram">Low VRAM</option><option value="cpu_assisted">CPU-Assisted</option></select></label>
<label>Style <select id="style"><option value="horror">Horror</option><option value="cinematic">Cinematic</option></select></label>
<label>TTS <select id="tts"><option value="indic-parler">Indic Parler-TTS (best; gated — needs HF_TOKEN)</option><option value="mms-hin">MMS Hindi (open, no token)</option><option value="silent">Silent (test only)</option></select></label>
</div></div>
<div class="card"><h2>3. Voice preview</h2>
<button onclick="previewVoice()">Preview narration</button> <audio id="previewAudio" controls></audio>
<p id="previewMsg"></p></div>
<div class="card"><h2>4. Generate</h2>
<button onclick="startGen()">Start generation</button>
<div id="progress"></div><div id="scenes"></div>
<video id="finalVideo" controls style="display:none"></video>
<p id="finalMsg"></p>
<button onclick="downloadFinal()" style="display:none" id="dlBtn">Download final MP4</button></div>
<script>
let jobId=null, projectId=null;
async function previewVoice(){const t=document.getElementById('script').value||'यह एक परीक्षण है।';
const r=await fetch('/api/preview-voice',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t,backend:document.getElementById('tts').value})});
if(r.ok){document.getElementById('previewAudio').src=URL.createObjectURL(await r.blob());document.getElementById('previewMsg').textContent='Preview ready.';}
else{document.getElementById('previewMsg').textContent='Preview failed: '+(await r.text());}}
async function startGen(){const body={script:document.getElementById('script').value,engine:document.getElementById('engine').value,profile:document.getElementById('profile').value,style:document.getElementById('style').value,tts_backend:document.getElementById('tts').value};
const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const d=await r.json(); if(!d.success){alert(d.error);return;} jobId=d.job_id; poll();}
async function poll(){const r=await fetch('/api/job/'+jobId); const j=await r.json();
document.getElementById('progress').textContent=j.stage+' '+j.done+'/'+j.total+' — '+j.message;
if(j.status==='complete'){projectId=j.manifest.project_id; const v=document.getElementById('finalVideo'); v.src='/api/download/'+projectId; v.style.display='block'; document.getElementById('dlBtn').style.display='inline'; document.getElementById('finalMsg').innerHTML='<span class="ok">Done!</span> Project '+projectId; renderScenes(j.manifest);}
else if(j.status==='failed'){document.getElementById('finalMsg').innerHTML='<span class="err">'+j.error+'</span>';}
else setTimeout(poll,3000);}
function renderScenes(m){const el=document.getElementById('scenes'); el.innerHTML='<h3>Scenes</h3>'; m.scenes.forEach((s,i)=>{const d=document.createElement('div'); d.innerHTML='<p>Scene '+(i+1)+': '+s.visual_prompt+'</p><video controls src="/api/scene/'+m.project_id+'/'+i+'"></video>'; el.appendChild(d);});}
function downloadFinal(){window.location='/api/download/'+projectId;}
</script></body></html>"""


def run(host: str = "0.0.0.0", port: int = 5000) -> None:
    from app.envcheck import runtime_problems

    for p in runtime_problems():
        print("RUNTIME PROBLEM:", p, flush=True)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    run(SETTINGS.host, SETTINGS.port)
