"""Pipeline: end-to-end orchestration (script -> scenes -> TTS -> video -> render).

Flow:
1. Hardware detect + precision/attention plans
2. ScriptManager.create_project (verbatim + hash)
3. ScenePlanner.plan (unique scenes, full coverage verified)
4. HindiTTSService.synthesize_script (sentence-level, stitched)
5. Per-scene video generation with CheckpointManager resume
6. AudioEngine (ambience + loudnorm) + VideoRenderer (concat/mux/subtitles)
7. QualityEvaluator + manifests + Kaggle export

Progress is reported via callback for the web UI. Interruption-safe:
re-running resumes from completed scenes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from config import PROFILES, SETTINGS

from .attention import AttentionBackendSelector
from .audio import generate_ambience, loudness_normalize, mix_narration_ambience, write_srt
from .checkpoint import CheckpointManager
from .hardware import HardwareDetector
from .memory import MemoryManager
from .precision import PrecisionManager
from .registry import ModelCompatibilityChecker, VideoModelRegistry
from .renderer import VideoRenderer
from .scene_planner import ScenePlanner
from .script_manager import ScriptManager
from .tts import HindiTTSService

ProgressFn = Callable[[str, int, int, str], None]


def _noop(stage: str, done: int, total: int, msg: str) -> None:
    pass


def get_engine(engine_id: str, model_repo: str, precision_plan: dict, attention_plan: dict,
               offload_mode: str = "sequential"):
    if engine_id == "wan2.1":
        from .engines.wan_adapter import WanAdapter

        return WanAdapter(model_repo, precision_plan, attention_plan, offload_mode)
    if engine_id == "cogvideox":
        from .engines.cogvideox_adapter import CogVideoXAdapter

        return CogVideoXAdapter(model_repo, precision_plan, attention_plan, offload_mode)
    if engine_id == "svd":
        from .engines.svd_adapter import SVDAdapter

        return SVDAdapter(model_repo, precision_plan, attention_plan, offload_mode)
    if engine_id == "freevideo-h3":
        from .engines.freevideo_adapter import FreeVideoAdapter

        return FreeVideoAdapter(model_repo, precision_plan, attention_plan, offload_mode,
                                freevideo_repo=SETTINGS.freevideo_repo)
    raise ValueError(f"Unknown engine: {engine_id}")


class Pipeline:
    def __init__(self, data_dir: Optional[Path] = None, progress: Optional[ProgressFn] = None):
        self.data_dir = Path(data_dir) if data_dir else SETTINGS.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.progress = progress or _noop
        self.hardware = HardwareDetector(self.data_dir / "hardware.json")
        self.memory = MemoryManager()
        self.script_mgr = ScriptManager(self.data_dir / "projects")

    def _report(self, stage: str, done: int, total: int, msg: str = "") -> None:
        self.progress(stage, done, total, msg)

    def run(self, script: str, *, engine_id: str = "wan2.1", profile: str = "balanced",
            tts_backend: str = "indic-parler", style: str = "horror",
            seed: int = 42, resume_project_id: Optional[str] = None,
            first_frame: Optional[Path] = None) -> dict[str, Any]:
        started = time.time()
        SETTINGS.ensure_dirs()
        prof = PROFILES.get(profile, PROFILES["balanced"])
        report = self.hardware.detect()
        self._report("hardware", 1, 1, self.hardware.summary())
        precision = PrecisionManager(report.capability).probe_and_plan()
        attention = AttentionBackendSelector(report.capability).select()
        self._report("plans", 1, 1, f"precision={precision.compute_dtype} attention={attention.backend}")

        # Project (verbatim script + hash)
        if resume_project_id:
            project = self.script_mgr.load_project(resume_project_id)
            if project.script_original != script:
                raise ValueError("Resume script does not match stored original (hash mismatch).")
        else:
            project = self.script_mgr.create_project(script, metadata={"engine": engine_id, "profile": profile})
        pdir = self.data_dir / "projects" / project.project_id
        ckpt = CheckpointManager(pdir)
        ckpt.set_status("running", {"started_at": started, "engine": engine_id, "profile": profile})

        # Scene plan
        planner = ScenePlanner(target_seconds=SETTINGS.scene_target_seconds, max_scenes=SETTINGS.max_scenes)
        scenes = planner.plan(project.script_original, style=style)
        if not planner.verify_coverage(project.script_original, scenes):
            raise RuntimeError("Scene plan does not cover the full script (bug). Aborting.")
        ckpt.save_json("scene_plan.json", {"scenes": [s.to_dict() for s in scenes]})
        self._report("scenes", len(scenes), len(scenes), f"{len(scenes)} scenes planned")

        # Narration (exact script)
        narration_path = pdir / "narration.wav"
        if not narration_path.exists():
            tts = HindiTTSService(backend=tts_backend, model=SETTINGS.tts_model,
                                  voice_description=SETTINGS.tts_voice_description)
            tts_obj = tts.synthesize_script(project.script_original, narration_path)
            tts.unload()
            if not tts_obj.success:
                ckpt.set_status("failed", {"error": f"TTS failed: {tts_obj.error}"})
                raise RuntimeError(f"Narration (TTS) failed: {tts_obj.error}")
            if tts_obj.is_silent_fallback and tts_backend != "silent":
                ckpt.set_status("failed", {"error": "silent narration without explicit request"})
                raise RuntimeError("Narration would be SILENT but a real TTS backend was requested. "
                                   f"Refusing to produce a fake video. Details: {tts_obj.error}")
            tts_result = tts_obj.to_dict()  # normalise: fresh run (dataclass) == resume (json dict)
            ckpt.save_json("narration_manifest.json", tts_result)
            if tts_obj.backend != tts_backend:
                self.memory.warn(f"TTS fallback: requested {tts_backend}, used {tts_obj.backend}. {tts_obj.error}")
            if tts_obj.is_silent_fallback:
                self.memory.warn("Narration is SILENT (explicitly requested test mode).")
        else:
            tts_result = ckpt.load_json("narration_manifest.json") or {}
        if not narration_path.exists():
            raise RuntimeError(f"Narration file missing after TTS: {narration_path}")
        self._report("narration", 1, 1,
                     f"narration done ({tts_result.get('backend', '?')}, "
                     f"{tts_result.get('duration_seconds', 0)}s)")

        # Engine + registry
        from .kaggle_storage import cached_bytes, choose_cache_dir

        model_id = {"wan2.1": "wan2.1-1.3b", "cogvideox": "cogvideox-2b",
                    "svd": "svd-xt", "freevideo-h3": "freevideo-h3"}.get(engine_id, engine_id)
        pre_spec = VideoModelRegistry(ModelCompatibilityChecker(report.capability, report.vram_total_bytes,
                                                                report.ram_total_bytes, 10**18)).get(model_id)
        need = max(0, int(pre_spec.download_gb * 1e9) - cached_bytes(pre_spec.hf_repo))
        cache_dir, cache_free, cache_notes = choose_cache_dir(int(need * 1.05) + 2 * 10**9)
        self._report("storage", 1, 1, f"HF cache {cache_dir} (free {cache_free/1e9:.1f} GB, "
                                      f"need {need/1e9:.1f} GB more for {pre_spec.hf_repo})")
        for n in cache_notes:
            self.memory.warnings.append(f"storage: {n}") if "relocating" in n else None
        checker = ModelCompatibilityChecker(report.capability, report.vram_total_bytes,
                                            report.ram_total_bytes, cache_free, disk_path=str(cache_dir))
        registry = VideoModelRegistry(checker)
        spec = registry.get(model_id)
        compat = checker.check(spec)
        if not compat["compatible"] and spec.status != "experimental":
            raise RuntimeError(f"Model {spec.id} incompatible: {compat['reasons']}")
        engine = get_engine(engine_id, spec.hf_repo, precision.to_dict(), attention.to_dict(), prof["offload"])

        # Per-scene generation (resumable)
        from .engines.base import GenerationRequest

        ckpt.cleanup_partial()
        scene_results: list[dict[str, Any]] = []
        for i, scene in enumerate(scenes):
            if ckpt.is_scene_complete(i):
                self._report("video", i + 1, len(scenes), f"scene {i + 1} cached (resume)")
                scene_results.append(ckpt.load_json(f"scenes/scene_{i:03d}.json") or {"success": True})
                continue
            self.memory.cleanup()
            self.memory.snapshot(f"before_scene_{i}")
            req = GenerationRequest(
                prompt=scene.visual_prompt, output_path=ckpt.scene_path(i),
                width=prof["width"], height=prof["height"], num_frames=prof["frames"],
                fps=SETTINGS.default_fps, num_inference_steps=prof["steps"],
                guidance_scale=SETTINGS.default_guidance, seed=seed + i,
                image_path=first_frame,
                progress_callback=lambda s, t, m, i=i: self._report("video", i, len(scenes), m),
            )
            result = engine.generate(req)
            ckpt.save_scene(i, result.to_dict())
            scene_results.append(result.to_dict())
            self._report("video", i + 1, len(scenes),
                         f"scene {i + 1}/{len(scenes)} {'ok' if result.success else 'FAILED: ' + result.error[:200]}")
            if not result.success:
                ckpt.set_status("failed", {"failed_scene": i, "error": result.error})
                raise RuntimeError(f"Scene {i} failed: {result.error}")

        # Render
        self._report("render", 1, 3, "concat")
        renderer = VideoRenderer(fps=SETTINGS.default_fps)
        clips = [ckpt.scene_path(i) for i in range(len(scenes))]
        concat_path = pdir / "concat.mp4"
        ok, err = renderer.concat_clips(clips, concat_path)
        if not ok:
            ckpt.set_status("failed", {"error": f"concat failed: {err}"})
            raise RuntimeError(f"Concat failed: {err}")
        # Ambience + mix + loudnorm
        self._report("render", 2, 3, "audio: ambience + mix + loudnorm")
        ambience_path = pdir / "ambience.m4a"
        ok, err = generate_ambience(ambience_path,
                                    seconds=max(5.0, len(scenes) * SETTINGS.scene_target_seconds),
                                    kind="wind" if style == "horror" else "drone")
        if not ok:
            ckpt.set_status("failed", {"error": f"ambience failed: {err}"})
            raise RuntimeError(f"Ambience generation failed: {err}")
        mixed_path = pdir / "mixed.m4a"
        ok, err = mix_narration_ambience(narration_path, ambience_path, mixed_path)
        if not ok:
            ckpt.set_status("failed", {"error": f"audio mix failed: {err}"})
            raise RuntimeError(f"Audio mix failed: {err}")
        final_audio = pdir / "final_audio.m4a"
        ok, err = loudness_normalize(mixed_path, final_audio)
        if not ok:
            ckpt.set_status("failed", {"error": f"loudnorm failed: {err}"})
            raise RuntimeError(f"Loudness normalization failed: {err}")
        self._report("render", 3, 3, "mux video + audio + subtitles")
        srt_path = write_srt([s.to_dict() for s in scenes], pdir / "subtitles.srt")
        final_path = pdir / "final.mp4"
        ok, err = renderer.mux_audio_subtitles(concat_path, final_audio, srt_path, final_path)
        if not ok:
            ckpt.set_status("failed", {"error": f"mux failed: {err}"})
            raise RuntimeError(f"Mux failed: {err}")

        # Quality + manifest
        from .quality import evaluate_clip

        quality = evaluate_clip(final_path)
        manifest = {
            "project_id": project.project_id, "script_hash": project.script_hash,
            "engine": engine_id, "model": spec.id, "profile": profile,
            "hardware": report.to_dict(), "precision": precision.to_dict(),
            "attention": attention.to_dict(),
            "scenes": [s.to_dict() for s in scenes],
            "scene_results": scene_results, "quality": quality,
            "memory_warnings": self.memory.warnings,
            "seconds": round(time.time() - started, 1),
            "final": str(final_path), "native_resolution": f"{prof['width']}x{prof['height']}",
            "upscaled": False,
        }
        ckpt.save_json("final_manifest.json", manifest)
        ckpt.set_status("complete", {"final": str(final_path)})
        self._report("done", 1, 1, f"complete: {final_path}")
        return manifest
