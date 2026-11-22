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
from .audio import (audio_duration, concat_audio, generate_ambience, loudness_normalize,
                    mix_narration_ambience, pad_audio_to_duration, write_srt)
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

        # ---- Narration FIRST, per scene (verbatim slices) -> exact timeline -------------------
        # The narration drives the timeline: each scene's video is generated/fitted to the length of
        # its own spoken text, so the full script is always heard and A/V stay aligned.
        narr_dir = pdir / "narration"
        narr_dir.mkdir(parents=True, exist_ok=True)
        narration_path = pdir / "narration.wav"
        narr_manifest = ckpt.load_json("narration_manifest.json") or {}
        scene_audio: list[dict[str, Any]] = narr_manifest.get("scenes", [])
        complete = (len(scene_audio) == len(scenes)
                    and all(Path(x["path"]).exists() for x in scene_audio)
                    and narr_manifest.get("script_hash") == project.script_hash)
        if not complete:
            if tts_backend == "silent":
                self.memory.warn("Narration is SILENT (explicitly requested test mode).")
            tts = HindiTTSService(backend=tts_backend, model=SETTINGS.tts_model,
                                  voice_description=SETTINGS.tts_voice_description)
            scene_audio = []
            used_backend = None
            for i, sc in enumerate(scenes):
                wav = narr_dir / f"narr_{i:03d}.wav"
                self._report("narration", i, len(scenes), f"scene {i + 1}/{len(scenes)} TTS")
                r = tts.synthesize_script(sc.narration, wav)
                if not r.success:
                    tts.unload()
                    ckpt.set_status("failed", {"error": f"TTS failed on scene {i}: {r.error}"})
                    raise RuntimeError(f"Narration (TTS) failed on scene {i + 1}: {r.error}")
                if r.is_silent_fallback and tts_backend != "silent":
                    tts.unload()
                    ckpt.set_status("failed", {"error": "silent narration without explicit request"})
                    raise RuntimeError("Narration would be SILENT but a real TTS backend was requested. "
                                       f"Refusing to produce a fake video. Details: {r.error}")
                if used_backend is None and r.backend != tts_backend:
                    self.memory.warn(f"TTS fallback: requested {tts_backend}, used {r.backend}. {r.error}")
                used_backend = r.backend
                scene_audio.append({"index": i, "path": str(wav), "seconds": round(audio_duration(wav), 3),
                                    "backend": r.backend, "sample_rate": r.sample_rate,
                                    "is_silent_fallback": r.is_silent_fallback})
            tts.unload()
            narr_manifest = {"script_hash": project.script_hash, "backend": used_backend,
                             "scenes": scene_audio,
                             "duration_seconds": round(sum(x["seconds"] for x in scene_audio), 3),
                             "is_silent_fallback": any(x["is_silent_fallback"] for x in scene_audio)}
            ckpt.save_json("narration_manifest.json", narr_manifest)
        # per-scene target length = speech + a short breath
        pause = float(getattr(SETTINGS, "scene_pause_seconds", 0.4))
        targets = [max(1.0, x["seconds"] + pause) for x in scene_audio]
        for sc, t in zip(scenes, targets):
            sc.duration_seconds = round(t, 3)
        self._report("narration", len(scenes), len(scenes),
                     f"narration done ({narr_manifest.get('backend')}, "
                     f"{narr_manifest.get('duration_seconds')}s, {len(scenes)} scenes)")

        # ---- Engine + registry ------------------------------------------------------------------
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
            if "relocating" in n:
                self.memory.warnings.append(f"storage: {n}")
        checker = ModelCompatibilityChecker(report.capability, report.vram_total_bytes,
                                            report.ram_total_bytes, cache_free, disk_path=str(cache_dir))
        registry = VideoModelRegistry(checker)
        spec = registry.get(model_id)
        compat = checker.check(spec)
        if not compat["compatible"] and spec.status != "experimental":
            raise RuntimeError(f"Model {spec.id} incompatible: {compat['reasons']}")
        engine = get_engine(engine_id, spec.hf_repo, precision.to_dict(), attention.to_dict(), prof["offload"])

        # ---- Per-scene generation (resumable), frame count follows narration ----------------------
        from .engines.base import GenerationRequest

        gen_fps = int(SETTINGS.default_fps)          # model-native (Wan: 16)
        max_frames = int(prof["frames"])
        ckpt.cleanup_partial()
        scene_results: list[dict[str, Any]] = []
        for i, scene in enumerate(scenes):
            want = int(round(targets[i] * gen_fps)) + 1
            want = max(9, min(max_frames, want))
            want = want - ((want - 1) % 4)            # Wan needs 4n+1
            if ckpt.is_scene_complete(i):
                self._report("video", i + 1, len(scenes), f"scene {i + 1} cached (resume)")
                scene_results.append(ckpt.load_json(f"scenes/scene_{i:03d}.json") or {"success": True})
                continue
            self.memory.cleanup()
            self.memory.snapshot(f"before_scene_{i}")
            req = GenerationRequest(
                prompt=scene.visual_prompt, output_path=ckpt.scene_path(i),
                width=prof["width"], height=prof["height"], num_frames=want,
                fps=gen_fps, num_inference_steps=prof["steps"],
                guidance_scale=SETTINGS.default_guidance, seed=seed + i,
                image_path=first_frame,
                progress_callback=lambda s, t, m, i=i: self._report("video", i, len(scenes), m),
            )
            result = engine.generate(req)
            rd = result.to_dict()
            rd["requested_frames"] = want
            rd["target_seconds"] = round(targets[i], 3)
            ckpt.save_scene(i, rd)
            scene_results.append(rd)
            self._report("video", i + 1, len(scenes),
                         f"scene {i + 1}/{len(scenes)} {'ok' if result.success else 'FAILED: ' + result.error[:200]}"
                         + (f" ({result.frames}f, {result.seconds:.0f}s, "
                            f"{result.peak_vram_bytes/1e9:.1f}GB VRAM)" if result.success else ""))
            if not result.success:
                ckpt.set_status("failed", {"failed_scene": i, "error": result.error})
                raise RuntimeError(f"Scene {i} failed: {result.error}")
        try:
            engine.unload()
        except Exception:
            pass
        self.memory.cleanup()

        # ---- Time-fit each clip to its narration, build exact per-scene audio -------------------
        self._report("render", 1, 4, "time-fit clips to narration")
        renderer = VideoRenderer(fps=int(getattr(SETTINGS, "output_fps", 24)))
        fit_dir = pdir / "fit"
        fit_dir.mkdir(exist_ok=True)
        fitted: list[Path] = []
        audio_parts: list[Path] = []
        timeline: list[dict[str, Any]] = []
        for i, scene in enumerate(scenes):
            fit_mp4 = fit_dir / f"fit_{i:03d}.mp4"
            ok, err, info = renderer.fit_clip_to_duration(ckpt.scene_path(i), targets[i], fit_mp4)
            if not ok:
                ckpt.set_status("failed", {"error": f"time-fit scene {i} failed: {err}"})
                raise RuntimeError(f"Time-fit failed on scene {i + 1}: {err[-400:]}")
            seg_len = max(info.get("out_seconds") or targets[i], 0.1)
            wav_fit = fit_dir / f"audio_{i:03d}.wav"
            ok, err = pad_audio_to_duration(Path(scene_audio[i]["path"]), seg_len, wav_fit)
            if not ok:
                raise RuntimeError(f"Audio pad failed on scene {i + 1}: {err[-300:]}")
            fitted.append(fit_mp4)
            audio_parts.append(wav_fit)
            scene.duration_seconds = round(seg_len, 3)
            timeline.append({"index": i, "narration_seconds": scene_audio[i]["seconds"], **info,
                             "segment_seconds": round(seg_len, 3)})
            if info.get("slow_factor", 1.0) > 1.0:
                self._report("render", 1, 4, f"scene {i + 1}: slow-motion x{info['slow_factor']}"
                                              + (" + ping-pong" if info.get("pingpong") else ""))

        self._report("render", 2, 4, "concat video + narration timeline")
        concat_path = pdir / "concat.mp4"
        ok, err = renderer.concat_clips(fitted, concat_path)
        if not ok:
            ckpt.set_status("failed", {"error": f"concat failed: {err}"})
            raise RuntimeError(f"Concat failed: {err}")
        ok, err = concat_audio(audio_parts, narration_path)
        if not ok:
            raise RuntimeError(f"Narration concat failed: {err}")
        total_seconds = sum(t["segment_seconds"] for t in timeline)

        self._report("render", 3, 4, "audio: ambience + mix + loudnorm")
        ambience_path = pdir / "ambience.m4a"
        ok, err = generate_ambience(ambience_path, seconds=total_seconds + 1.0,
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
        self._report("render", 4, 4, "mux video + audio + subtitles")
        srt_path = write_srt([s.to_dict() for s in scenes], pdir / "subtitles.srt")
        final_path = pdir / "final.mp4"
        ok, err = renderer.mux_audio_subtitles(concat_path, final_audio, srt_path, final_path)
        if not ok:
            ckpt.set_status("failed", {"error": f"mux failed: {err}"})
            raise RuntimeError(f"Mux failed: {err}")
        final_dur = renderer.duration(final_path)
        if abs(final_dur - total_seconds) > 0.5 + 0.02 * total_seconds:
            ckpt.set_status("failed", {"error": f"duration mismatch {final_dur} vs {total_seconds}"})
            raise RuntimeError(f"Final duration {final_dur:.2f}s != timeline {total_seconds:.2f}s — "
                               "narration would be cut. Aborting instead of shipping a truncated video.")

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
            "narration": {k: v for k, v in narr_manifest.items() if k != "scenes"},
            "timeline": timeline, "duration_seconds": round(final_dur, 3),
            "generation_fps": gen_fps, "output_fps": renderer.fps,
            "memory_warnings": self.memory.warnings,
            "seconds": round(time.time() - started, 1),
            "final": str(final_path), "native_resolution": f"{prof['width']}x{prof['height']}",
            "upscaled": False,
        }
        ckpt.save_json("final_manifest.json", manifest)
        ckpt.set_status("complete", {"final": str(final_path)})
        self._report("done", 1, 1, f"complete: {final_path}")
        return manifest
