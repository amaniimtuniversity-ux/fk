"""ScenePlanner: script -> unique scene plan (script itself NEVER rewritten).

Each scene: narration segment (verbatim slice), visual prompt (derived,
English for the video model), character/location/lighting/camera/movement/
atmosphere/action/duration/SFX/continuity. No generic prompt reuse: the
visual prompt is built from the narration segment + continuity state.

This is rule-based (no LLM call, no paid API, no data exfiltration).
An optional LLM hook can be plugged in later without changing the contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .script_manager import split_sentences


@dataclass
class Scene:
    index: int
    narration: str  # verbatim slice of the original script
    visual_prompt: str  # English prompt for the video model (derived)
    character: str = ""
    location: str = ""
    lighting: str = "cinematic low-key lighting"
    camera_angle: str = "medium shot"
    camera_movement: str = "slow dolly-in"
    atmosphere: str = "tense, atmospheric"
    action: str = ""
    duration_seconds: float = 5.0
    sfx: str = ""
    continuity: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "narration": self.narration,
            "visual_prompt": self.visual_prompt, "character": self.character,
            "location": self.location, "lighting": self.lighting,
            "camera_angle": self.camera_angle, "camera_movement": self.camera_movement,
            "atmosphere": self.atmosphere, "action": self.action,
            "duration_seconds": self.duration_seconds, "sfx": self.sfx,
            "continuity": self.continuity,
        }


# Keyword hints for visual derivation (Hindi + Hinglish + English).
_HINTS: list[tuple[tuple[str, ...], dict[str, str]]] = [
    (("haveli", "पुरानी हवेली", "old mansion", "mansion"), {"location": "an old decaying haveli at night", "atmosphere": "haunted, dusty, eerie"}),
    (("jungle", "जंगल", "forest", "pedh", "पेड़"), {"location": "a dense dark forest", "atmosphere": "ominous, misty"}),
    (("gaon", "गाँव", "village"), {"location": "a rural Indian village at dusk", "atmosphere": "quiet, melancholic"}),
    (("kua", "कुआँ", "well"), {"location": "an old stone well in a courtyard", "atmosphere": "eerie, still"}),
    (("train", "ट्रेन", "rail"), {"location": "a dim railway compartment", "atmosphere": "claustrophobic, rattling"}),
    (("hospital", "अस्पताल", "ward"), {"location": "an empty hospital corridor", "atmosphere": "sterile, unsettling"}),
    (("bhoot", "भूत", "ghost", "chudail", "चुड़ैल", "aatma", "आत्मा"), {"atmosphere": "supernatural horror, dread"}),
    (("baarish", "बारिश", "rain"), {"lighting": "rainy night, neon reflections", "atmosphere": "wet, melancholic"}),
    (("raat", "रात", "night", "andhera", "अंधेरा"), {"lighting": "moonlit night, deep shadows"}),
    (("subah", "सुबह", "morning"), {"lighting": "soft golden morning light"}),
]

_CAMERA_CYCLE = [
    ("wide shot", "slow crane shot"),
    ("medium shot", "slow dolly-in"),
    ("close-up", "static with subtle handheld sway"),
    ("low-angle shot", "slow push-in"),
    ("over-the-shoulder", "slow pan right"),
]


class ScenePlanner:
    # Measured on real Hindi TTS output (MMS-TTS hin, sentence-stitched): ~1.8-1.9 words/second.
    # Keeping scenes near the video model's native max clip (Wan2.1: 81 frames = 5 s) minimises
    # slow-motion time-fitting later.
    HINDI_WORDS_PER_SECOND = 1.85

    def __init__(self, target_seconds: float = 5.0, max_scenes: int = 24,
                 words_per_second: float | None = None):
        self.target_seconds = target_seconds
        self.words_per_second = words_per_second or self.HINDI_WORDS_PER_SECOND
        self.max_scenes = max_scenes

    def plan(self, script: str, style: str = "horror") -> list[Scene]:
        sentences = split_sentences(script)
        if not sentences:
            raise ValueError("Script has no sentences to plan.")
        # Group sentences into scenes by target duration using the measured Hindi speech rate.
        scenes: list[Scene] = []
        current: list[str] = []
        current_words = 0
        target_words = max(6, int(round(self.target_seconds * self.words_per_second)))
        for s in sentences:
            current.append(s)
            current_words += len(s.split())
            if current_words >= target_words:
                scenes.append(self._make_scene(len(scenes), " ".join(current), style))
                current, current_words = [], 0
        if current:
            scenes.append(self._make_scene(len(scenes), " ".join(current), style))
        if len(scenes) > self.max_scenes:
            # Merge overflow into last scenes (never drop narration).
            overflow = scenes[self.max_scenes - 1 :]
            merged_narration = " ".join(sc.narration for sc in overflow)
            scenes = scenes[: self.max_scenes - 1] + [
                self._make_scene(self.max_scenes - 1, merged_narration, style)
            ]
        # Continuity: thread character/location forward.
        continuity = ""
        for sc in scenes:
            sc.continuity = continuity
            if sc.character:
                continuity = f"Same character: {sc.character}. Same location: {sc.location}."
        return scenes

    def _make_scene(self, index: int, narration: str, style: str) -> Scene:
        low = narration.lower()
        character = location = atmosphere = lighting = ""
        for keywords, fields in _HINTS:
            if any(k in low for k in keywords):
                location = fields.get("location", location)
                atmosphere = fields.get("atmosphere", atmosphere)
                lighting = fields.get("lighting", lighting)
        if not location:
            location = "a moody Indian interior at night" if style == "horror" else "a cinematic Indian setting"
        if not atmosphere:
            atmosphere = "tense, atmospheric, cinematic" if style == "horror" else "warm, cinematic"
        if not lighting:
            lighting = "cinematic low-key lighting, volumetric haze"
        angle, movement = _CAMERA_CYCLE[index % len(_CAMERA_CYCLE)]
        # Visual prompt: derived from narration keywords + scene fields.
        # Keep it specific per scene (no generic reuse).
        visual = (
            f"{location}, {atmosphere}, {lighting}, {angle}, {movement}, "
            f"photorealistic, cinematic color grade, 16:9, high detail, "
            f"scene {index + 1}: {self._keywords(narration)}"
        )
        return Scene(
            index=index, narration=narration, visual_prompt=visual,
            character=character or "protagonist in traditional Indian clothing",
            location=location, lighting=lighting, camera_angle=angle,
            camera_movement=movement, atmosphere=atmosphere,
            action=self._keywords(narration),
            duration_seconds=self.target_seconds,
            sfx="low drone, distant wind" if style == "horror" else "soft ambience",
        )

    @staticmethod
    def _keywords(narration: str, max_words: int = 12) -> str:
        words = [w for w in narration.replace("।", " ").split() if len(w) > 2]
        return " ".join(words[:max_words]) or "mysterious scene"

    def verify_coverage(self, script: str, scenes: list[Scene]) -> bool:
        """Every word of the script appears in exactly one scene narration."""
        import re

        norm = lambda s: re.sub(r"\s+", " ", s).strip()
        joined = norm(" ".join(sc.narration for sc in scenes))
        return joined == norm(script)
