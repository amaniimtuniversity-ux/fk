"""Tests: script preservation (Test D), different scripts (Test E), scene coverage."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.scene_planner import ScenePlanner
from app.script_manager import ScriptManager, hash_script, split_sentences

HINDI_A = "रात को पुरानी हवेली में अजीब आवाज़ें आ रही थीं। मैंने दरवाज़ा खोला और अंदर घना अंधेरा था। अचानक पीछे से किसी के कदमों की आहट सुनाई दी।"
HINDI_B = "सुबह की ट्रेन प्लेटफॉर्म पर रुकी थी। बूढ़ा चायवाला मुस्कुराकर बोला, चलो बेटा, सफर लंबा है। मैंने खिड़की से बाहर देखा और पहाड़ दिख रहे थे।"


def test_script_hash_stable():
    assert hash_script(HINDI_A) == hash_script(HINDI_A)
    assert hash_script(HINDI_A) != hash_script(HINDI_B)


def test_split_sentences_preserves_text():
    parts = split_sentences(HINDI_A)
    assert len(parts) == 3
    assert " ".join(parts) == HINDI_A


def test_project_verbatim_and_hash(tmp_path):
    mgr = ScriptManager(tmp_path / "projects")
    proj = mgr.create_project(HINDI_A)
    assert proj.script_original == HINDI_A
    loaded = mgr.load_project(proj.project_id)
    assert loaded.script_original == HINDI_A
    assert loaded.script_hash == proj.script_hash
    assert mgr.verify_narration(proj, HINDI_A) is True


def test_scene_coverage_full_script():
    planner = ScenePlanner(target_seconds=5.0)
    scenes = planner.plan(HINDI_A)
    assert planner.verify_coverage(HINDI_A, scenes) is True
    # Narration slices are verbatim substrings.
    for sc in scenes:
        assert sc.narration in HINDI_A


def test_different_scripts_different_plans():
    planner = ScenePlanner()
    a = planner.plan(HINDI_A)
    b = planner.plan(HINDI_B)
    prompts_a = [s.visual_prompt for s in a]
    prompts_b = [s.visual_prompt for s in b]
    assert prompts_a != prompts_b
    assert a[0].narration != b[0].narration


def test_empty_script_rejected(tmp_path):
    mgr = ScriptManager(tmp_path / "projects")
    try:
        mgr.create_project("   ")
        raise AssertionError("should have raised")
    except ValueError:
        pass
