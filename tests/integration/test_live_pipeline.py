"""Ring 2 live checks against real project data and a real ComfyUI.

    COMFYRACK_TEST_PROJECT=/path/to/your/project \
      python -m pytest tests/integration -m integration -v
"""
import os
import pathlib
import pytest

from comfyrack import Rack
from comfyrack.lora_meta import scan
from comfyrack.shots import load_shots, validate

pytestmark = pytest.mark.integration

PROJECT = os.environ.get("COMFYRACK_TEST_PROJECT")
LORAS = os.environ.get("COMFYRACK_TEST_LORAS")


@pytest.fixture(scope="module")
def project():
    if not PROJECT or not pathlib.Path(PROJECT).is_dir():
        pytest.skip("COMFYRACK_TEST_PROJECT not set to a real project")
    return pathlib.Path(PROJECT)


def test_lora_scan_finds_triggers_in_real_files():
    if not LORAS:
        pytest.skip("COMFYRACK_TEST_LORAS not set")
    d = pathlib.Path(LORAS)
    if not d.is_dir():
        pytest.skip(f"no LoRA directory at {d}")
    metas = scan(d)
    assert len(metas) > 10
    with_trigger = [m for m in metas.values() if m.trigger]
    assert with_trigger, "no LoRA yielded a trigger token"


def test_id_lora_yields_its_known_trigger():
    lora = os.environ.get("COMFYRACK_TEST_ID_LORA")
    trigger = os.environ.get("COMFYRACK_TEST_ID_TRIGGER")
    if not lora or not trigger:
        pytest.skip("COMFYRACK_TEST_ID_LORA and COMFYRACK_TEST_ID_TRIGGER not set")
    p = pathlib.Path(lora)
    if not p.is_file():
        pytest.skip(f"no LoRA at {p}")
    from comfyrack.lora_meta import read_meta
    assert read_meta(p).trigger == trigger


def test_every_real_shot_list_loads_and_validates(project):
    shots_dir = project / "shots"
    if not shots_dir.is_dir():
        pytest.skip("no shots directory")
    files = sorted(shots_dir.glob("*_shots.yaml"))
    assert files, "no shot lists found"
    failures = []
    for path in files:
        sl = load_shots(path)
        problems = validate(sl)
        if problems:
            failures.append((path.name, problems))
    for name, problems in failures:
        print(name, problems)
    assert not failures


def test_real_character_profiles_load_with_triggers(project, monkeypatch):
    monkeypatch.chdir(project)
    rack = Rack()
    chars = rack.profile.characters()
    if not chars:
        pytest.skip("no characters configured; run comfyrack init first")
    assert any(c.trigger for c in chars.values())


def test_prompt_assembly_over_the_real_lexicon(project, monkeypatch):
    monkeypatch.chdir(project)
    rack = Rack()
    if not rack.lexicon.all():
        pytest.skip("lexicon empty; run comfyrack lex sync first")
    term = sorted(rack.lexicon.all())[0]
    entry = rack.lexicon.get(term)
    lineage = sorted(entry.renders)[0]
    text, _warnings = rack.assemble_prompt(lineage, "booru", subject=term,
                                           action="standing on a dock")
    assert text and "standing on a dock" in text
