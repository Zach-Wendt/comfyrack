import json
import yaml
import pytest
from comfyrack import config
from comfyrack.cli.main import main


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USER_CONFIG", tmp_path / "nonexistent_user_config")
    monkeypatch.chdir(tmp_path)
    # Init fresh project
    main(["init"])
    
    # Add subject profile under .comfyrack/subjects/cyber-car/recipe.yaml
    sub_dir = tmp_path / ".comfyrack" / "subjects" / "cyber-car"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "recipe.yaml").write_text(yaml.safe_dump({
        "subject": "cyber-car",
        "base_model_family": "flux",
        "id_lora": {"name": "cyber_car.safetensors", "strength": 0.95, "trigger": "cYbErC4r"},
        "style_lora": {"name": "sci_fi_style.safetensors", "strength": 0.8},
        "sampler": {"sampler_name": "euler", "scheduler": "normal", "steps": 25},
    }), encoding="utf-8")

    # Add scene profile under .comfyrack/scenes/neon-alley/scene.yaml
    scene_dir = tmp_path / ".comfyrack" / "scenes" / "neon-alley"
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "scene.yaml").write_text(yaml.safe_dump({
        "description": "a futuristic alleyway bathed in glowing neon lights"
    }), encoding="utf-8")
    (scene_dir / "reference_plate.png").write_bytes(b"PNG")

    # Add lexicon entries for cyber-car (subject) and neon-alley (scene)
    main(["lex", "add", "cyber-car", "--kind", "subject", "--lineage", "flux", "--text", "a sleek futuristic cyber sports car"])
    main(["lex", "add", "neon-alley", "--kind", "scene", "--lineage", "flux", "--text", "dark cyberpunk alleyway with neon signs"])

    return tmp_path


def test_init_creates_config_with_subjects_and_scenes_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    config_text = (tmp_path / ".comfyrack" / "config.toml").read_text(encoding="utf-8")
    assert 'subjects = ".comfyrack/subjects"' in config_text
    assert 'scenes = ".comfyrack/scenes"' in config_text


def test_subject_list_and_show(tmp_path, capsys, monkeypatch):
    setup(tmp_path, monkeypatch)
    capsys.readouterr()

    # Subject list
    code = main(["subject", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "cyber-car" in out and "cYbErC4r" in out and "cyber_car.safetensors" in out

    # Subject show
    code = main(["subject", "show", "cyber-car"])
    out = capsys.readouterr().out
    assert code == 0
    assert "cYbErC4r" in out and "sci_fi_style.safetensors" in out and "euler" in out


def test_scene_list_and_show(tmp_path, capsys, monkeypatch):
    setup(tmp_path, monkeypatch)
    capsys.readouterr()

    # Scene list
    code = main(["scene", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "neon-alley" in out and "plate" in out and "futuristic alleyway" in out

    # Scene show
    code = main(["scene", "show", "neon-alley"])
    out = capsys.readouterr().out
    assert code == 0
    assert "reference_plate.png" in out and "futuristic alleyway" in out


def test_subject_and_scene_fallback_to_char_and_loc(tmp_path, capsys, monkeypatch):
    setup(tmp_path, monkeypatch)
    # Add a legacy character and location
    char_dir = tmp_path / ".comfyrack" / "characters" / "legacy-char"
    char_dir.mkdir(parents=True, exist_ok=True)
    (char_dir / "recipe.yaml").write_text(yaml.safe_dump({
        "character": "legacy-char",
        "id_lora": {"name": "legacy.safetensors", "strength": 0.8, "trigger": "leg_trig"}
    }), encoding="utf-8")

    loc_dir = tmp_path / ".comfyrack" / "locations" / "legacy-loc"
    loc_dir.mkdir(parents=True, exist_ok=True)
    (loc_dir / "location.yaml").write_text(yaml.safe_dump({
        "description": "legacy location description"
    }), encoding="utf-8")
    capsys.readouterr()

    # Querying subject command for a legacy character should fall back smoothly
    code = main(["subject", "show", "legacy-char"])
    out = capsys.readouterr().out
    assert code == 0 and "leg_trig" in out

    # Querying scene command for a legacy location should fall back smoothly
    code = main(["scene", "show", "legacy-loc"])
    out = capsys.readouterr().out
    assert code == 0 and "legacy location description" in out


def test_acceptance_subject_and_scene_only_project_prompt_preview_and_lex_lookups(tmp_path, capsys, monkeypatch):
    """Acceptance criteria test:
    A freshly-init'ed project with ONLY declared subjects and scenes (no chapter/shot-list file at all)
    can run `comfyrack prompt preview` and lexicon lookups end to end using only subject/scene declarations.
    """
    setup(tmp_path, monkeypatch)
    capsys.readouterr()

    # Verify lexicon show for subject and scene
    code = main(["lex", "show", "cyber-car", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["kind"] == "subject"

    code = main(["lex", "show", "neon-alley", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["kind"] == "scene"

    # Verify prompt preview with --subject and --scene
    code = main([
        "prompt", "preview",
        "--lineage", "flux",
        "--dialect", "natural",
        "--subject", "cyber-car",
        "--scene", "neon-alley",
        "--action", "speeding past glowing skyscrapers",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "a sleek futuristic cyber sports car" in out
    assert "dark cyberpunk alleyway with neon signs" in out
    assert "speeding past glowing skyscrapers" in out
