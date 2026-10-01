import yaml
from comfyrack.cli.main import main


def setup(tmp_path):
    (tmp_path / ".comfyrack").mkdir(parents=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\n\n'
        '[paths]\ncharacters = "chars"\nlocations = "locs"\nrecipes = "recs"\n',
        encoding="utf-8")
    ch = tmp_path / "chars" / "char_a"
    ch.mkdir(parents=True)
    (ch / "recipe.yaml").write_text(yaml.safe_dump({
        "character": "char_a", "base_model_family": "anima_illustrious",
        "id_lora": {"name": "your_character_lora.safetensors", "strength": 0.8,
                    "trigger": "trig_a"},
        "style_lora": {"name": "your_style_lora.safetensors", "strength": 1.0},
        "sampler": {"sampler_name": "er_sde", "scheduler": "simple", "steps": 20},
    }), encoding="utf-8")
    loc = tmp_path / "locs" / "bedroom"
    loc.mkdir(parents=True)
    (loc / "reference_plate.png").write_bytes(b"PNG")
    recs = tmp_path / "recs"
    recs.mkdir()
    (recs / "anima_v1.yaml").write_text("steps: 20\ncfg: 4.0\n", encoding="utf-8")
    return tmp_path


def run(tmp_path, argv, capsys, monkeypatch):
    monkeypatch.chdir(setup(tmp_path))
    code = main(argv)
    return code, capsys.readouterr().out


def test_char_list_shows_each_character_with_its_id_lora(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["char", "list"], capsys, monkeypatch)
    assert code == 0 and "char_a" in out and "your_character_lora.safetensors" in out


def test_char_show_prints_the_locked_recipe(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["char", "show", "char_a"], capsys, monkeypatch)
    assert code == 0
    assert "trig_a" in out and "er_sde" in out and "your_style_lora.safetensors" in out


def test_char_show_unknown_exits_one_listing_the_known(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["char", "show", "nobody"], capsys, monkeypatch)
    assert code == 1 and "char_a" in out


def test_loc_list_marks_which_locations_have_a_plate(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["loc", "list"], capsys, monkeypatch)
    assert code == 0 and "bedroom" in out


def test_loc_show_prints_the_plate_path(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["loc", "show", "bedroom"], capsys, monkeypatch)
    assert code == 0 and "reference_plate.png" in out


def test_recipe_list_shows_recipe_names(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["recipe", "list"], capsys, monkeypatch)
    assert code == 0 and "anima_v1" in out


def test_recipe_show_prints_the_resolved_values(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["recipe", "show", "anima_v1"], capsys, monkeypatch)
    assert code == 0 and "steps" in out and "20" in out


def test_char_list_with_no_characters_reports_a_definitive_zero(tmp_path, capsys, monkeypatch):
    (tmp_path / ".comfyrack").mkdir(parents=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["char", "list"]) == 0
    assert "0 found" in capsys.readouterr().out
