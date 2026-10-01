import json
import struct
import yaml
from comfyrack import config
from comfyrack.cli.main import main


def write_safetensors(path, metadata):
    header = {"__metadata__": metadata,
              "w": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00")


def setup(tmp_path, monkeypatch):
    # Pin USER_CONFIG to a nonexistent path so a developer's real
    # ~/.comfyrack/config.toml never leaks [machines]/[paths] entries into
    # these tests, matching the hermeticity pin test_cli.py's
    # _pin_hermetic_registry() applies for its own tests.
    monkeypatch.setattr(config, "USER_CONFIG", tmp_path / "nonexistent_user_config")
    cfg = tmp_path / ".comfyrack"
    cfg.mkdir(parents=True)
    (tmp_path / "loras").mkdir()
    (cfg / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\n\n'
        '[paths]\nlexicon = "lex"\nloras = "loras"\n', encoding="utf-8")
    (tmp_path / "lex").mkdir()
    (tmp_path / "lex" / "char_a.yaml").write_text(yaml.safe_dump(
        {"term": "char_a", "kind": "character",
         "renders": {"illustrious": {"trigger": "trig_a",
                                     "tags": "1girl, (straight blonde hair:1.2)"}}}),
        encoding="utf-8")
    return tmp_path


def run(tmp_path, argv, capsys, monkeypatch):
    monkeypatch.chdir(setup(tmp_path, monkeypatch))
    code = main(argv)
    return code, capsys.readouterr().out


def test_lex_list_shows_terms_with_kinds_and_lineages(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["lex", "list"], capsys, monkeypatch)
    assert code == 0
    assert "char_a" in out and "character" in out and "illustrious" in out


def test_lex_list_reports_a_definitive_zero_when_empty(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(setup(tmp_path, monkeypatch))
    for p in (tmp_path / "lex").glob("*.yaml"):
        p.unlink()
    code = main(["lex", "list"])
    assert code == 0
    assert "0 found" in capsys.readouterr().out


def test_lex_show_prints_the_render_for_each_lineage(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["lex", "show", "char_a"], capsys, monkeypatch)
    assert code == 0
    assert "trig_a, 1girl, (straight blonde hair:1.2)" in out


def test_lex_show_unknown_term_exits_one_with_the_add_command(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["lex", "show", "harborview"], capsys, monkeypatch)
    assert code == 1
    assert "comfyrack lex add" in out


def test_lex_sync_reports_what_it_created(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(setup(tmp_path, monkeypatch))
    write_safetensors(tmp_path / "loras" / "char_b_cel.safetensors", {
        "ss_base_model_version": "anima",
        "ss_tag_frequency": json.dumps({"i": {"j0sh": 10, "1boy": 10, "brown hair": 10}})})
    code = main(["lex", "sync"])
    out = capsys.readouterr().out
    assert code == 0
    assert "char_b_cel" in out and "created" in out


def test_lex_add_writes_an_entry_that_show_can_read_back(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(setup(tmp_path, monkeypatch))
    assert main(["lex", "add", "harborview-coast", "--kind", "location",
                 "--lineage", "illustrious", "--tags", "ocean, wooden dock"]) == 0
    capsys.readouterr()
    assert main(["lex", "show", "harborview-coast"]) == 0
    assert "ocean, wooden dock" in capsys.readouterr().out


def test_lex_add_without_kind_preserves_an_existing_entrys_kind(
        tmp_path, capsys, monkeypatch):
    # `char_a` already exists with kind="character" (see setup()). Adding a new
    # lineage without repeating --kind must not silently downgrade it to the
    # argparse default "term".
    monkeypatch.chdir(setup(tmp_path, monkeypatch))
    assert main(["lex", "add", "char_a", "--lineage", "sdxl",
                 "--tags", "1girl, blonde hair"]) == 0
    capsys.readouterr()
    assert main(["lex", "show", "char_a", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "character"


def test_lex_add_with_explicit_kind_still_overrides_an_existing_entry(
        tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(setup(tmp_path, monkeypatch))
    assert main(["lex", "add", "char_a", "--kind", "term", "--lineage", "sdxl",
                 "--tags", "1girl, blonde hair"]) == 0
    capsys.readouterr()
    assert main(["lex", "show", "char_a", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "term"


def test_lex_add_without_kind_on_a_brand_new_term_defaults_to_term(
        tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(setup(tmp_path, monkeypatch))
    assert main(["lex", "add", "brand-new-term", "--lineage", "sdxl",
                 "--tags", "some, tags"]) == 0
    capsys.readouterr()
    assert main(["lex", "show", "brand-new-term", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "term"


def test_prompt_preview_assembles_without_contacting_comfyui(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["prompt", "preview", "--lineage", "illustrious",
                               "--dialect", "booru", "--subject", "char_a",
                               "--action", "mid-dance, arms raised"],
                    capsys, monkeypatch)
    assert code == 0
    assert "trig_a, 1girl, (straight blonde hair:1.2), mid-dance, arms raised" in out


def test_prompt_preview_refuses_an_ungrounded_term_with_exit_two(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["prompt", "preview", "--lineage", "illustrious",
                               "--dialect", "booru",
                               "--action", "standing on the Harborview Coast"],
                    capsys, monkeypatch)
    assert code == 2
    assert "Harborview" in out


def test_allow_unknown_lets_the_preview_through(tmp_path, capsys, monkeypatch):
    code, out = run(tmp_path, ["prompt", "preview", "--lineage", "illustrious",
                               "--dialect", "booru", "--allow-unknown",
                               "--action", "standing on the Harborview Coast"],
                    capsys, monkeypatch)
    assert code == 0
    assert "Harborview" in out


def test_lex_commands_are_hermetic_against_the_real_user_config(
        tmp_path, capsys, monkeypatch):
    """setup()/run() never pinned comfyrack.config.USER_CONFIG, so on a machine
    with a real ~/.comfyrack/config.toml, that file leaks into every test in this
    module. Simulate it with a malformed "real" config: if it's never overridden
    to a nonexistent path, Config.load() tries to parse it and every command in
    this file breaks on an unrelated file the developer happens to have."""
    leak_cfg = tmp_path / "leaky_home_config.toml"
    leak_cfg.write_text("this is not valid toml [[[", encoding="utf-8")
    monkeypatch.setattr(config, "USER_CONFIG", leak_cfg)

    code, out = run(tmp_path, ["lex", "list"], capsys, monkeypatch)
    assert code == 0
    assert "char_a" in out
