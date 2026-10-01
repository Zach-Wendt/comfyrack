import pytest
from comfyrack.config import Config
from comfyrack.registry import Registry
from comfyrack import errors


def make_manifest(root, family, name, description="d", extra=""):
    d = root / family
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text("{}", encoding="utf-8")
    (d / f"{name}.manifest.yaml").write_text(
        f'description: "{description}"\n'
        f"workflow: {name}.json\n"
        f'output_node_title: "Save Image"\n'
        f"flags:\n"
        f"  seed:\n"
        f"    node: \"3\"\n"
        f"    node_title: \"KSampler\"\n"
        f"    param: seed\n"
        f"    type: int\n" + extra,
        encoding="utf-8")


def make_registry(tmp_path, project_cfg="[machines]\ndefault = \"http://x:8188\"\n"):
    (tmp_path / "proj" / ".comfyrack").mkdir(parents=True, exist_ok=True)
    (tmp_path / "proj" / ".comfyrack" / "config.toml").write_text(project_cfg, encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path / "proj", user_config=tmp_path / "none.toml")
    return Registry(cfg,
                    builtin_dir=tmp_path / "builtin",
                    user_dir=tmp_path / "userreg")


def test_discover_returns_manifests_from_every_layer_tagged_by_layer(tmp_path):
    make_manifest(tmp_path / "builtin", "ltx2.3", "i2v-basic")
    make_manifest(tmp_path / "userreg", "krea2", "txt2img")
    make_manifest(tmp_path / "proj" / ".comfyrack" / "registry", "anima", "character-scene")
    reg = make_registry(tmp_path)
    got = {(m.name, m.layer) for m in reg.discover()}
    assert got == {("i2v-basic", "builtin"), ("txt2img", "user"),
                   ("character-scene", "project")}


def test_project_layer_wins_over_builtin_for_the_same_name(tmp_path):
    make_manifest(tmp_path / "builtin", "anima", "character-scene", description="builtin version")
    make_manifest(tmp_path / "proj" / ".comfyrack" / "registry", "anima", "character-scene",
                  description="project version")
    reg = make_registry(tmp_path)
    assert reg.load("character-scene").description == "project version"


def test_shadowed_names_are_reported_rather_than_resolved_silently(tmp_path):
    make_manifest(tmp_path / "builtin", "anima", "character-scene")
    make_manifest(tmp_path / "proj" / ".comfyrack" / "registry", "anima", "character-scene")
    reg = make_registry(tmp_path)
    assert ("character-scene", "project", "builtin") in reg.shadowed()


def test_load_unknown_name_suggests_near_matches(tmp_path):
    make_manifest(tmp_path / "builtin", "anima", "character-scene")
    reg = make_registry(tmp_path)
    with pytest.raises(errors.NotFoundError) as ei:
        reg.load("character-scen")
    assert "character-scene" in ei.value.help_text


def test_manifest_flags_carry_node_id_and_title(tmp_path):
    make_manifest(tmp_path / "builtin", "anima", "character-scene")
    reg = make_registry(tmp_path)
    flag = reg.load("character-scene").flags["seed"]
    assert (flag.node, flag.node_title, flag.type) == ("3", "KSampler", "int")


def test_family_metadata_is_read_from_family_yaml(tmp_path):
    make_manifest(tmp_path / "builtin", "anima", "character-scene")
    (tmp_path / "builtin" / "anima" / "family.yaml").write_text(
        "base_model_family: illustrious\nprompt_dialect: booru\n",
        encoding="utf-8")
    reg = make_registry(tmp_path)
    fam = reg.family("anima")
    assert (fam.base_model_family, fam.prompt_dialect) == ("illustrious", "booru")


def test_family_counts_marks_in_scope_families_active(tmp_path):
    make_manifest(tmp_path / "builtin", "anima", "character-scene")
    make_manifest(tmp_path / "builtin", "ltx2.3", "i2v-basic")
    make_manifest(tmp_path / "builtin", "ltx2.3", "t2v-basic")
    reg = make_registry(tmp_path, project_cfg='[list]\nscope = ["anima"]\n')
    counts = dict((fam, (n, note)) for fam, n, note in reg.family_counts())
    assert counts["anima"] == (1, "active")
    assert counts["ltx2.3"][0] == 2
    assert "comfyrack list --family ltx2.3" in counts["ltx2.3"][1]


def test_find_searches_every_family_regardless_of_scope(tmp_path):
    make_manifest(tmp_path / "builtin", "ltx2.3", "i2v-basic")
    reg = make_registry(tmp_path, project_cfg='[list]\nscope = ["anima"]\n')
    assert [m.name for m in reg.find("i2v")] == ["i2v-basic"]


def test_run_resolves_an_out_of_scope_workflow_normally(tmp_path):
    make_manifest(tmp_path / "builtin", "ltx2.3", "i2v-basic")
    reg = make_registry(tmp_path, project_cfg='[list]\nscope = ["anima"]\n')
    assert reg.load("i2v-basic").family == "ltx2.3"


# -- a malformed manifest names itself (final review, Important 7) -----------
#
# discover() parses EVERY manifest on every call, and cmd_home/cmd_list/cmd_find/
# Rack.list all route through it, so one bad file used to take down the whole
# listing with a traceback that never said which file was bad.

def _write_raw_manifest(root, family, name, body):
    d = root / family
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text("{}", encoding="utf-8")
    (d / f"{name}.manifest.yaml").write_text(body, encoding="utf-8")


def test_malformed_yaml_in_a_manifest_names_the_offending_file(tmp_path):
    _write_raw_manifest(tmp_path / "builtin", "anima", "broken",
                        'description: "x\nworkflow: [unclosed\n')
    reg = make_registry(tmp_path)
    with pytest.raises(errors.ComfyrackError) as ei:
        reg.discover()
    assert "broken.manifest.yaml" in ei.value.message
    assert ei.value.help_text


def test_a_manifest_missing_a_required_key_names_the_file_and_the_key(tmp_path):
    _write_raw_manifest(tmp_path / "builtin", "anima", "nokey",
                        'description: "x"\nworkflow: nokey.json\n')   # no output_node_title
    reg = make_registry(tmp_path)
    with pytest.raises(errors.ComfyrackError) as ei:
        reg.discover()
    assert "nokey.manifest.yaml" in ei.value.message
    assert "output_node_title" in ei.value.message


def test_a_flag_missing_param_names_the_file(tmp_path):
    _write_raw_manifest(tmp_path / "builtin", "anima", "noparam",
                        'description: "x"\nworkflow: noparam.json\n'
                        'output_node_title: "Save Image"\n'
                        'flags:\n  seed:\n    node: "3"\n    type: int\n')
    reg = make_registry(tmp_path)
    with pytest.raises(errors.ComfyrackError) as ei:
        reg.discover()
    assert "noparam.manifest.yaml" in ei.value.message
    assert "param" in ei.value.message


def test_a_good_manifest_alongside_a_bad_one_is_unaffected_when_loaded_directly(tmp_path):
    """load() parses only the one name asked for, so a bad neighbour must not
    break an unrelated lookup."""
    make_manifest(tmp_path / "builtin", "anima", "good")
    _write_raw_manifest(tmp_path / "builtin", "anima", "bad", 'workflow: [unclosed\n')
    reg = make_registry(tmp_path)
    assert reg.load("good").description == "d"
