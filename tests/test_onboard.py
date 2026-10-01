import json
import pathlib
import yaml
import pytest
from comfyrack.graph import Graph
from comfyrack.onboard import infer_flags, onboard, init_project
from comfyrack import errors

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

OBJECT_INFO = {
    "KSampler": {"input": {"required": {
        "model": ["MODEL"], "seed": ["INT", {}], "steps": ["INT", {}],
        "cfg": ["FLOAT", {}], "sampler_name": [["euler", "er_sde"], {}]}}},
    "SaveImage": {"input": {"required": {"filename_prefix": ["STRING", {}],
                                         "images": ["IMAGE"]}}},
}

GRAPH = Graph({
    "3": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
          "inputs": {"seed": 1, "steps": 20, "cfg": 4.0, "sampler_name": "er_sde",
                     "model": ["4", 0]}},
    "9": {"class_type": "SaveImage", "_meta": {"title": "Save Image"},
          "inputs": {"filename_prefix": "out", "images": ["3", 0]}},
})


def test_inferred_flags_cover_every_widget_input():
    flags = infer_flags(GRAPH, OBJECT_INFO)
    assert {"seed", "steps", "cfg", "sampler_name", "filename_prefix"} <= set(flags)


def test_inferred_flags_exclude_link_inputs():
    flags = infer_flags(GRAPH, OBJECT_INFO)
    assert "model" not in flags and "images" not in flags


def test_inferred_flags_carry_node_id_title_and_param():
    f = infer_flags(GRAPH, OBJECT_INFO)["seed"]
    assert (f["node"], f["node_title"], f["param"]) == ("3", "KSampler", "seed")


def test_inferred_flag_types_follow_the_object_info_declaration():
    flags = infer_flags(GRAPH, OBJECT_INFO)
    assert flags["seed"]["type"] == "int"
    assert flags["cfg"]["type"] == "float"
    assert flags["sampler_name"]["type"] == "str"


def test_inferred_flags_are_not_required_because_the_graph_supplies_defaults():
    assert infer_flags(GRAPH, OBJECT_INFO)["seed"]["required"] is False


def test_duplicate_param_names_across_nodes_get_disambiguated_by_title():
    g = Graph({
        "1": {"class_type": "KSampler", "_meta": {"title": "Base"},
              "inputs": {"seed": 1, "steps": 1, "cfg": 1.0, "sampler_name": "euler"}},
        "2": {"class_type": "KSampler", "_meta": {"title": "Refiner"},
              "inputs": {"seed": 2, "steps": 2, "cfg": 2.0, "sampler_name": "euler"}},
    })
    flags = infer_flags(g, OBJECT_INFO)
    assert "base_seed" in flags and "refiner_seed" in flags


def test_onboard_writes_the_json_and_a_manifest_with_inferred_flags(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps(GRAPH.data), encoding="utf-8")
    dest = tmp_path / "registry" / "anima"
    json_path, manifest_path = onboard(src, dest, "character-scene", OBJECT_INFO)
    assert json_path.name == "character-scene.json"
    man = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert man["workflow"] == "character-scene.json"
    assert man["output_node_title"] == "Save Image"
    assert "seed" in man["flags"]


def test_onboard_detects_the_single_save_node_as_the_output(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps(GRAPH.data), encoding="utf-8")
    _, manifest_path = onboard(src, tmp_path / "reg", "w", OBJECT_INFO)
    assert yaml.safe_load(manifest_path.read_text())["output_node_title"] == "Save Image"


def test_onboard_refuses_a_ui_format_file(tmp_path):
    src = tmp_path / "ui.json"
    src.write_text((FIXTURES / "ui_basic.json").read_text(encoding="utf-8"),
                   encoding="utf-8")
    with pytest.raises(errors.ComfyrackError) as ei:
        onboard(src, tmp_path / "reg", "w", OBJECT_INFO)
    assert "API format" in str(ei.value)
    assert "Export (API)" in (ei.value.help_text or "")


def test_onboard_detects_output_via_object_info_output_node_flag(tmp_path):
    # ImageSaver is not in OUTPUT_CLASSES; only /object_info's output_node flag
    # marks it as the output.
    object_info = dict(OBJECT_INFO)
    object_info["ImageSaver"] = {
        "output_node": True,
        "input": {"required": {"filename_prefix": ["STRING", {}]}},
    }
    g = Graph({
        "3": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
              "inputs": {"seed": 1, "steps": 20, "cfg": 4.0, "sampler_name": "er_sde"}},
        "9": {"class_type": "ImageSaver", "_meta": {"title": "Save It"},
              "inputs": {"filename_prefix": "out"}},
    })
    src = tmp_path / "src.json"
    src.write_text(json.dumps(g.data), encoding="utf-8")
    _, manifest_path = onboard(src, tmp_path / "reg", "w", object_info)
    assert yaml.safe_load(manifest_path.read_text())["output_node_title"] == "Save It"


def test_onboard_refuses_when_no_output_node_can_be_identified(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"3": {"class_type": "KSampler", "_meta": {"title": "K"},
                                     "inputs": {"seed": 1}}}), encoding="utf-8")
    with pytest.raises(errors.ComfyrackError) as ei:
        onboard(src, tmp_path / "reg", "w", OBJECT_INFO)
    assert "output node" in str(ei.value).lower()


def test_onboard_refuses_when_output_node_is_ambiguous(tmp_path):
    g = Graph({
        "3": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
              "inputs": {"seed": 1, "steps": 20, "cfg": 4.0, "sampler_name": "er_sde"}},
        "9": {"class_type": "SaveImage", "_meta": {"title": "Save A"},
              "inputs": {"filename_prefix": "a"}},
        "10": {"class_type": "SaveImage", "_meta": {"title": "Save B"},
               "inputs": {"filename_prefix": "b"}},
    })
    src = tmp_path / "src.json"
    src.write_text(json.dumps(g.data), encoding="utf-8")
    with pytest.raises(errors.ComfyrackError) as ei:
        onboard(src, tmp_path / "reg", "w", OBJECT_INFO)
    assert "save a" in str(ei.value).lower() and "save b" in str(ei.value).lower()


def test_onboard_refuses_missing_source_file_with_a_helpful_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(errors.ComfyrackError) as ei:
        onboard(missing, tmp_path / "reg", "w", OBJECT_INFO)
    assert ei.value.help_text


def test_onboard_refuses_malformed_json_with_a_helpful_error(tmp_path):
    src = tmp_path / "bad.json"
    src.write_text("{this is not json", encoding="utf-8")
    with pytest.raises(errors.ComfyrackError) as ei:
        onboard(src, tmp_path / "reg", "w", OBJECT_INFO)
    assert "json" in str(ei.value).lower()
    assert ei.value.help_text


def test_onboard_refuses_to_overwrite_an_existing_manifest_without_force(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps(GRAPH.data), encoding="utf-8")
    dest = tmp_path / "reg" / "fam"
    first_json, first_manifest = onboard(src, dest, "w", OBJECT_INFO)
    with pytest.raises(errors.ComfyrackError) as ei:
        onboard(src, dest, "w", OBJECT_INFO)
    assert str(first_json) in str(ei.value) or str(first_manifest) in str(ei.value)
    assert "force" in ei.value.help_text.lower()
    # The refusal must not have clobbered what was already there.
    assert yaml.safe_load(first_manifest.read_text(encoding="utf-8"))["workflow"] == "w.json"


def test_onboard_overwrites_when_force_is_true(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps(GRAPH.data), encoding="utf-8")
    dest = tmp_path / "reg" / "fam"
    onboard(src, dest, "w", OBJECT_INFO)
    json_path, manifest_path = onboard(src, dest, "w", OBJECT_INFO, force=True)
    assert json_path.is_file() and manifest_path.is_file()


def test_onboard_manifest_round_trips_through_registry(tmp_path, monkeypatch):
    # Hermeticity: this test constructs a real Registry, so per the branch's
    # ruling, pin BUILTIN_DIR/USER_DIR to empty tmp dirs even though we also
    # pass builtin_dir/user_dir explicitly below.
    from comfyrack import registry as registry_mod
    from comfyrack.config import Config
    from comfyrack.registry import Registry

    builtin_dir = tmp_path / "empty_builtin"
    user_dir = tmp_path / "empty_user"
    builtin_dir.mkdir()
    user_dir.mkdir()
    monkeypatch.setattr(registry_mod, "BUILTIN_DIR", builtin_dir)
    monkeypatch.setattr(registry_mod, "USER_DIR", user_dir)

    project_root = tmp_path / "proj"
    (project_root / ".comfyrack").mkdir(parents=True)
    (project_root / ".comfyrack" / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\n', encoding="utf-8")

    src = tmp_path / "src.json"
    src.write_text(json.dumps(GRAPH.data), encoding="utf-8")
    dest = project_root / ".comfyrack" / "registry" / "anima"
    onboard(src, dest, "character-scene", OBJECT_INFO)

    config = Config.load(start_dir=project_root)
    reg = Registry(config, builtin_dir=builtin_dir, user_dir=user_dir)
    manifest = reg.load("character-scene")

    assert manifest.output_node_title == "Save Image"
    seed_flag = manifest.flags["seed"]
    assert (seed_flag.param, seed_flag.node, seed_flag.type) == ("seed", "3", "int")


def test_init_writes_a_config_and_is_idempotent(tmp_path):
    p1 = init_project(tmp_path)
    assert p1.is_file()
    body = p1.read_text(encoding="utf-8")
    p2 = init_project(tmp_path)
    assert p2.read_text(encoding="utf-8") == body   # second call does not clobber


def test_init_writes_the_given_url(tmp_path):
    p = init_project(tmp_path, "http://127.0.0.1:8190")
    assert 'default = "http://127.0.0.1:8190"' in p.read_text(encoding="utf-8")
