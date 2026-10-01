import pytest
from unittest.mock import MagicMock
from comfyrack import Rack, errors


def setup_project(tmp_path, scope=None):
    cfgdir = tmp_path / ".comfyrack"
    cfgdir.mkdir(parents=True)
    scope_block = f'\n[list]\nscope = {scope!r}\n' if scope else ""
    (cfgdir / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\nremote = "http://b:8188"\n' + scope_block,
        encoding="utf-8")
    reg = cfgdir / "registry" / "anima"
    reg.mkdir(parents=True)
    (reg / "w.json").write_text("{}", encoding="utf-8")
    (reg / "w.manifest.yaml").write_text(
        'description: "a workflow"\nworkflow: w.json\n'
        'output_node_title: "Save Image"\nflags: {}\n', encoding="utf-8")
    # Pin builtin_dir and user_dir to empty tmp dirs following test_registry.py convention
    builtin_dir = tmp_path / "builtin"
    user_dir = tmp_path / "userreg"
    builtin_dir.mkdir(parents=True, exist_ok=True)
    user_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path, builtin_dir, user_dir


def test_rack_resolves_config_and_registry_from_the_working_directory(tmp_path):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)
    assert [m.name for m in rack.list()] == ["w"]


def test_rack_describe_returns_the_manifest(tmp_path):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)
    assert rack.describe("w").description == "a workflow"


def test_rack_client_targets_the_named_machine(tmp_path):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)
    assert rack.client("remote").url == "http://b:8188"


def test_rack_client_defaults_to_the_default_machine(tmp_path, monkeypatch):
    monkeypatch.delenv("COMFY_URL", raising=False)
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)
    assert rack.client().url == "http://a:8188"


def test_rack_machine_set_at_construction_is_the_default_for_calls(tmp_path):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir, machine="remote")
    assert rack.client().url == "http://b:8188"


def test_rack_find_matches_by_substring_and_ignores_list_scope(tmp_path):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path, scope=["qwen"])
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)
    assert [m.name for m in rack.find("workflow")] == ["w"]
    # Verify that find() returns empty for nonexistent queries
    assert rack.find("nonexistent") == []


def test_rack_load_is_an_alias_for_describe(tmp_path):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)
    # load and describe should return equal manifests (load calls describe)
    assert rack.load("w") == rack.describe("w")


def test_rack_run_with_no_recipe_forwards_correctly(tmp_path, monkeypatch):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)

    # Mock runner.run to verify what Rack forwards
    call_args = {}
    def mock_run(registry, client, name, recipe=None, overrides=None, out_path=None,
                 results_dir=None, skip_preflight=False, recipe_name=None,
                 loras_dir=None, on_progress=None):
        call_args.update({"recipe": recipe, "recipe_name": recipe_name})
        return MagicMock()

    monkeypatch.setattr("comfyrack.runner.run", mock_run)
    rack.run("w")
    assert call_args["recipe"] is None
    assert call_args["recipe_name"] is None


def test_rack_run_with_string_recipe_loads_and_forwards_both_values_and_name(tmp_path, monkeypatch):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)

    # Mock recipes.load_recipe and runner.run
    call_args = {}
    def mock_load_recipe(config, name):
        return {"seed": 42}

    def mock_run(registry, client, name, recipe=None, overrides=None, out_path=None,
                 results_dir=None, skip_preflight=False, recipe_name=None,
                 loras_dir=None, on_progress=None):
        call_args.update({"recipe": recipe, "recipe_name": recipe_name})
        return MagicMock()

    monkeypatch.setattr("comfyrack.recipes.load_recipe", mock_load_recipe)
    monkeypatch.setattr("comfyrack.runner.run", mock_run)
    rack.run("w", recipe="test_recipe")
    assert call_args["recipe"] == {"seed": 42}
    assert call_args["recipe_name"] == "test_recipe"


def test_rack_run_with_dict_recipe_forwards_dict_with_no_name(tmp_path, monkeypatch):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)

    call_args = {}
    def mock_run(registry, client, name, recipe=None, overrides=None, out_path=None,
                 results_dir=None, skip_preflight=False, recipe_name=None,
                 loras_dir=None, on_progress=None):
        call_args.update({"recipe": recipe, "recipe_name": recipe_name})
        return MagicMock()

    monkeypatch.setattr("comfyrack.runner.run", mock_run)
    recipe_dict = {"seed": 99}
    rack.run("w", recipe=recipe_dict)
    assert call_args["recipe"] == recipe_dict
    assert call_args["recipe_name"] is None


def test_rack_run_with_invalid_recipe_type_raises_usage_error(tmp_path, monkeypatch):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)

    # Monkeypatch runner.run to ensure it's never reached (hermetic: no network)
    def should_not_reach(*args, **kwargs):
        raise AssertionError("runner.run should not be reached for invalid recipe type")

    monkeypatch.setattr("comfyrack.runner.run", should_not_reach)

    # Pass an invalid recipe type (list) — must raise UsageError BEFORE reaching runner.run
    with pytest.raises(errors.UsageError) as ei:
        rack.run("w", recipe=[1, 2, 3])
    assert "recipe" in ei.value.message.lower() or "recipe" in (ei.value.help_text or "").lower()


def test_rack_preflight_loads_manifest_and_graph_and_calls_check(tmp_path, monkeypatch):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)

    # Mock runner.load_graph, preflight.check, and client.object_info
    call_args = {}
    mock_graph = MagicMock()
    def mock_load_graph(manifest, client):
        call_args["load_graph_manifest"] = manifest
        return mock_graph

    def mock_check(graph, object_info, lora_meta=None, checkpoint_lineage=None,
                        requires=None):
        call_args.update({"check_graph": graph, "checkpoint_lineage": checkpoint_lineage})
        return []

    # Mock registry.family to track if it was called
    family_calls = []
    def mock_family(name):
        family_calls.append(name)
        return None

    monkeypatch.setattr("comfyrack.runner.load_graph", mock_load_graph)
    monkeypatch.setattr("comfyrack.preflight.check", mock_check)
    rack.registry.family = mock_family

    # Mock client.object_info to avoid network calls
    original_client = rack.client()
    original_client.object_info = MagicMock(return_value={})

    result = rack.preflight("w")
    assert result == []
    assert call_args["check_graph"] is mock_graph
    # Verify that registry.family() was called to resolve the lineage
    assert family_calls == ["anima"]


def test_rack_discovery_delegates_call_underlying_discover_functions(tmp_path, monkeypatch):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)

    # Mock the discovery functions
    call_tracking = {}

    def make_mock(func_name):
        def mock_func(client, *args, **kwargs):
            call_tracking[func_name] = {"args": args, "kwargs": kwargs, "client_url": client.url}
            if func_name == "node_types":
                return [{"class_type": "KSampler"}]
            elif func_name == "node_info":
                return {"class_type": "KSampler", "inputs": {}}
            elif func_name == "models":
                return {"model1": {}}
            elif func_name == "queue_summary":
                return {"queue_pending": 0}
            elif func_name == "stats_summary":
                return {"memory": {}}
        return mock_func

    monkeypatch.setattr("comfyrack.discover.node_types", make_mock("node_types"))
    monkeypatch.setattr("comfyrack.discover.node_info", make_mock("node_info"))
    monkeypatch.setattr("comfyrack.discover.models", make_mock("models"))
    monkeypatch.setattr("comfyrack.discover.queue_summary", make_mock("queue_summary"))
    monkeypatch.setattr("comfyrack.discover.stats_summary", make_mock("stats_summary"))

    # Call all discovery methods
    nodes = rack.nodes(pattern="KSampler")
    assert "node_types" in call_tracking
    # pattern is passed as a positional argument, not keyword
    assert call_tracking["node_types"]["args"][0] == "KSampler"

    node = rack.node("KSampler")
    assert "node_info" in call_tracking
    assert call_tracking["node_info"]["args"][0] == "KSampler"

    models = rack.models(type_="checkpoint")
    assert "models" in call_tracking
    # type_ is passed as a positional argument
    assert call_tracking["models"]["args"][0] == "checkpoint"

    queue = rack.queue()
    assert "queue_summary" in call_tracking

    stats = rack.stats()
    assert "stats_summary" in call_tracking


def test_rack_describe_accepts_the_family_slash_name_that_list_prints(tmp_path):
    proj_dir, builtin_dir, user_dir = setup_project(tmp_path)
    rack = Rack(start_dir=proj_dir, builtin_dir=builtin_dir, user_dir=user_dir)
    assert rack.describe("anima/w") == rack.describe("w")
    with pytest.raises(errors.NotFoundError):
        rack.describe("other/w")
