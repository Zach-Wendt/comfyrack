import json
import pathlib

import pytest
import yaml
from comfyrack.cli.main import main
from comfyrack import Rack, registry


CHARACTER_SCENE_GRAPH = {
    "3": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
          "inputs": {"seed": 1, "steps": 20}},
    "9": {"class_type": "SaveImage", "_meta": {"title": "Save Image"},
          "inputs": {"filename_prefix": "out"}},
}


def setup_project(tmp_path):
    cfgdir = tmp_path / ".comfyrack"
    cfgdir.mkdir(parents=True)
    (cfgdir / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\n\n[list]\nscope = ["anima"]\n',
        encoding="utf-8")
    workflow_json = {"character-scene": CHARACTER_SCENE_GRAPH, "i2v-basic": {}}
    modes = {"character-scene": "t2i", "i2v-basic": "i2v"}
    for family, name in (("anima", "character-scene"), ("ltx2.3", "i2v-basic")):
        d = cfgdir / "registry" / family
        d.mkdir(parents=True)
        (d / f"{name}.json").write_text(json.dumps(workflow_json[name]), encoding="utf-8")
        (d / f"{name}.manifest.yaml").write_text(
            f'description: "{name} desc"\nworkflow: {name}.json\n'
            f'mode: [{modes[name]}]\n'
            f'output_node_title: "Save Image"\n'
            f'flags:\n  seed:\n    node: "3"\n    param: seed\n    type: int\n',
            encoding="utf-8")
    return tmp_path


def _pin_hermetic_registry(tmp_path, monkeypatch):
    # Hermeticity: pin the module-level registry defaults to empty tmp dirs so
    # a developer's real ~/.comfyrack/registry (or a later-populated repo-root
    # registry/) never leaks into these tests. The CLI builds its own Rack()
    # with no builtin_dir/user_dir override, so this is the only lever.
    builtin_dir = tmp_path / "empty_builtin"
    user_dir = tmp_path / "empty_user"
    builtin_dir.mkdir(parents=True, exist_ok=True)
    user_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(registry, "BUILTIN_DIR", builtin_dir)
    monkeypatch.setattr(registry, "USER_DIR", user_dir)
    # Pin USER_CONFIG to a nonexistent path under tmp_path so the host's real
    # ~/.comfyrack/config.toml never leaks into tests (spec: thread user_config).
    from comfyrack import config
    monkeypatch.setattr(config, "USER_CONFIG", tmp_path / "nonexistent_user_config")


def run_cli(tmp_path, argv, capsys, monkeypatch):
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(setup_project(tmp_path))
    code = main(argv)
    return code, capsys.readouterr().out


def run_cli_full(tmp_path, argv, capsys, monkeypatch):
    """Like run_cli but also returns stderr, for tests that must prove stdout/
    stderr separation rather than only inspecting what landed on stdout."""
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(setup_project(tmp_path))
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class FakeClient:
    """A transport double standing in for comfyrack.http.ComfyClient. Covers every
    method Rack's discovery/run/preflight/interrupt paths call, so CLI tests can
    drive commands.cmd_* end to end through main() without touching a network."""
    url = "http://fake:8188"

    OI = {
        "KSampler": {"input": {"required": {"seed": ["INT", {}], "steps": ["INT", {}]}},
                     "output": ["LATENT"]},
        "SaveImage": {"input": {"required": {"filename_prefix": ["STRING", {}]}},
                      "output": []},
        "CheckpointLoaderSimple": {"input": {"required": {
            "ckpt_name": [["anima_baseV10.safetensors", "krea2.safetensors"], {}]}},
            "output": ["MODEL", "CLIP", "VAE"]},
        "LoraLoader": {"input": {"required": {
            "lora_name": [["your_character_lora.safetensors"], {}]}}, "output": ["MODEL"]},
    }

    def __init__(self):
        self.submitted = None
        self.interrupted = False

    def object_info(self, class_type=None, timeout=60):
        if class_type:
            return {class_type: self.OI[class_type]}
        return self.OI

    def queue_status(self):
        return {"queue_running": [["x"]], "queue_pending": [["y"], ["z"]]}

    def system_stats(self):
        return {"devices": [{"name": "NVIDIA RTX 3060",
                             "vram_total": 12884901888, "vram_free": 6442450944}]}

    def queue_prompt(self, graph, client_id=None, prompt_id=None):
        self.submitted = graph
        self.prompt_id = prompt_id or "p1"
        return self.prompt_id

    def wait(self, prompt_id, timeout_s=1800):
        return {"outputs": {"9": {"images": [
            {"filename": "r.png", "subfolder": "", "type": "output"}]}}}

    def outputs(self, entry, node_id):
        from comfyrack.http import ComfyClient
        return ComfyClient.outputs(entry, node_id)

    def view(self, filename, subfolder="", type_="output"):
        return b"IMAGEBYTES"

    def upload_image(self, path):
        return "uploaded/ref.png"

    def embeddings(self):
        return ["alpha_embedding.safetensors", "zeta_embedding.safetensors"]

    def history(self, prompt_id):
        from comfyrack.errors import NotFoundError
        if prompt_id == "abc-123":
            return {"outputs": {
                "9": {"images": [{"filename": "r.png", "subfolder": "",
                                  "type": "output"}]},
                "5": {"text": ["hello"]},
            }}
        raise NotFoundError(f"no history entry for prompt {prompt_id!r}")

    def interrupt(self):
        self.interrupted = True


def _patch_fake_client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(Rack, "client", lambda self, machine=None: fake)
    return fake


def _patch_share(monkeypatch):
    """Point comfyrack.share at a fake tailscale so the `share` subcommand's
    --json case can succeed without invoking a real `tailscale` binary or hitting
    the network. Only the three seams the happy path consults are replaced; the
    dedicated tests in tests/test_share.py cover the failure branches."""
    import subprocess
    from comfyrack import share
    status = json.dumps({"BackendState": "Running", "Self": {"DNSName": "m.x.ts.net."}})
    # Already-served so enable() takes no `serve --bg` call and returns cleanly.
    served = json.dumps({"Web": {"m.x.ts.net:8188": {
        "Handlers": {"/": {"Proxy": "http://127.0.0.1:8188"}}}}})

    def fake_run(args):
        a = list(args)
        if a[:2] == ["serve", "status", "--json"]:
            return subprocess.CompletedProcess(["tailscale", *a], 0, served, "")
        return subprocess.CompletedProcess(["tailscale", *a], 0, status, "")

    monkeypatch.setattr(share, "_run", fake_run)
    monkeypatch.setattr(share, "_answers", lambda url: True)
    monkeypatch.setattr(share.shutil, "which",
                        lambda name: "/fake/tailscale" if name == "tailscale" else None)


def test_bare_invocation_shows_live_data_not_help_text(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, [], capsys, monkeypatch)
    assert code == 0
    assert "anima" in out
    assert "usage:" not in out.lower()


def test_bare_invocation_with_json_emits_parseable_json(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["--json"], capsys, monkeypatch)
    assert code == 0
    data = json.loads(out)
    # The home payload became an object when shadowing was added to it: an agent
    # reading --json must see shadowing too, not only the human reading the table.
    assert "anima" in {d["family"] for d in data["families"]}
    assert data["shadowed"] == []


def test_list_collapses_out_of_scope_families_to_a_count_line(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["list"], capsys, monkeypatch)
    assert code == 0
    assert "character-scene" not in out          # families are summarised, not expanded
    assert "ltx2.3" in out and "--family ltx2.3" in out


def test_list_with_no_expansion_flags_and_json_emits_parseable_json(tmp_path, capsys, monkeypatch):
    # `comfyrack list --json` (no --family, no --all, no --layer) falls through to
    # the home view -- it must still honor --json rather than silently returning
    # a table (the exact defect Finding 1 flagged).
    code, out = run_cli(tmp_path, ["list", "--json"], capsys, monkeypatch)
    assert code == 0
    data = json.loads(out)
    assert "anima" in {d["family"] for d in data["families"]}


def test_list_with_family_expands_that_family(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["list", "--family", "ltx2.3"], capsys, monkeypatch)
    assert code == 0
    assert "i2v-basic" in out


def test_list_all_expands_everything(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["list", "--all"], capsys, monkeypatch)
    assert code == 0
    assert "character-scene" in out and "i2v-basic" in out


def test_list_with_layer_filters_to_that_layer(tmp_path, capsys, monkeypatch):
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(setup_project(tmp_path))

    # Every manifest in the fixture lives in the project layer, so --layer project
    # must narrow to (not drop) both workflows...
    code = main(["list", "--all", "--layer", "project"])
    out = capsys.readouterr().out
    assert code == 0
    assert "character-scene" in out and "i2v-basic" in out

    # ...while a layer with nothing in it must yield the definitive empty state,
    # not a bare blank line and not the unfiltered result.
    code = main(["list", "--all", "--layer", "builtin"])
    out = capsys.readouterr().out
    assert code == 0
    assert "character-scene" not in out and "i2v-basic" not in out
    assert "0 found" in out


def test_list_with_mode_filters_to_that_mode(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["list", "--all", "--mode", "i2v"], capsys, monkeypatch)
    assert code == 0
    assert "i2v-basic" in out and "character-scene" not in out


def test_list_with_mode_and_json_includes_mode_field(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["list", "--all", "--mode", "i2v", "--json"],
                        capsys, monkeypatch)
    assert code == 0
    data = json.loads(out)
    assert {r["name"] for r in data} == {"i2v-basic"}
    assert data[0]["mode"] == ["i2v"]


def test_list_with_mode_that_matches_nothing_is_a_definitive_zero(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["list", "--all", "--mode", "v2v"], capsys, monkeypatch)
    assert code == 0
    assert "0 found" in out


def test_find_searches_outside_scope(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["find", "i2v"], capsys, monkeypatch)
    assert code == 0 and "i2v-basic" in out


def test_find_with_mode_filters_matches(tmp_path, capsys, monkeypatch):
    # Both fixture workflows match the query "e" (character-scene, i2v-basic), but
    # only i2v-basic carries mode i2v -- so --mode must narrow, not just search.
    code, out = run_cli(tmp_path, ["find", "-", "--mode", "i2v"], capsys, monkeypatch)
    assert code == 0
    assert "i2v-basic" in out and "character-scene" not in out


def test_find_with_mode_and_json_includes_mode_field(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["find", "-", "--mode", "i2v", "--json"],
                        capsys, monkeypatch)
    assert code == 0
    data = json.loads(out)
    assert data and all("i2v" in r["mode"] for r in data)


def test_find_with_no_matches_prints_a_definitive_zero(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["find", "zzzz"], capsys, monkeypatch)
    assert code == 0
    assert "0 found" in out


def test_describe_prints_the_flag_table(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["describe", "character-scene"], capsys, monkeypatch)
    assert code == 0 and "seed" in out and "int" in out


def test_describe_unknown_workflow_exits_one_with_a_suggestion(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["describe", "character-scen"], capsys, monkeypatch)
    assert code == 1
    assert out.startswith("error:")
    assert "character-scene" in out


def test_unknown_flag_exits_two_and_lists_valid_flags(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["list", "--famly", "anima"], capsys, monkeypatch)
    assert code == 2
    assert "--famly" in out and "--family" in out


def test_unknown_flag_on_a_different_subcommand_lists_that_subcommands_flags(tmp_path, capsys, monkeypatch):
    # Pins the routing: the fix must key off the failing subcommand, not just
    # special-case `list`. `run`'s own flags (e.g. --recipe) must appear, and
    # `list`'s flags (e.g. --family) must not be what gets reported here.
    code, out = run_cli(tmp_path, ["run", "character-scene", "--bogus"], capsys, monkeypatch)
    assert code == 2
    assert "--bogus" in out and "--recipe" in out


def test_unknown_subcommand_exits_two(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["frobnicate"], capsys, monkeypatch)
    assert code == 2
    assert out.startswith("error:")


def test_json_flag_emits_parseable_json(tmp_path, capsys, monkeypatch):
    code, out = run_cli(tmp_path, ["list", "--all", "--json"], capsys, monkeypatch)
    assert code == 0
    assert {r["name"] for r in json.loads(out)} == {"character-scene", "i2v-basic"}


# -- stdout/stderr separation (Finding 3) ------------------------------------

def test_success_path_writes_only_to_stdout_never_stderr(tmp_path, capsys, monkeypatch):
    code, out, err = run_cli_full(tmp_path, ["list", "--all"], capsys, monkeypatch)
    assert code == 0
    assert err == ""
    assert "i2v-basic" in out


def test_error_path_writes_only_to_stdout_with_a_help_line_never_stderr(tmp_path, capsys, monkeypatch):
    code, out, err = run_cli_full(tmp_path, ["describe", "character-scen"], capsys, monkeypatch)
    assert code == 1
    assert err == ""
    assert out.startswith("error:")
    assert "help:" in out


# -- shadowing is reported, not resolved silently (final review, Important 5) --

def _shadowing_builtin(tmp_path, monkeypatch):
    """Give the builtin layer a manifest whose name the project layer also uses,
    so the project copy wins by precedence."""
    builtin = tmp_path / "shadow_builtin" / "anima"
    builtin.mkdir(parents=True)
    (builtin / "character-scene.json").write_text(
        json.dumps(CHARACTER_SCENE_GRAPH), encoding="utf-8")
    (builtin / "character-scene.manifest.yaml").write_text(
        'description: "the shipped one"\nworkflow: character-scene.json\n'
        'output_node_title: "Save Image"\n', encoding="utf-8")
    monkeypatch.setattr(registry, "BUILTIN_DIR", tmp_path / "shadow_builtin")


def test_home_reports_a_project_manifest_shadowing_a_builtin(tmp_path, capsys, monkeypatch):
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(setup_project(tmp_path))
    _shadowing_builtin(tmp_path, monkeypatch)

    code = main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "shadowed:" in out
    # Name the entry and both layers -- "1 name(s)" alone would not tell an agent
    # which workflow it is now silently getting a different version of.
    assert "character-scene" in out
    assert "project over builtin" in out


def test_home_says_nothing_about_shadowing_when_nothing_shadows(
        tmp_path, capsys, monkeypatch):
    """The other half: a noise line on every run would train agents to ignore it."""
    code, out = run_cli(tmp_path, [], capsys, monkeypatch)
    assert code == 0
    assert "shadowed" not in out


def test_home_json_carries_the_shadowed_entries(tmp_path, capsys, monkeypatch):
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(setup_project(tmp_path))
    _shadowing_builtin(tmp_path, monkeypatch)

    code = main(["--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["shadowed"] == [
        {"name": "character-scene", "winner": "project", "shadowed": "builtin"}]


# -- --json is honored by EVERY subcommand (final review, Important 4) -------
#
# Eight of fifteen accepted --json and returned a table or newline-joined paths
# anyway. Accepting a flag and ignoring it is worse than rejecting it: the
# loud-unknown-flag rule exists precisely so an agent can detect this, and silent
# acceptance defeats it. Enumerate every subcommand so a new one cannot slip
# through, and assert the parse actually succeeds rather than that the string
# merely looks JSON-ish.

def _all_subcommand_argvs(src_json):
    return [
        pytest.param([], id="bare"),
        pytest.param(["list", "--all"], id="list"),
        pytest.param(["find", "i2v"], id="find"),
        pytest.param(["describe", "character-scene"], id="describe"),
        pytest.param(["run", "character-scene", "--skip-preflight"], id="run"),
        pytest.param(["preflight", "character-scene"], id="preflight"),
        pytest.param(["nodes"], id="nodes"),
        pytest.param(["node", "KSampler"], id="node"),
        pytest.param(["models"], id="models"),
        pytest.param(["upload", src_json], id="upload"),
        pytest.param(["outputs", "abc-123"], id="outputs"),
        pytest.param(["machines"], id="machines"),
        pytest.param(["queue"], id="queue"),
        pytest.param(["stats"], id="stats"),
        pytest.param(["interrupt"], id="interrupt"),
        pytest.param(["onboard", src_json, "--family", "newfam", "--name", "new-wf"],
                     id="onboard"),
        pytest.param(["init"], id="init"),
        pytest.param(["setup", "character-scene"], id="setup"),
        pytest.param(["share"], id="share"),
        pytest.param(["deps", "backfill", "--family", "anima"], id="deps"),
        pytest.param(["lex", "list"], id="lex"),
        pytest.param(["prompt", "preview", "--lineage", "illustrious", "--action", "a wooden dock"],
                     id="prompt"),
        pytest.param(["char", "list"], id="char"),
        pytest.param(["subject", "list"], id="subject"),
        pytest.param(["loc", "list"], id="loc"),
        pytest.param(["scene", "list"], id="scene"),
        pytest.param(["recipe", "list"], id="recipe"),
        pytest.param(["batch", "__SHOTS__", "--validate"], id="batch"),
        pytest.param(["judge", "sheet.png", "--against", "__SHOTS__", "--rubric-only"],
                     id="judge"),
    ]


@pytest.mark.parametrize("argv", _all_subcommand_argvs("__SRC__"))
def test_every_subcommand_honors_json(argv, tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    _patch_share(monkeypatch)
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(setup_project(tmp_path))
    src = tmp_path / "src.json"
    src.write_text(json.dumps(CHARACTER_SCENE_GRAPH), encoding="utf-8")
    # A minimal clean shot list: two shots, both with seeds, so `validate` returns
    # no problems and `batch --validate` / `judge --rubric-only` need neither a
    # GPU nor a live ComfyUI to exit 0.
    shots_yaml = tmp_path / "shots.yaml"
    shots_yaml.write_text(yaml.safe_dump({"shots": [
        {"name": "s01", "action_text": "a scene", "seed": 1},
        {"name": "s02", "action_text": "another scene", "seed": 2},
    ]}), encoding="utf-8")
    argv = [str(src) if a == "__SRC__" else str(shots_yaml) if a == "__SHOTS__" else a
            for a in argv]

    code = main(argv + ["--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.out
    assert captured.err == ""
    # json.loads on the WHOLE stdout: a command that emitted its table and then a
    # JSON blob, or emitted a bare table, both fail here.
    parsed = json.loads(captured.out)
    # ...and a structured payload, not a scalar. json.dumps("a\nb") is perfectly
    # valid JSON, so parseability alone still passes for a command that just
    # stringified its table -- which is the defect being fixed.
    assert isinstance(parsed, (dict, list)), f"scalar JSON payload: {parsed!r}"


def test_the_json_subcommand_list_covers_every_registered_subcommand():
    """Guards the guard: a new subcommand added without a --json case above would
    otherwise leave this whole parametrization silently incomplete."""
    from comfyrack.cli.main import build_parser
    registered = set(build_parser().subparsers_by_name)
    covered = {p.values[0][0] for p in _all_subcommand_argvs("x") if p.values[0]}
    assert covered == registered


def test_run_json_returns_the_paths_as_data_not_a_newline_joined_string(
        tmp_path, capsys, monkeypatch):
    """`comfyrack run --json` used to return newline-joined paths. Pin the actual
    payload, not just that something parsed."""
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["run", "character-scene", "--skip-preflight", "--json"],
                        capsys, monkeypatch)
    assert code == 0
    data = json.loads(out)
    assert data["workflow"] == "character-scene"
    assert [pathlib.Path(p).name for p in data["paths"]] == ["r.png"]
    assert data["provenance"]["workflow"] == "character-scene"


def test_preflight_json_returns_the_problem_list_not_an_ascii_table(
        tmp_path, capsys, monkeypatch):
    """`comfyrack preflight --json` used to return the ASCII table. Drive the
    problems branch, not just the clean one, so an empty list cannot pass for a
    correct render."""
    fake = _patch_fake_client(monkeypatch)
    monkeypatch.setitem(fake.OI, "KSampler", {
        "input": {"required": {"seed": [["only-this"], {}], "steps": ["INT", {}]}},
        "output": ["LATENT"]})
    code, out = run_cli(tmp_path, ["preflight", "character-scene", "--json"],
                        capsys, monkeypatch)
    assert code == 0
    data = json.loads(out)
    assert data["workflow"] == "character-scene"
    assert data["count"] == len(data["problems"]) == 1
    assert data["problems"][0]["kind"] == "invalid_combo_value"
    assert data["problems"][0]["node"] == "3"


def test_json_before_the_subcommand_is_not_silently_dropped(tmp_path, capsys, monkeypatch):
    """argparse copies a subparser's whole namespace onto the parent's, so a plain
    `default=False` on the subparser's own --json overwrote a top-level --json back
    to False -- accepting the flag and ignoring it, in the other position."""
    code, out = run_cli(tmp_path, ["--json", "list", "--all"], capsys, monkeypatch)
    assert code == 0
    assert {r["name"] for r in json.loads(out)} == {"character-scene", "i2v-basic"}


def test_machine_before_the_subcommand_is_not_silently_dropped(tmp_path, capsys, monkeypatch):
    """Same clobbering, same fix, different flag: `comfyrack --machine box stats`
    must reach Rack as 'box' rather than None."""
    seen = []

    def record(self, machine=None):
        seen.append(machine)
        return FakeClient()

    monkeypatch.setattr(Rack, "client", record)
    code, out = run_cli(tmp_path, ["--machine", "box", "stats"], capsys, monkeypatch)
    assert code == 0
    assert seen and seen[0] == "box"


# -- unreachable ComfyUI (final review, Critical 2) --------------------------

def test_an_unreachable_comfyui_is_a_structured_stdout_error_not_a_traceback(
        tmp_path, capsys, monkeypatch):
    """The most common runtime failure there is. Before the transport wrapper this
    escaped as `UNCAUGHT: requests.exceptions.ConnectionError` with an empty stdout.
    A real ComfyClient is used deliberately -- injecting a FakeClient here is exactly
    what made this invisible to the suite."""
    import requests
    from comfyrack.http import ComfyClient

    class DeadSession:
        def get(self, url, **kw):
            raise requests.ConnectionError("connection refused")

        def post(self, url, **kw):
            raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(
        Rack, "client",
        lambda self, machine=None: ComfyClient("http://dead:9999", session=DeadSession()))

    code, out, err = run_cli_full(tmp_path, ["stats"], capsys, monkeypatch)
    assert code == 1
    assert err == ""
    assert out.startswith("error:")
    assert "help:" in out
    assert "http://dead:9999" in out


def test_an_unexpected_exception_still_lands_on_stdout_with_exit_one(
        tmp_path, capsys, monkeypatch):
    """The last-resort handler. Any escape at all must obey the output contract
    rather than printing a traceback to stderr."""
    def boom(self, machine=None):
        raise ZeroDivisionError("nope")

    monkeypatch.setattr(Rack, "client", boom)
    code, out, err = run_cli_full(tmp_path, ["stats"], capsys, monkeypatch)
    assert code == 1
    assert err == ""
    assert out.startswith("error:")
    assert "ZeroDivisionError" in out
    assert "help:" in out


# -- commands driven through a fake transport (Finding 2) --------------------

def test_run_dispatches_through_rack_and_returns_the_output_path(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["run", "character-scene", "--skip-preflight"],
                        capsys, monkeypatch)
    assert code == 0
    assert "r.png" in out


def test_preflight_reports_zero_problems_via_the_fake_client(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["preflight", "character-scene"], capsys, monkeypatch)
    assert code == 0
    assert "0 problems" in out


def test_nodes_lists_node_types_via_the_fake_client(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["nodes"], capsys, monkeypatch)
    assert code == 0
    assert "KSampler" in out and "count: 4" in out


def test_node_shows_inputs_outputs_and_widgets_via_the_fake_client(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["node", "KSampler"], capsys, monkeypatch)
    assert code == 0
    assert "LATENT" in out and "seed" in out


def test_models_groups_by_category_via_the_fake_client(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["models"], capsys, monkeypatch)
    assert code == 0
    assert "checkpoints" in out and "anima_baseV10.safetensors" in out


def test_models_with_an_unknown_type_gives_a_definitive_empty_state(
        tmp_path, capsys, monkeypatch):
    """discover.models returns {type_: []} for an unknown --type, which rendered as
    a `bogus 0` row -- a positive-looking result for a category that does not
    exist."""
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["models", "--type", "bogus"], capsys, monkeypatch)
    assert code == 0
    assert "0 found" in out
    assert "bogus  0" not in out


def test_models_with_a_real_type_still_lists_it(tmp_path, capsys, monkeypatch):
    """The filter must drop only genuinely empty categories."""
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["models", "--type", "checkpoints"], capsys, monkeypatch)
    assert code == 0
    assert "anima_baseV10.safetensors" in out
    assert "0 found" not in out


# -- embeddings in models (covers the ComfyUI MCP embedding listing) -----------

def test_models_with_type_embeddings_lists_the_embedding_names(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["models", "--type", "embeddings"], capsys, monkeypatch)
    assert code == 0
    assert "alpha_embedding.safetensors" in out
    assert "zeta_embedding.safetensors" in out


def test_bare_models_includes_the_embeddings_category(tmp_path, capsys, monkeypatch):
    """The bare listing and `--type embeddings` must agree on one source of
    truth (Rack.models), so the category appears without a --type too."""
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["models"], capsys, monkeypatch)
    assert code == 0
    assert "embeddings" in out
    assert "alpha_embedding.safetensors" in out


def test_models_embeddings_json_carries_the_names(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["models", "--type", "embeddings", "--json"],
                        capsys, monkeypatch)
    assert code == 0
    data = json.loads(out)
    assert data["embeddings"] == ["alpha_embedding.safetensors",
                                  "zeta_embedding.safetensors"]


def test_bare_models_json_includes_the_embeddings_category(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["models", "--json"], capsys, monkeypatch)
    assert code == 0
    data = json.loads(out)
    assert data["embeddings"] == ["alpha_embedding.safetensors",
                                  "zeta_embedding.safetensors"]


# -- upload: send a local image to ComfyUI (covers the MCP upload tool) ------

def test_upload_prints_the_server_side_name(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    src = tmp_path / "pic.png"
    src.write_bytes(b"PNGBYTES")
    code, out = run_cli(tmp_path, ["upload", str(src)], capsys, monkeypatch)
    assert code == 0
    assert "uploaded/ref.png" in out


def test_upload_json_returns_the_local_path_and_server_name(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    src = tmp_path / "pic.png"
    src.write_bytes(b"PNGBYTES")
    code, out = run_cli(tmp_path, ["upload", str(src), "--json"], capsys, monkeypatch)
    assert code == 0
    assert json.loads(out) == {"path": str(src), "name": "uploaded/ref.png"}


def test_upload_missing_file_is_an_error_with_a_help_line(tmp_path, capsys, monkeypatch):
    """A missing local file must be a structured ComfyrackError, not the
    FileNotFoundError an unchecked open() inside upload_image would raise."""
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["upload", str(tmp_path / "nope.png")],
                        capsys, monkeypatch)
    assert code == 1
    assert out.startswith("error:")
    assert "help:" in out


# -- outputs: list (and fetch) a prompt's files (covers the MCP output tools) --

def test_outputs_lists_every_file_then_a_count(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["outputs", "abc-123"], capsys, monkeypatch)
    assert code == 0
    assert "r.png" in out
    assert "count: 1" in out


def test_outputs_json_carries_files_with_null_paths_and_text(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["outputs", "abc-123", "--json"], capsys, monkeypatch)
    assert code == 0
    data = json.loads(out)
    assert data["prompt_id"] == "abc-123"
    assert data["files"] == [{"node": "9", "filename": "r.png", "subfolder": "",
                              "type": "output", "path": None}]
    assert data["text"] == [{"node": "5", "text": "hello"}]


def test_outputs_with_out_downloads_each_file_into_the_dir(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    dest = tmp_path / "dl"
    code, out = run_cli(tmp_path, ["outputs", "abc-123", "--out", str(dest)],
                        capsys, monkeypatch)
    assert code == 0
    # The dir did not exist before: creating it is the command's job.
    assert (dest / "r.png").read_bytes() == b"IMAGEBYTES"
    assert str(dest / "r.png") in out


def test_outputs_unknown_prompt_id_is_an_error_naming_the_id(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["outputs", "nope-999"], capsys, monkeypatch)
    assert code == 1
    assert out.startswith("error:")
    assert "nope-999" in out


@pytest.mark.parametrize("flag,value", [
    ("--family", "../../.."),
    ("--family", "a/b"),
    ("--name", "../escape"),
    ("--name", "with space"),
])
def test_onboard_rejects_a_family_or_name_that_escapes_the_registry_root(
        flag, value, tmp_path, capsys, monkeypatch):
    """`dest = Path(layer_root) / args.family` with --family unvalidated wrote
    outside the registry root entirely."""
    _patch_fake_client(monkeypatch)
    src = tmp_path / "src.json"
    argv = ["onboard", str(src), "--family", "newfam", "--name", "new-wf"]
    argv[argv.index(flag) + 1] = value

    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(setup_project(tmp_path))
    src.write_text(json.dumps(CHARACTER_SCENE_GRAPH), encoding="utf-8")
    code = main(argv)
    out = capsys.readouterr().out
    assert code == 2
    assert out.startswith("error:")
    assert "help:" in out
    # Nothing was written anywhere above the project.
    assert not (tmp_path.parent / "newfam").exists()


def test_queue_reports_running_and_pending_via_the_fake_client(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["queue"], capsys, monkeypatch)
    assert code == 0
    assert "1 running" in out and "2 pending" in out


def test_stats_reports_device_and_vram_via_the_fake_client(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["stats"], capsys, monkeypatch)
    assert code == 0
    # Assert the whole rendered line, not `"12.0" in out`: cmd_stats renders
    # "{free} GB free of {total} GB", so swapping free and total still leaves
    # "12.0" present and the old assertion passed either way.
    assert out.strip() == ("device: NVIDIA RTX 3060\n"
                           "vram: 6.0 GB free of 12.0 GB")


def test_interrupt_sends_the_signal_via_the_fake_client(tmp_path, capsys, monkeypatch):
    fake = _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["interrupt"], capsys, monkeypatch)
    assert code == 0
    assert out.strip() == "interrupt: sent"
    assert fake.interrupted is True


# -- machines: the fleet list a backend reads with `machines --json` ----------
#
# A pure config read. The [machines] table stays the single machine list, so
# this reports what comfyrack's own config resolved to rather than probing the
# network -- a backend must be able to learn the fleet with nothing running.

MACHINES_TOML = ('[machines]\n'
                 'box = "http://box:8188"\n'
                 'alpha = "http://alpha:8188"\n'
                 'default = "http://box:8188"\n')


def _write_machines_config(tmp_path):
    """A project config with two named machines, plus a `default` entry. The
    `default` key stays in the listing: it is a machine like any other, and
    Config.load does not special-case it."""
    cfgdir = tmp_path / ".comfyrack"
    cfgdir.mkdir(parents=True, exist_ok=True)
    (cfgdir / "config.toml").write_text(MACHINES_TOML, encoding="utf-8")
    return tmp_path


def test_machines_json_lists_every_machine_in_config_order_and_the_default(
        tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("COMFY_URL", raising=False)
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(_write_machines_config(tmp_path))
    code = main(["machines", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    # Config order, not sorted: the table is the user's own list, and sorting it
    # would make two machines interchangeable in the output.
    assert data["machines"] == [{"name": "box", "url": "http://box:8188"},
                                {"name": "alpha", "url": "http://alpha:8188"},
                                {"name": "default", "url": "http://box:8188"}]
    assert data["default"] == "http://box:8188"


def test_machines_text_output_names_every_machine_with_its_url_and_the_default(
        tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("COMFY_URL", raising=False)
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(_write_machines_config(tmp_path))
    code = main(["machines"])
    out = capsys.readouterr().out
    assert code == 0
    assert "box  http://box:8188" in out
    assert "alpha  http://alpha:8188" in out
    # The last line is what work without --machine would land on, so pin it as
    # a last line rather than a substring somewhere in the middle.
    assert out.strip().splitlines()[-1] == "default: http://box:8188"


def test_machines_with_no_config_at_all_is_an_empty_list_and_the_builtin_default(
        tmp_path, capsys, monkeypatch):
    # Deliberately skip setup_project/_write_machines_config: no project config
    # exists anywhere above cwd, and _pin_hermetic_registry points USER_CONFIG at
    # a nonexistent path, so this is the true no-config case.
    monkeypatch.delenv("COMFY_URL", raising=False)
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    code = main(["machines", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["machines"] == []
    # No [machines] table still resolves a default -- that is the URL a command
    # with no --machine would use, so reporting nothing here would leave the
    # caller unable to tell "no machines" from "no idea where work goes".
    assert data["default"] == "http://127.0.0.1:8188"


def test_machines_default_follows_comfy_url_over_the_config_default(
        tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("COMFY_URL", "http://from-env:8188")
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(_write_machines_config(tmp_path))
    code = main(["machines", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    # The env var only steers the default. The configured machines are still
    # reported, and one whose name is `default` still says what it points at.
    assert data["default"] == "http://from-env:8188"
    assert [m["name"] for m in data["machines"]] == ["box", "alpha", "default"]


def test_machines_never_contacts_a_comfyui(tmp_path, capsys, monkeypatch):
    """The property the whole command exists for. A client here would make a
    backend's fleet discovery depend on every machine being reachable."""
    def boom(self, machine=None):
        raise AssertionError("`machines` must not build a client")

    monkeypatch.setattr(Rack, "client", boom)
    monkeypatch.delenv("COMFY_URL", raising=False)
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(_write_machines_config(tmp_path))
    code = main(["machines", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["default"] == "http://box:8188"


# -- onboard / init through the CLI (Task 15 review, Important 3) -----------

def test_onboard_falls_back_to_the_project_registry_when_no_layer_root_given(
        tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    src = tmp_path / "src.json"
    src.write_text(json.dumps(CHARACTER_SCENE_GRAPH), encoding="utf-8")
    code, out = run_cli(tmp_path, ["onboard", str(src), "--family", "newfam",
                                   "--name", "new-wf"], capsys, monkeypatch)
    assert code == 0
    expected = tmp_path / ".comfyrack" / "registry" / "newfam" / "new-wf.json"
    assert str(expected) in out
    assert expected.is_file()


def test_onboard_explicit_layer_root_overrides_the_project_default(
        tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    src = tmp_path / "src.json"
    src.write_text(json.dumps(CHARACTER_SCENE_GRAPH), encoding="utf-8")
    other_root = tmp_path / "elsewhere"
    code, out = run_cli(tmp_path, ["onboard", str(src), "--family", "newfam",
                                   "--name", "new-wf", "--layer-root", str(other_root)],
                        capsys, monkeypatch)
    assert code == 0
    expected = other_root / "newfam" / "new-wf.json"
    assert str(expected) in out
    assert expected.is_file()
    # must NOT have also landed in the project registry
    assert not (tmp_path / ".comfyrack" / "registry" / "newfam").exists()


def test_onboard_with_no_project_and_no_layer_root_is_a_usage_error(
        tmp_path, capsys, monkeypatch):
    # Deliberately skip setup_project: no .comfyrack/config.toml anywhere above cwd.
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src.json"
    src.write_text(json.dumps(CHARACTER_SCENE_GRAPH), encoding="utf-8")
    code = main(["onboard", str(src), "--family", "f", "--name", "n"])
    out = capsys.readouterr().out
    assert code == 2
    assert out.startswith("error:")
    assert "--layer-root" in out or "comfyrack init" in out


def test_onboard_refuses_to_overwrite_without_force_via_the_cli(
        tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    _pin_hermetic_registry(tmp_path, monkeypatch)
    monkeypatch.chdir(setup_project(tmp_path))
    src = tmp_path / "src.json"
    src.write_text(json.dumps(CHARACTER_SCENE_GRAPH), encoding="utf-8")
    argv = ["onboard", str(src), "--family", "newfam", "--name", "new-wf"]

    code1 = main(argv)
    capsys.readouterr()  # discard first call's output
    assert code1 == 0

    code2 = main(argv)
    out2 = capsys.readouterr().out
    assert code2 == 1
    assert out2.startswith("error:")
    assert "--force" in out2


def test_init_writes_a_config_via_the_cli_and_is_idempotent(tmp_path, capsys, monkeypatch):
    _pin_hermetic_registry(tmp_path, monkeypatch)
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    code1 = main(["init"])
    out1 = capsys.readouterr().out
    assert code1 == 0
    cfg = project_dir / ".comfyrack" / "config.toml"
    assert cfg.is_file()
    body1 = cfg.read_text(encoding="utf-8")

    code2 = main(["init"])
    out2 = capsys.readouterr().out
    assert code2 == 0
    assert cfg.read_text(encoding="utf-8") == body1


# -- progress goes to stderr (final review, Important 4) ---------------------
#
# Task 16's own brief said "have the CLI pass [on_progress] that writes to
# stderr" and nothing ever did: Rack.run()'s docstring claimed "a CLI passes
# one" while cmd_run passed nothing, so a five-minute render printed absolutely
# nothing until it finished. The reason it must be STDERR and not stdout is the
# output contract -- stdout carries the result and has to stay parseable -- so
# these tests assert both halves: progress appears on stderr, AND stdout is
# byte-identical to the run that produced no progress at all.

class ProgressFakeClient(FakeClient):
    """FakeClient plus the websocket progress stream.

    The base class deliberately has NO `progress` attribute, which is exactly
    what runner._start_progress checks before opening a socket -- so every
    other CLI test in this file keeps running with no thread and no websocket."""

    def progress(self, prompt_id, on_event, ws_factory=None):
        on_event({"type": "progress", "data": {"value": 1, "max": 20}})
        on_event({"type": "progress", "data": {"value": 20, "max": 20}})
        on_event({"type": "executing", "data": {"node": None}})


RUN_ARGV = ["run", "character-scene", "--out", "r.png", "--skip-preflight"]


def test_run_prints_progress_to_stderr_leaving_stdout_byte_identical(
        tmp_path, capsys, monkeypatch):
    # `--out r.png` is relative, so stdout is the same string in both project
    # dirs and a byte-for-byte comparison means what it says.
    quiet = FakeClient()
    monkeypatch.setattr(Rack, "client", lambda self, machine=None: quiet)
    code_a, out_a, err_a = run_cli_full(tmp_path / "quiet", RUN_ARGV, capsys,
                                        monkeypatch)

    talkative = ProgressFakeClient()
    monkeypatch.setattr(Rack, "client", lambda self, machine=None: talkative)
    code_b, out_b, err_b = run_cli_full(tmp_path / "loud", RUN_ARGV, capsys,
                                        monkeypatch)

    assert code_a == 0 and code_b == 0
    # A client with no progress stream produces no progress output at all.
    assert err_a == ""
    # A client that emits events produces them on STDERR...
    assert "step 1/20 (5%)" in err_b
    assert "step 20/20 (100%)" in err_b
    assert "finished" in err_b
    # ...and stdout is untouched by any of it.
    assert out_a == out_b == "r.png\n"


def test_a_library_run_with_no_callback_never_opens_the_progress_stream(
        tmp_path, capsys, monkeypatch):
    """The default stays silent: `on_progress=None` means runner._start_progress
    returns without touching client.progress() at all, so library callers and
    every pre-existing test pay for no socket and no thread."""
    from comfyrack import runner
    from comfyrack.graph import Graph

    calls = []

    class CountingProgressClient(ProgressFakeClient):
        def progress(self, prompt_id, on_event, ws_factory=None):
            calls.append(prompt_id)
            return super().progress(prompt_id, on_event, ws_factory)

    client = CountingProgressClient()
    graph = Graph(json.loads(json.dumps(CHARACTER_SCENE_GRAPH)))

    class M:
        output_node_title = "Save Image"
        output_type = "image"
        outputs = []
        family, name = "anima", "character-scene"

    runner.submit(client, graph, M(), out_path=tmp_path / "r.png")

    assert calls == []
    assert capsys.readouterr().err == ""


# -- setup / deps backfill through the CLI (deps plan + record) ----------------

def test_setup_with_nothing_missing_prints_the_empty_state(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    code, out = run_cli(tmp_path, ["setup", "character-scene"], capsys, monkeypatch)
    assert code == 0
    assert "nothing missing" in out
    assert "character-scene" in out


def test_setup_json_carries_packs_models_and_unplaced_classes(tmp_path, capsys, monkeypatch):
    from comfyrack import deps
    fake = _patch_fake_client(monkeypatch)
    plan = {
        "workflow": "character-scene",
        "machine": fake.url,
        "packs": [{"name": "pack-a", "git": "https://github.com/x/pack-a",
                   "classes": ["MissingOne"]}],
        "models": [{"directory": "loras", "name": "m.safetensors",
                    "url": "https://example.com/m.safetensors", "size": 1024}],
        "loras": ["mine.safetensors"],
        "unplaced_classes": ["Mystery"],
    }
    monkeypatch.setattr(deps, "plan", lambda manifest, graph, client: plan)
    code, out = run_cli(tmp_path, ["setup", "character-scene", "--json"], capsys, monkeypatch)
    assert code == 0
    data = json.loads(out)
    assert data["packs"] == plan["packs"]
    assert data["models"] == plan["models"]
    assert data["loras"] == ["mine.safetensors"]
    assert data["unplaced_classes"] == ["Mystery"]


def test_deps_backfill_without_write_leaves_the_manifest_unchanged(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    manifest = tmp_path / ".comfyrack" / "registry" / "anima" / "character-scene.manifest.yaml"
    code, out = run_cli(tmp_path, ["deps", "backfill", "--family", "anima"], capsys, monkeypatch)
    assert code == 0
    assert "would change" in out
    # The manifest on disk is byte-identical to the fixture: no requires block
    # was added without --write.
    assert manifest.read_text(encoding="utf-8") == (
        'description: "character-scene desc"\n'
        'workflow: character-scene.json\n'
        'mode: [t2i]\n'
        'output_node_title: "Save Image"\n'
        'flags:\n'
        '  seed:\n'
        '    node: "3"\n'
        '    param: seed\n'
        '    type: int\n')


def test_deps_backfill_with_write_saves_the_record(tmp_path, capsys, monkeypatch):
    _patch_fake_client(monkeypatch)
    manifest = tmp_path / ".comfyrack" / "registry" / "anima" / "character-scene.manifest.yaml"
    code, out = run_cli(tmp_path, ["deps", "backfill", "--family", "anima", "--write"],
                        capsys, monkeypatch)
    assert code == 0
    after = manifest.read_text(encoding="utf-8")
    assert "requires:" in after
    assert "KSampler: unknown" in after
    assert "written" in out


def test_setup_text_lists_loras_under_your_loras(tmp_path, capsys, monkeypatch):
    from comfyrack import deps
    fake = _patch_fake_client(monkeypatch)
    plan = {"workflow": "character-scene", "machine": fake.url, "packs": [], "models": [],
            "loras": ["mine.safetensors"], "unplaced_classes": []}
    monkeypatch.setattr(deps, "plan", lambda manifest, graph, client: plan)
    code, out = run_cli(tmp_path, ["setup", "character-scene"], capsys, monkeypatch)
    assert code == 0
    assert "your LoRAs" in out and "mine.safetensors" in out
    assert "1 LoRA(s) missing" in out
