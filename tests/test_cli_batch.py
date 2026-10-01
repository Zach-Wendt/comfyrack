import json

import yaml
import pytest
from comfyrack import Rack, judge
from comfyrack.shots import load_shots
from comfyrack.cli.main import main

SAMPLE = {
    "sequence": 8, "primary_character": "char_a", "comfy_url": None,
    "characters": ["char_a"],
    "shots": [
        {"name": "s01", "character": "char_a", "people": "solo",
         "positive_prefix": "trig_a", "action_text": "dancing on a dock",
         "location_text": "a wooden dock", "seed": 1, "id_strength": 0.45},
        {"name": "s02", "character": "char_a", "people": "two_person",
         "positive_prefix": "trig_a", "action_text": "talking with a friend",
         "location_text": "a wooden dock", "seed": 2, "id_strength": 0.45},
    ],
}


def setup(tmp_path):
    (tmp_path / ".comfyrack").mkdir(parents=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\n\n[paths]\nshots = "shots"\n',
        encoding="utf-8")
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "seq01_shots.yaml").write_text(
        yaml.safe_dump(SAMPLE, sort_keys=False), encoding="utf-8")
    return tmp_path


def test_rubric_lists_every_shot_with_its_intended_content(tmp_path):
    sl = load_shots(setup(tmp_path) / "shots" / "seq01_shots.yaml")
    rubric = judge.build_rubric(sl)
    assert "s01" in rubric and "dancing on a dock" in rubric
    assert "s02" in rubric


def test_rubric_states_the_expected_headcount_per_shot(tmp_path):
    sl = load_shots(setup(tmp_path) / "shots" / "seq01_shots.yaml")
    rubric = judge.build_rubric(sl)
    assert "exactly 1 person" in rubric and "exactly 2 people" in rubric


def test_rubric_can_be_restricted_to_one_shot(tmp_path):
    sl = load_shots(setup(tmp_path) / "shots" / "seq01_shots.yaml")
    rubric = judge.build_rubric(sl, only="s02")
    assert "s02" in rubric and "s01" not in rubric


def test_batch_validate_reports_problems_and_exits_two(tmp_path, capsys, monkeypatch):
    root = setup(tmp_path)
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    del data["shots"][0]["seed"]
    (root / "shots" / "bad.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.chdir(root)
    code = main(["batch", "shots/bad.yaml", "--validate"])
    out = capsys.readouterr().out
    assert code == 2
    assert "seed" in out


def test_batch_validate_on_a_clean_list_exits_zero(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(setup(tmp_path))
    code = main(["batch", "shots/seq01_shots.yaml", "--validate"])
    out = capsys.readouterr().out
    assert code == 0
    assert "2 shots" in out


def test_batch_dry_run_lists_shots_and_their_requirements(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(setup(tmp_path))
    code = main(["batch", "shots/seq01_shots.yaml", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "s01" in out and "s02" in out


def test_batch_on_a_missing_file_exits_one(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(setup(tmp_path))
    code = main(["batch", "shots/nope.yaml", "--validate"])
    assert code == 1
    assert capsys.readouterr().out.startswith("error:")


# -- render path -------------------------------------------------------------
#
# Everything above only exercises --validate/--dry-run/a missing file, none of
# which reach cmd_batch's actual render call: `outcome = run_batch(...)` then
# `results = outcome["results"]`. run_batch() used to return a bare list; a
# task-13-era cmd_batch written against that old contract did
# `results = run_batch(...)` then `for r in results`, which -- against the new
# dict return -- iterates the dict's two KEYS ("results", "index_errors") as
# strings, and `r["shot"]` on a string crashes immediately. This was invisible
# because nothing drove cmd_batch through an actual render. These tests do.

def setup_render(tmp_path, id_flags=True):
    """Like setup(), plus a project-registry workflow so `--workflow` has a
    manifest to resolve and runner.run() has a graph to load/preflight/submit
    against a FakeRenderClient instead of a live ComfyUI instance. Two
    machines (`a`, `b`) so the index_errors test can make one of them dead.

    The workflow mirrors the shape of the real registry/anima/character-scene
    one for every flag run_batch submits: a LoraLoader whose strengths are
    baked to 0.0 and expected to arrive per shot, a positive and a negative
    CLIPTextEncode, and a KSampler seed. The earlier stub declared only `seed`,
    so nothing in this file could ever have caught run_batch failing to pass a
    shot's `id_strength` through -- the workflow had no id-strength knob to
    miss. `id_flags=False` reproduces that stub deliberately, for the test that
    a workflow with no identity LoRA refuses shots that declare one.
    """
    root = setup(tmp_path)
    (root / ".comfyrack" / "config.toml").write_text(
        '[machines]\na = "http://a:8188"\nb = "http://b:8188"\n\n'
        '[paths]\nshots = "shots"\n',
        encoding="utf-8")
    reg_dir = root / ".comfyrack" / "registry" / "testfam"
    reg_dir.mkdir(parents=True)
    (reg_dir / "test-native.json").write_text(json.dumps({
        "3": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
              "inputs": {"seed": 1, "steps": 20}},
        "5": {"class_type": "LoraLoader", "_meta": {"title": "ID LoRA"},
              "inputs": {"lora_name": "char_a.safetensors",
                         "strength_model": 0.0, "strength_clip": 0.0}},
        "6": {"class_type": "CLIPTextEncode", "_meta": {"title": "Positive Prompt"},
              "inputs": {"text": ""}},
        "7": {"class_type": "CLIPTextEncode", "_meta": {"title": "Negative Prompt"},
              "inputs": {"text": ""}},
        "9": {"class_type": "SaveImage", "_meta": {"title": "Save Image"},
              "inputs": {"filename_prefix": "out"}},
    }), encoding="utf-8")
    id_flag_yaml = (
        '  id_strength_model:\n    node: "5"\n    param: strength_model\n'
        '    type: float\n'
        '  id_strength_clip:\n    node: "5"\n    param: strength_clip\n'
        '    type: float\n') if id_flags else ""
    (reg_dir / "test-native.manifest.yaml").write_text(
        'description: "test workflow"\nworkflow: test-native.json\n'
        'output_node_title: "Save Image"\n'
        'flags:\n  seed:\n    node: "3"\n    param: seed\n    type: int\n'
        '  positive_text:\n    node: "6"\n    param: text\n    type: str\n'
        '  negative_text:\n    node: "7"\n    param: text\n    type: str\n'
        + id_flag_yaml,
        encoding="utf-8")
    return root


class FakeRenderClient:
    """A transport double for comfyrack.http.ComfyClient covering everything
    a real render call touches: dispatch.index() (object_info) and
    runner.run() (object_info/queue_prompt/wait/outputs/view)."""
    OI = {"KSampler": {"input": {"required": {"seed": ["INT", {}], "steps": ["INT", {}]}}},
         "LoraLoader": {"input": {"required": {
             "lora_name": [["char_a.safetensors"], {}],
             "strength_model": ["FLOAT", {}], "strength_clip": ["FLOAT", {}]}}},
         "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {}]}}},
         "SaveImage": {"input": {"required": {"filename_prefix": ["STRING", {}]}}}}

    def __init__(self, url):
        self.url = url
        self.submitted = []

    def object_info(self, class_type=None):
        return {class_type: self.OI[class_type]} if class_type else self.OI

    def queue_prompt(self, graph, client_id=None, prompt_id=None):
        self.submitted.append(graph)
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


class DeadRenderClient:
    def __init__(self, url):
        self.url = url

    def object_info(self, class_type=None):
        from comfyrack.errors import ComfyrackError
        raise ComfyrackError(f"cannot reach ComfyUI at {self.url}")


def test_batch_render_path_drives_run_batch_end_to_end(tmp_path, capsys, monkeypatch):
    """Regression for the dict-vs-list contract break: with the fix, this must
    reach run_batch()'s real render path (not --validate/--dry-run) and
    survive cmd_batch's `outcome["results"]` unpacking without raising."""
    root = setup_render(tmp_path)
    monkeypatch.setattr(Rack, "client_for_url", lambda self, url: FakeRenderClient(url))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native"])

    out = capsys.readouterr().out
    assert code == 0
    assert "count: 2/2 ok" in out


def test_batch_render_path_returns_parseable_json_with_per_shot_results(tmp_path, capsys, monkeypatch):
    root = setup_render(tmp_path)
    monkeypatch.setattr(Rack, "client_for_url", lambda self, url: FakeRenderClient(url))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] == 2
    assert {r["shot"] for r in payload["results"]} == {"s01", "s02"}
    assert all(r["status"] == "ok" for r in payload["results"])
    assert payload["index_errors"] == {}


def test_batch_render_path_surfaces_index_errors_in_text_and_json(tmp_path, capsys, monkeypatch):
    """machine `b` is dead. The batch must still complete via `a`, and the
    fact that `b` could not be indexed must show up both in the printed
    summary and in the JSON payload -- index_errors is data now, not a
    warnings.warn() side channel nothing displays."""
    root = setup_render(tmp_path)

    def factory(self, url):
        return DeadRenderClient(url) if "b" in url else FakeRenderClient(url)

    monkeypatch.setattr(Rack, "client_for_url", factory)
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native",
                "--machines", "a,b", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] == 2
    assert "b" in payload["index_errors"]
    assert "cannot reach" in payload["index_errors"]["b"]

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native",
                "--machines", "a,b"])
    text_out = capsys.readouterr().out
    assert code == 0
    assert "index_errors:" in text_out
    assert "b" in text_out


# -- id_strength reaches the submitted graph (final review, Critical) ---------
#
# The strongest form of the Critical's regression test: not "the overrides dict
# contained the key" but "the graph that was queued carries the weight". The
# workflow JSON bakes strength_model/strength_clip to 0.0 -- the real
# character-scene graph does too, because the strength is expected to arrive per
# shot -- so a graph still showing 0.0 here is exactly the faceless,
# finished-looking render the whole batch reported as `2/2 ok`.

def memoizing_factory(store, cls=None):
    """A client_for_url double that returns the SAME client per URL.

    A `lambda self, url: FakeRenderClient(url)` hands out a fresh object every
    call, so whatever it recorded is discarded before a test can look at it --
    which is why nothing here could previously assert on a submitted graph."""
    cls = cls or FakeRenderClient

    def factory(self, url):
        if url not in store:
            store[url] = cls(url)
        return store[url]
    return factory


def submitted_graphs(store):
    return [g for client in store.values() for g in getattr(client, "submitted", [])]


def test_batch_render_submits_each_shots_id_strength_into_the_graph(
        tmp_path, capsys, monkeypatch):
    root = setup_render(tmp_path)
    clients = {}
    monkeypatch.setattr(Rack, "client_for_url", memoizing_factory(clients))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native"])

    assert code == 0
    assert "count: 2/2 ok" in capsys.readouterr().out
    graphs = submitted_graphs(clients)
    assert len(graphs) == 2
    for graph in graphs:
        # Both sample shots author id_strength: 0.45. The baked value is 0.0.
        assert graph["5"]["inputs"]["strength_model"] == 0.45
        assert graph["5"]["inputs"]["strength_clip"] == 0.45


def test_batch_against_a_workflow_with_no_id_strength_flag_fails_before_rendering(
        tmp_path, capsys, monkeypatch):
    """The other shape of the same Critical: a workflow with no identity-LoRA
    knob cannot carry an authored id_strength, so the batch refuses up front
    instead of rendering every shot at whatever the graph bakes in."""
    root = setup_render(tmp_path, id_flags=False)
    clients = {}
    monkeypatch.setattr(Rack, "client_for_url", memoizing_factory(clients))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native"])

    out = capsys.readouterr().out
    assert code == 2
    assert "id_strength" in out and "s01" in out
    assert submitted_graphs(clients) == []


# -- --machine is honored, not ignored (final review, Important 3) -----------

def test_batch_machine_flag_constrains_routing_to_that_machine(
        tmp_path, capsys, monkeypatch):
    """batch is the highest-GPU-time command in the package. An earlier fix
    round made an unroutable `--machine` "not blocking", which in practice meant
    a user's explicit machine choice for a real render batch was dropped with no
    signal at all. One named machine is simply the degenerate `--machines` case:
    it becomes the whole routing map."""
    root = setup_render(tmp_path)
    clients = {}
    monkeypatch.setattr(Rack, "client_for_url", memoizing_factory(clients))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native",
                 "--machine", "b", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert {r["machine"] for r in payload["results"]} == {"b"}
    # `a` was never even probed, let alone rendered on.
    assert set(clients) == {"http://b:8188"}


def test_batch_refuses_machine_and_machines_together_rather_than_picking_one(
        tmp_path, capsys, monkeypatch):
    root = setup_render(tmp_path)
    monkeypatch.setattr(Rack, "client_for_url", memoizing_factory({}))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native",
                 "--machine", "a", "--machines", "a,b"])

    out = capsys.readouterr().out
    assert code == 2
    assert "--machine" in out and "--machines" in out
    assert "help:" in out


def test_batch_with_neither_machine_flag_still_uses_the_default(
        tmp_path, capsys, monkeypatch):
    """Ring 1 must not break: no --machine, no --machines, still renders."""
    root = setup_render(tmp_path)
    monkeypatch.setattr(Rack, "client_for_url", memoizing_factory({}))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native"])

    assert code == 0
    assert "count: 2/2 ok" in capsys.readouterr().out


# -- a fully-failed batch is detectable from the exit code (Important 7) -----

class FailingRenderClient(FakeRenderClient):
    def queue_prompt(self, graph, client_id=None, prompt_id=None):
        from comfyrack.errors import ComfyrackError
        raise ComfyrackError("CUDA out of memory")


class OneShotFailingRenderClient(FakeRenderClient):
    """Fails only the shot seeded 2, so the batch is a partial success."""

    def queue_prompt(self, graph, client_id=None, prompt_id=None):
        from comfyrack.errors import ComfyrackError
        if graph["3"]["inputs"]["seed"] == 2:
            raise ComfyrackError("CUDA out of memory")
        return super().queue_prompt(graph, client_id, prompt_id)


def test_a_batch_where_every_shot_fails_exits_non_zero(tmp_path, capsys, monkeypatch):
    """Only the `count: 0/2` line used to say so, and a scripted caller
    chaining a render into an upscale or a judge pass reads the exit code, not
    the table."""
    root = setup_render(tmp_path)
    monkeypatch.setattr(Rack, "client_for_url",
                        memoizing_factory({}, FailingRenderClient))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native"])

    out = capsys.readouterr().out
    assert code == 1
    assert out.startswith("error:")
    # The per-shot detail is carried INTO the error, not lost with the table.
    assert "s01" in out and "s02" in out and "CUDA out of memory" in out
    assert "help:" in out


def test_a_batch_where_every_shot_is_unsupported_exits_non_zero(
        tmp_path, capsys, monkeypatch):
    """`unsupported` counts as "nothing rendered" too -- a shot list made
    entirely of `via: shot_builder` composites produces no images at all."""
    root = setup_render(tmp_path)
    data = {"sequence": 8, "primary_character": "char_a", "shots": [
        {"name": "s01", "via": "shot_builder",
         "base": {"character": "char_a", "seed": 1, "action_text": "x",
                  "positive_prefix": "trig_a", "location_text": "y"},
         "composite": {"seed": 2, "refs": [{"character": "char_b", "clause": "..."}]}}]}
    (root / "shots" / "via_only.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(Rack, "client_for_url", memoizing_factory({}))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/via_only.yaml", "--workflow", "test-native"])

    out = capsys.readouterr().out
    assert code == 1
    assert "unsupported" in out


def test_a_batch_with_at_least_one_ok_shot_still_exits_zero(
        tmp_path, capsys, monkeypatch):
    """Partial success is unchanged: 1 of 2 rendering is a real result the
    caller should go collect, and re-running the one that failed is a different
    action from re-running the batch."""
    root = setup_render(tmp_path)
    monkeypatch.setattr(Rack, "client_for_url",
                        memoizing_factory({}, OneShotFailingRenderClient))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native"])

    out = capsys.readouterr().out
    assert code == 0
    assert "count: 1/2 ok" in out


# -- batch progress lands on stderr too (Important 4) ------------------------

class ProgressRenderClient(FakeRenderClient):
    def progress(self, prompt_id, on_event, ws_factory=None):
        on_event({"type": "progress", "data": {"value": 10, "max": 20}})
        on_event({"type": "executing", "data": {"node": None}})


def test_batch_progress_is_labelled_per_shot_on_stderr_never_stdout(
        tmp_path, capsys, monkeypatch):
    root = setup_render(tmp_path)
    monkeypatch.setattr(Rack, "client_for_url",
                        memoizing_factory({}, ProgressRenderClient))
    monkeypatch.chdir(root)

    code = main(["batch", "shots/seq01_shots.yaml", "--workflow", "test-native",
                 "--json"])

    captured = capsys.readouterr()
    assert code == 0
    # stdout stays parseable JSON with nothing interleaved.
    assert json.loads(captured.out)["ok"] == 2
    assert "step" not in captured.out
    # ...and every progress line on stderr names the shot it belongs to.
    assert "s01: step 10/20 (50%)" in captured.err
    assert "s02: step 10/20 (50%)" in captured.err


# -- the rubric actually reaches the workflow --------------------------------
#
# Surfaced by Finding 2's fix, and only by it. judge.run() submitted an override
# literally named `prompt` -- a canonical ALIAS of `positive_text`, which the
# builtin qwen/vl-judge manifest does not declare (it declares `custom_prompt`).
# resolve_values() silently dropped the unrecognised key, so the generated
# rubric reached NOTHING: the VL model judged every sheet against whatever
# prompt the workflow JSON baked in, and returned a confident verdict about the
# wrong question. Nothing in the suite drove judge.run(), which is why it shipped.

class FakeJudgeRack:
    def __init__(self, manifest, text="s01: ok - looks right"):
        self.manifest = manifest
        self.text = text
        self.calls = []

    def describe(self, name):
        return self.manifest

    def run(self, name, overrides=None, machine=None, **kw):
        self.calls.append({"name": name, "overrides": overrides})
        return type("R", (), {"text": self.text, "paths": []})()


def a_judge_manifest(*flag_names):
    from pathlib import Path as _Path
    from comfyrack.registry import Flag, Manifest
    return Manifest(
        name="vl-judge", family="qwen", layer="builtin", description="",
        workflow_path=_Path("vl-judge.json"), output_node_title="Save Text File",
        output_type="text",
        flags={n: Flag(param=n, node="2", type="str") for n in flag_names})


def test_the_builtin_vl_judge_manifest_declares_a_prompt_flag_judge_can_target():
    """Against the REAL registry manifest, not a stand-in: this is the exact
    pairing that was broken."""
    from comfyrack import Rack
    manifest = Rack().describe("vl-judge")
    assert judge.prompt_flag(manifest) == "custom_prompt"
    assert "custom_prompt" in manifest.flags


def test_judge_run_sends_the_rubric_to_the_flag_the_manifest_declares(tmp_path):
    rack = FakeJudgeRack(a_judge_manifest("image", "custom_prompt"))
    sl = load_shots(setup(tmp_path) / "shots" / "seq01_shots.yaml")

    judge.run(rack, tmp_path / "sheet.png", sl)

    overrides = rack.calls[0]["overrides"]
    assert "custom_prompt" in overrides
    assert "s01" in overrides["custom_prompt"]      # the generated rubric
    assert "prompt" not in overrides                # not the undeclared alias


def test_judge_run_falls_back_to_positive_text_when_that_is_the_prompt_flag(tmp_path):
    rack = FakeJudgeRack(a_judge_manifest("image", "positive_text"))
    sl = load_shots(setup(tmp_path) / "shots" / "seq01_shots.yaml")

    judge.run(rack, tmp_path / "sheet.png", sl)

    assert "s01" in rack.calls[0]["overrides"]["positive_text"]


def test_a_workflow_with_no_prompt_flag_is_refused_rather_than_judged_blind(tmp_path):
    from comfyrack import errors
    rack = FakeJudgeRack(a_judge_manifest("image"))
    sl = load_shots(setup(tmp_path) / "shots" / "seq01_shots.yaml")

    with pytest.raises(errors.UsageError) as ei:
        judge.run(rack, tmp_path / "sheet.png", sl)

    assert "custom_prompt" in (ei.value.help_text or "")
    assert rack.calls == []
