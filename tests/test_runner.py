import pytest
from pathlib import Path
from comfyrack.graph import Graph
from comfyrack.registry import Manifest, Flag
from comfyrack import runner, errors

OBJECT_INFO = {"KSampler": {"input": {"required": {"seed": ["INT", {}], "steps": ["INT", {}]}}},
               "SaveImage": {"input": {"required": {"filename_prefix": ["STRING", {}]}}}}

GRAPH_DATA = {
    "3": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
          "inputs": {"seed": 1, "steps": 20}},
    "9": {"class_type": "SaveImage", "_meta": {"title": "Save Image"},
          "inputs": {"filename_prefix": "out"}},
}


class FakeClient:
    url = "http://box:8188"

    def __init__(self, entry=None, uploaded=None):
        self.entry = entry or {"outputs": {"9": {"images": [
            {"filename": "r.png", "subfolder": "", "type": "output"}]}}}
        self.submitted = None
        self.uploaded = uploaded or []

    def object_info(self, class_type=None):
        return OBJECT_INFO

    def queue_prompt(self, graph, client_id=None, prompt_id=None):
        self.submitted = graph
        self.prompt_id = prompt_id or "p1"
        return self.prompt_id

    def wait(self, prompt_id, timeout_s=1800):
        return self.entry

    def outputs(self, entry, node_id):
        from comfyrack.http import ComfyClient
        return ComfyClient.outputs(entry, node_id)

    def view(self, filename, subfolder="", type_="output"):
        return b"IMAGEBYTES"

    def upload_image(self, path):
        self.uploaded.append(str(path))
        return "uploaded/ref.png"


def a_manifest(**over):
    base = dict(name="character-scene", family="anima", layer="builtin", description="",
                workflow_path=Path("w.json"), output_node_title="Save Image",
                output_type="image",
                flags={"seed": Flag(param="seed", node="3", type="int"),
                       "steps": Flag(param="steps", node="3", type="int")})
    base.update(over)
    return Manifest(**base)


def test_loading_an_api_format_workflow_makes_no_object_info_request(tmp_path):
    """All 100 builtin workflows are already API format, and convert.load_workflow
    only needs node defs for a UI-format one -- so an eager fetch here spent a
    multi-MB round trip on every registered workflow to convert nothing."""
    import json
    wf = tmp_path / "w.json"
    wf.write_text(json.dumps(GRAPH_DATA), encoding="utf-8")

    class CountingClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.oi_calls = 0

        def object_info(self, class_type=None):
            self.oi_calls += 1
            return OBJECT_INFO

    client = CountingClient()
    m = a_manifest(workflow_path=wf)
    g = runner.load_graph(m, client)
    assert client.oi_calls == 0
    assert set(g.data) == {"3", "9"}   # it really did load, it did not no-op


def test_loading_a_ui_format_workflow_still_fetches_object_info(tmp_path):
    """The other branch: making the fetch lazy must not make it never happen, or
    every UI-format workflow breaks."""
    import json
    wf = tmp_path / "ui.json"
    wf.write_text(json.dumps({
        "nodes": [{"id": 3, "type": "KSampler", "widgets_values": [1, 20]}],
        "links": []}), encoding="utf-8")

    class CountingClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.oi_calls = 0

        def object_info(self, class_type=None):
            self.oi_calls += 1
            return OBJECT_INFO

    client = CountingClient()
    g = runner.load_graph(a_manifest(workflow_path=wf), client)
    assert client.oi_calls == 1
    assert g.data["3"]["inputs"] == {"seed": 1, "steps": 20}


def test_a_full_run_fetches_object_info_once(tmp_path, monkeypatch):
    """run() calls load_graph then preflight_graph; both used to fetch unfiltered."""
    import json
    wf = tmp_path / "w.json"
    wf.write_text(json.dumps(GRAPH_DATA), encoding="utf-8")

    calls = []

    class CountingClient(FakeClient):
        def object_info(self, class_type=None):
            calls.append(class_type)
            return OBJECT_INFO

    m = a_manifest(workflow_path=wf)
    reg = type("R", (), {"load": staticmethod(lambda n: m),
                         "family": staticmethod(lambda n: None)})()
    runner.run(reg, CountingClient(), "character-scene", overrides={"seed": 7},
               results_dir=tmp_path)
    # Preflight legitimately needs it once. load_graph must not add a second.
    assert calls == [None]


def test_patch_applies_resolved_values_to_the_right_nodes():
    g = runner.patch(Graph(GRAPH_DATA).copy(), a_manifest(), {"seed": 42, "steps": 30})
    assert g.data["3"]["inputs"]["seed"] == 42
    assert g.data["3"]["inputs"]["steps"] == 30


def test_patch_uploads_image_typed_values_and_substitutes_the_server_name(tmp_path):
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    m = a_manifest(flags={"ref": Flag(param="image", node="3", type="image")})
    data = dict(GRAPH_DATA)
    data["3"] = {"class_type": "KSampler", "_meta": {"title": "KSampler"},
                 "inputs": {"image": ""}}
    client = FakeClient()
    g = runner.patch(Graph(data).copy(), m, {"ref": str(img)}, client=client)
    assert g.data["3"]["inputs"]["image"] == "uploaded/ref.png"
    assert client.uploaded == [str(img)]


def test_run_writes_the_output_file_and_returns_its_path(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(GRAPH_DATA).copy())
    reg = type("R", (), {"load": staticmethod(lambda n: a_manifest()),
                         "family": staticmethod(lambda n: None)})()
    res = runner.run(reg, FakeClient(), "character-scene",
                     overrides={"seed": 7}, results_dir=tmp_path)
    assert res.paths[0].read_bytes() == b"IMAGEBYTES"


def test_run_with_an_unset_image_flag_uses_the_baked_ref_instead_of_crashing(
        tmp_path, monkeypatch):
    """The shape of 67 of the 100 builtin manifests: a required `type: image` flag
    over a LoadImage node with a filename baked into the workflow JSON. Left unset,
    resolve_values() falls back to that baked value; patch() used to hand it to
    open() and die with FileNotFoundError, so `comfyrack run harmonize --set
    positive_text=...` crashed on most of the shipped corpus."""
    data = {
        "20": {"class_type": "LoadImage", "_meta": {"title": "Load Image"},
               "inputs": {"image": "server_side_pic.png"}},
        "9": {"class_type": "SaveImage", "_meta": {"title": "Save Image"},
              "inputs": {"filename_prefix": "out"}},
    }
    m = a_manifest(flags={"image": Flag(param="image", node="20", type="image",
                                        required=True)})
    monkeypatch.setattr(runner, "load_graph", lambda mf, c: Graph(data).copy())
    reg = type("R", (), {"load": staticmethod(lambda n: m),
                         "family": staticmethod(lambda n: None)})()
    client = FakeClient()
    res = runner.run(reg, client, "harmonize", results_dir=tmp_path,
                     skip_preflight=True)
    assert res.paths[0].read_bytes() == b"IMAGEBYTES"
    assert client.uploaded == []
    assert client.submitted["20"]["inputs"]["image"] == "server_side_pic.png"


def test_run_writes_a_provenance_sidecar_next_to_the_output(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(GRAPH_DATA).copy())
    reg = type("R", (), {"load": staticmethod(lambda n: a_manifest()),
                         "family": staticmethod(lambda n: None)})()
    res = runner.run(reg, FakeClient(), "character-scene",
                     overrides={"seed": 7}, results_dir=tmp_path)
    side = res.paths[0].with_suffix(".json")
    assert json.loads(side.read_text())["values"]["seed"] == 7


def test_run_returns_text_for_a_text_output_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(GRAPH_DATA).copy())
    m = a_manifest(output_type="text")
    reg = type("R", (), {"load": staticmethod(lambda n: m),
                         "family": staticmethod(lambda n: None)})()
    client = FakeClient(entry={"outputs": {"9": {"string": ["a clean verdict"]}}})
    res = runner.run(reg, client, "vl-judge", results_dir=tmp_path)
    assert res.text == "a clean verdict"
    assert res.paths == []
    # The documented contract: no artifact on disk means no sidecar on disk, but the
    # record is still built and returned.
    assert list(tmp_path.rglob("*.json")) == []
    assert res.record["workflow"] == "character-scene"


def test_run_preflights_and_refuses_to_queue_an_invalid_graph(tmp_path, monkeypatch):
    bad = dict(GRAPH_DATA)
    bad["4"] = {"class_type": "NotInstalledNode", "_meta": {"title": "x"}, "inputs": {}}
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(bad).copy())
    reg = type("R", (), {"load": staticmethod(lambda n: a_manifest()),
                         "family": staticmethod(lambda n: None)})()
    client = FakeClient()
    with pytest.raises(errors.ComfyrackError) as ei:
        runner.run(reg, client, "character-scene", results_dir=tmp_path)
    assert "NotInstalledNode" in str(ei.value)
    assert client.submitted is None  # nothing was queued


def test_skip_preflight_allows_the_run_to_proceed(tmp_path, monkeypatch):
    bad = dict(GRAPH_DATA)
    bad["4"] = {"class_type": "NotInstalledNode", "_meta": {"title": "x"}, "inputs": {}}
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(bad).copy())
    reg = type("R", (), {"load": staticmethod(lambda n: a_manifest()),
                         "family": staticmethod(lambda n: None)})()
    client = FakeClient()
    runner.run(reg, client, "character-scene", results_dir=tmp_path, skip_preflight=True)
    assert client.submitted is not None


def test_run_submits_patched_graph_with_resolved_values(tmp_path, monkeypatch):
    """Finding 1: Verify resolved values actually reach the submitted graph."""
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(GRAPH_DATA).copy())
    reg = type("R", (), {"load": staticmethod(lambda n: a_manifest()),
                         "family": staticmethod(lambda n: None)})()
    client = FakeClient()
    runner.run(reg, client, "character-scene",
               overrides={"seed": 7, "steps": 50}, results_dir=tmp_path)
    assert client.submitted["3"]["inputs"]["seed"] == 7
    assert client.submitted["3"]["inputs"]["steps"] == 50


def test_submit_starts_progress_before_queue_prompt_with_the_same_prompt_id(
        tmp_path):
    """The prompt_id is generated before anything is queued and the progress
    pump starts before queue_prompt: a job that finishes before the websocket
    connects would otherwise be invisible to the pump. queue_prompt blocks until
    the pump has started, so the ordering is deterministic, not racy."""
    import threading

    class OrderClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.progress_called = threading.Event()
            self.submitted_at_progress = "unset"
            self.progress_prompt_id = None
            self.queue_prompt_id = None

        def progress(self, prompt_id, on_event, ws_factory=None, client_id=None):
            self.submitted_at_progress = self.submitted
            self.progress_prompt_id = prompt_id
            self.progress_called.set()

        def queue_prompt(self, graph, client_id=None, prompt_id=None):
            assert self.progress_called.wait(5), "progress pump never started"
            self.submitted = graph
            self.queue_prompt_id = prompt_id
            return prompt_id or "p1"

    client = OrderClient()
    runner.submit(client, Graph(GRAPH_DATA).copy(), a_manifest(),
                  results_dir=tmp_path, on_progress=lambda line: None)
    assert client.submitted_at_progress is None  # progress ran first
    assert client.progress_prompt_id == client.queue_prompt_id  # same id


def test_patch_leaves_absent_flags_unchanged(tmp_path):
    """Finding 2: Patch must not modify inputs for flags absent from values."""
    m = a_manifest(flags={
        "seed": Flag(param="seed", node="3", type="int"),
        "steps": Flag(param="steps", node="3", type="int"),
        "optional": Flag(param="optional_param", node="3", type="int")
    })
    data = dict(GRAPH_DATA)
    data["3"]["inputs"]["optional_param"] = 999  # Original value
    g = runner.patch(Graph(data).copy(), m, {"seed": 42})  # Only seed in values
    assert g.data["3"]["inputs"]["seed"] == 42
    assert g.data["3"]["inputs"]["optional_param"] == 999  # Must remain unchanged


def test_run_with_multi_image_output_and_out_path_names_correctly(tmp_path, monkeypatch):
    """Finding 3: out_path + multi-image must prevent collision via derived naming.

    Server filenames ["a.png", "result.png"] would collide with out_path basename
    under old code. Asserts both files exist with CORRECT content (catches overwrite)."""
    # FakeClient variant with two images; second has out_path's basename
    class MultiImageClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.entry = {"outputs": {"9": {"images": [
                {"filename": "a.png", "subfolder": "", "type": "output"},
                {"filename": "result.png", "subfolder": "", "type": "output"}
            ]}}}

        def view(self, filename, subfolder="", type_="output"):
            # Return different content per filename so overwrite is detectable
            if "a.png" in filename:
                return b"FIRST_IMAGE_CONTENT"
            elif "result.png" in filename:
                return b"SECOND_IMAGE_CONTENT"
            return b"IMAGEBYTES"

    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(GRAPH_DATA).copy())
    reg = type("R", (), {"load": staticmethod(lambda n: a_manifest()),
                         "family": staticmethod(lambda n: None)})()
    out_file = tmp_path / "result.png"
    res = runner.run(reg, MultiImageClient(), "character-scene",
                     out_path=str(out_file), results_dir=tmp_path)

    # Assert correct content first (proves no overwrite/data-loss)
    assert out_file.read_bytes() == b"FIRST_IMAGE_CONTENT"
    second_file = tmp_path / "result_2.png"
    assert second_file.read_bytes() == b"SECOND_IMAGE_CONTENT"

    # Assert correct paths
    assert res.paths[0] == out_file
    assert res.paths[1] == second_file


def test_out_naming_an_existing_folder_writes_into_it_under_the_default_name(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(GRAPH_DATA).copy())
    reg = type("R", (), {"load": staticmethod(lambda n: a_manifest()),
                         "family": staticmethod(lambda n: None)})()
    dest = tmp_path / "existing"
    dest.mkdir()
    res = runner.run(reg, FakeClient(), "character-scene",
                     out_path=str(dest), results_dir=tmp_path / "unused")
    assert [p.parent for p in res.paths] == [dest]
    assert res.paths[0].read_bytes() == b"IMAGEBYTES"
    assert not (tmp_path / "unused").exists()


def test_patch_raises_usage_error_for_a_local_image_file_without_client(tmp_path):
    """Finding 4: a local file that needs uploading and no client is a usage error.

    The value must be a file that really exists: an image flag now only triggers an
    upload when it names a local file, so a made-up path exercises the pass-through
    branch instead and proves nothing about this one."""
    img = tmp_path / "ref.png"
    img.write_bytes(b"x")
    m = a_manifest(flags={"ref": Flag(param="image", node="3", type="image")})
    data = dict(GRAPH_DATA)
    data["3"]["inputs"]["image"] = ""

    with pytest.raises(errors.UsageError) as ei:
        runner.patch(Graph(data).copy(), m, {"ref": str(img)}, client=None)

    assert "image flag 'ref'" in str(ei.value)
    assert ei.value.help_text is not None


def test_patch_passes_a_server_side_image_ref_through_without_uploading():
    """The other branch. A LoadImage node always has a filename baked in, and
    resolve_values() falls back to it -- so this is what most of the builtin corpus
    actually produces. Opening it as a local path is the FileNotFoundError crash."""
    m = a_manifest(flags={"ref": Flag(param="image", node="3", type="image")})
    data = dict(GRAPH_DATA)
    data["3"]["inputs"]["image"] = ""
    client = FakeClient()
    g = runner.patch(Graph(data).copy(), m, {"ref": "server_side_pic.png"}, client=client)
    assert g.data["3"]["inputs"]["image"] == "server_side_pic.png"
    assert client.uploaded == []


def test_patch_passes_a_server_side_image_ref_through_even_with_no_client():
    """The UsageError must NOT fire for a pass-through: it is not a usage mistake to
    patch a server-side ref without a transport."""
    m = a_manifest(flags={"ref": Flag(param="image", node="3", type="image")})
    data = dict(GRAPH_DATA)
    data["3"]["inputs"]["image"] = ""
    g = runner.patch(Graph(data).copy(), m, {"ref": "server_side_pic.png"}, client=None)
    assert g.data["3"]["inputs"]["image"] == "server_side_pic.png"


def test_preflight_graph_resolves_lineage_from_family(monkeypatch):
    """Finding 5: preflight_graph must pass checkpoint_lineage resolved from family.base_model_family.

    Tests both branches:
    1. When family exists and has base_model_family, it must be forwarded.
    2. When family is None, checkpoint_lineage must be None.
    """
    # Case 1: Family with base_model_family
    check_kwargs_case1 = {}
    def mock_check_case1(graph, object_info, lora_meta=None, checkpoint_lineage=None,
                        requires=None):
        check_kwargs_case1.update({"checkpoint_lineage": checkpoint_lineage})
        return []

    family_obj = type("F", (), {"base_model_family": "anima"})()
    registry = type("R", (), {"family": lambda self, n: family_obj})()
    client = FakeClient()
    monkeypatch.setattr("comfyrack.preflight.check", mock_check_case1)

    result = runner.preflight_graph(registry, client, a_manifest(), Graph(GRAPH_DATA).copy())
    assert result == []
    assert check_kwargs_case1["checkpoint_lineage"] == "anima", \
        f"Expected checkpoint_lineage='anima', got {check_kwargs_case1['checkpoint_lineage']}"

    # Case 2: Family is None
    check_kwargs_case2 = {}
    def mock_check_case2(graph, object_info, lora_meta=None, checkpoint_lineage=None,
                        requires=None):
        check_kwargs_case2.update({"checkpoint_lineage": checkpoint_lineage})
        return []

    registry_no_family = type("R", (), {"family": lambda self, n: None})()
    monkeypatch.setattr("comfyrack.preflight.check", mock_check_case2)

    result = runner.preflight_graph(registry_no_family, client, a_manifest(), Graph(GRAPH_DATA).copy())
    assert result == []
    assert check_kwargs_case2["checkpoint_lineage"] is None, \
        f"Expected checkpoint_lineage=None, got {check_kwargs_case2['checkpoint_lineage']}"
