"""Tests for comfyrack.deps: resolvers, record, plan, apply on both Manager APIs.

No test reaches the network: every Lookups is built with a fake get_json and every
HTTP HEAD is a fake. tests/conftest.py additionally blocks the module defaults.
"""
import json
import pathlib

import pytest

from comfyrack import deps, errors
from comfyrack.graph import Graph

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"
_DEPS = _FIXTURES / "deps"

OBJECT_INFO = json.loads((_DEPS / "object_info.json").read_text(encoding="utf-8"))
UI_EXPORT = json.loads((_DEPS / "ui_export.json").read_text(encoding="utf-8"))


def make_lookups(registry=None, enm=None, model_list=None, comfy_org=None, search=None):
    """A deps.Lookups whose get_json answers from per-test tables.

    registry: {class_type -> comfy.org node payload}
    enm: extension-node-map.json body {repo_url: [class_type, ...]}
    model_list: model-list.json "models" entries
    comfy_org: HF API author=Comfy-Org listing entries
    search: {search query -> HF API search entries}
    Anything unlisted answers None/[] -- the same as the conftest block.
    """
    registry = registry or {}
    enm = enm or {}
    model_list = model_list or []
    comfy_org = comfy_org or []
    search = search or {}

    def fake_get_json(url, params=None):
        if url.startswith("https://api.comfy.org/"):
            class_type = url[len("https://api.comfy.org/comfy-nodes/"):-len("/node")]
            return registry.get(class_type)
        if url == deps.MANAGER_RAW + "extension-node-map.json":
            return enm
        if url == deps.MANAGER_RAW + "model-list.json":
            return {"models": model_list}
        if url == deps.HF_API:
            if params and params.get("author") == "Comfy-Org":
                return comfy_org
            if params and params.get("search"):
                return search.get(params["search"], [])
            return []
        return None

    return deps.Lookups(get_json=fake_get_json)


class _FakeResponse:
    """Just enough of a requests.Response for deps' fakes: status_code, headers,
    json(), text."""

    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class _FakePlanClient:
    """object_info(), url, and _request() -- everything plan() touches."""

    url = "http://fake:8188"

    def __init__(self, object_info, listings):
        self._object_info = object_info
        self.listings = listings  # models folder -> [filename, ...]
        self.requests = []

    def object_info(self):
        return self._object_info

    def _request(self, method, path, **kw):
        self.requests.append((method, path, kw))
        if path.startswith("/api/models/"):
            folder = path[len("/api/models/"):]
            return _FakeResponse(200, list(self.listings.get(folder, [])))
        return _FakeResponse(404, [], "not found")


class _FakeManagerClient:
    """_request() only: version routes, queue start/status/history, install POSTs.
    api is "v2", "legacy", or None (nothing answers)."""

    url = "http://fake:8188"

    def __init__(self, api, history=None, status=None, install_model_status=200):
        self.api = api
        self.requests = []
        self._history = history or {}
        self._status = status or {"is_processing": False, "pending_count": 0,
                                   "done_count": 1, "total_count": 1}
        self._install_model_status = install_model_status

    def _request(self, method, path, **kw):
        self.requests.append((method, path, kw))
        if path in ("/api/v2/manager/version", "/api/manager/version"):
            ok = self.api == ("v2" if path.startswith("/api/v2/") else "legacy")
            return (_FakeResponse(200, {"version": "1.0"}, '{"version": "1.0"}') if ok
                    else _FakeResponse(404, [], "not found"))
        if path.endswith("/queue/status"):
            return _FakeResponse(200, self._status)
        if path.endswith("/queue/history"):
            return _FakeResponse(200, self._history)
        if path.endswith("/queue/install_model"):
            return _FakeResponse(self._install_model_status, [], "bad model")
        return _FakeResponse(200, {}, "")


# -- case 1: UI exports ---------------------------------------------------------

def test_ui_nodes_walks_subgraph_nodes():
    nodes = deps.ui_nodes(UI_EXPORT)
    assert [n["type"] for n in nodes] == ["KSampler", "SomeCnrNode", "SomeAuxNode",
                                          "SubLoader"]


def test_ui_sources_returns_the_subgraph_model_with_its_url():
    classes, models = deps.ui_sources(UI_EXPORT)
    assert set(classes) == {"KSampler", "SomeCnrNode", "SomeAuxNode", "SubLoader"}
    assert classes["SomeCnrNode"] == {"cnr_id": "comfy-core"}
    assert classes["SomeAuxNode"] == {"aux_id": "owner/repo"}
    assert models == [{"name": "sub/model.safetensors",
                       "url": "https://example.com/model.safetensors",
                       "directory": "loras"}]


# -- case 2: resolve_node -> CORE ------------------------------------------------

def test_resolve_node_maps_core_python_modules_to_core():
    lookups = make_lookups()
    assert deps.resolve_node("KSampler", OBJECT_INFO, {}, None, lookups) == deps.CORE
    assert deps.resolve_node("TextGenerate", OBJECT_INFO, {}, None, lookups) == deps.CORE


def test_resolve_node_honours_a_comfy_core_cnr_id_for_a_class_object_info_lacks():
    ui_classes = {"SomeCnrNode": {"cnr_id": "comfy-core"}}
    lookups = make_lookups()
    assert deps.resolve_node("SomeCnrNode", OBJECT_INFO, ui_classes, None,
                              lookups) == deps.CORE


# -- case 3: a python_module that names the wrong folder -------------------------

def test_resolve_node_takes_the_folder_that_actually_defines_the_class(tmp_path):
    """comfyui-workflow-encrypt re-exports ComfyUI's NODE_CLASS_MAPPINGS, so
    /object_info tags InpaintCropImproved with the encrypt pack's module even
    though the class lives in the inpaint pack. A folder counts only if its
    source names the class."""
    custom = tmp_path / "custom_nodes"
    wrong = custom / "comfyui-workflow-encrypt"
    wrong.mkdir(parents=True)
    (wrong / "__init__.py").write_text("# workflow encrypt pack\n", encoding="utf-8")
    right = custom / "ComfyUI-Inpaint"
    right.mkdir()
    (right / "inpaint.py").write_text(
        'NODE_CLASS_MAPPINGS = {"InpaintCropImproved": InpaintCropImproved}\n'
        "class InpaintCropImproved:\n    pass\n",
        encoding="utf-8")
    lookups = make_lookups()
    found = deps.resolve_node("InpaintCropImproved", OBJECT_INFO, {}, custom, lookups)
    assert found == {"name": "ComfyUI-Inpaint"}


# -- case 4: folder_pack ----------------------------------------------------------

def test_folder_pack_reads_registry_id_and_git_from_pyproject(tmp_path):
    folder = tmp_path / "ComfyUI-Foo"
    folder.mkdir()
    (folder / "pyproject.toml").write_text(
        '[project]\nname = "comfyui-foo"\n\n'
        '[project.urls]\nRepository = "https://github.com/x/ComfyUI-Foo.git"\n\n'
        '[tool.comfy]\nchannel = "default"\n',
        encoding="utf-8")
    assert deps.folder_pack(folder) == {
        "name": "ComfyUI-Foo", "registry_id": "comfyui-foo",
        "git": "https://github.com/x/ComfyUI-Foo"}


def test_folder_pack_with_no_pyproject_and_no_git_is_only_a_name(tmp_path):
    folder = tmp_path / "plain-pack"
    folder.mkdir()
    assert deps.folder_pack(folder) == {"name": "plain-pack"}


# -- case 5: resolve_node source precedence ----------------------------------------

def test_resolve_node_prefers_the_registry_over_a_ui_cnr_id():
    lookups = make_lookups(registry={
        "Ghost": {"id": "reg-pack", "repository": "https://github.com/x/reg-pack"}})
    ui_classes = {"Ghost": {"cnr_id": "ui-pack"}}
    assert deps.resolve_node("Ghost", {}, ui_classes, None, lookups) == {
        "name": "reg-pack", "registry_id": "reg-pack",
        "git": "https://github.com/x/reg-pack"}


def test_resolve_node_falls_back_to_aux_id_when_the_registry_has_nothing():
    lookups = make_lookups()
    ui_classes = {"Ghost": {"aux_id": "owner/repo"}}
    assert deps.resolve_node("Ghost", {}, ui_classes, None, lookups) == {
        "name": "repo", "git": "https://github.com/owner/repo"}


def test_resolve_node_uses_the_extension_node_map_when_it_names_one_repo():
    lookups = make_lookups(enm={"https://github.com/owner/one": ["Ghost"]})
    assert deps.resolve_node("Ghost", {}, {}, None, lookups) == {
        "name": "one", "git": "https://github.com/owner/one"}


def test_resolve_node_gives_up_when_the_extension_node_map_names_two_repos():
    lookups = make_lookups(enm={"https://github.com/owner/one": ["Ghost"],
                               "https://github.com/owner/two": ["Ghost"]})
    assert deps.resolve_node("Ghost", {}, {}, None, lookups) is None


# -- case 6: resolve_model -----------------------------------------------------------

def test_resolve_model_prefers_the_ui_url():
    ui_models = [{"name": "sub/model.safetensors", "url": "https://ui",
                  "directory": "loras"}]
    existing = [{"name": "model.safetensors", "url": "https://existing",
                 "directory": "checkpoints"}]
    lookups = make_lookups(model_list=[{"filename": "model.safetensors",
                                        "url": "https://ml"}])
    assert deps.resolve_model("model.safetensors", "lora_name", ui_models, existing,
                              lookups) == {"directory": "loras",
                                           "name": "model.safetensors",
                                           "url": "https://ui"}


def test_resolve_model_falls_back_to_the_existing_record_url():
    existing = [{"name": "model.safetensors", "url": "https://existing",
                 "directory": "checkpoints"}]
    assert deps.resolve_model("model.safetensors", "lora_name", [], existing,
                              make_lookups()) == {"directory": "checkpoints",
                                                  "name": "model.safetensors",
                                                  "url": "https://existing"}


def test_resolve_model_uses_the_model_list_entry_and_its_save_path_folder():
    lookups = make_lookups(model_list=[{"filename": "model.safetensors",
                                        "url": "https://ml",
                                        "save_path": "loras/sub"}])
    # sam_model_name is not in PARAM_DIRS, so the record's folder is the only
    # directory hint and the model list's save_path segment wins.
    assert deps.resolve_model("model.safetensors", "sam_model_name", [], [],
                              lookups) == {"directory": "loras",
                                           "name": "model.safetensors",
                                           "url": "https://ml"}


def test_resolve_model_falls_back_to_a_hugging_face_hit():
    lookups = make_lookups(comfy_org=[{"id": "owner/repo", "downloads": 10,
                                       "siblings": [{"rfilename": "loras/model.safetensors"}]}])
    assert deps.resolve_model("model.safetensors", "sam_model_name", [], [],
                              lookups) == {
        "directory": "loras", "name": "model.safetensors",
        "url": "https://huggingface.co/owner/repo/resolve/main/loras/model.safetensors"}


def test_resolve_model_with_no_source_is_unknown():
    assert deps.resolve_model("model.safetensors", "sam_model_name", [], [],
                              make_lookups()) == {"directory": "unknown",
                                                  "name": "model.safetensors",
                                                  "source": "unknown"}


# -- case 7: Lookups.hf_file ----------------------------------------------------------

def test_hf_file_prefers_a_downloaded_repo_over_a_zero_download_mirror():
    """A search on the full stem matched only a re-upload named after the file
    (0 downloads); the shorter prefix query reached the original repo, and the
    original wins."""
    filename = "lightx2v_lightning_lora.safetensors"
    mirror = {"id": "mirror/reupload", "downloads": 0,
              "siblings": [{"rfilename": filename}]}
    original = {"id": "lightx2v/Lightning", "downloads": 5000,
                "siblings": [{"rfilename": filename}]}
    lookups = make_lookups(search={"lightx2v_lightning_lora": [mirror],
                                   "lightx2v-lightning": [original]})
    assert lookups.hf_file(filename) == ("lightx2v/Lightning", filename)


def test_hf_file_takes_the_comfy_org_listing_before_any_search():
    calls = []
    comfy = [{"id": "Comfy-Org/repackaged", "downloads": 3,
              "siblings": [{"rfilename": "loras/model.safetensors"}]}]

    def fake_get_json(url, params=None):
        calls.append(params)
        if params and params.get("author") == "Comfy-Org":
            return comfy
        raise AssertionError("no search may run when the Comfy-Org listing hits")

    lookups = deps.Lookups(get_json=fake_get_json)
    assert lookups.hf_file("model.safetensors", "loras") == (
        "Comfy-Org/repackaged", "loras/model.safetensors")
    assert calls and all(p and "search" not in p for p in calls)


# -- case 8: record() -----------------------------------------------------------------

def test_record_maps_nodes_packs_models_and_unknowns():
    graph = Graph({
        "1": {"class_type": "KSampler", "inputs": {"seed": 1}},
        "2": {"class_type": "TextGenerate", "inputs": {"text": "hi"}},
        "3": {"class_type": "InpaintCropImproved", "inputs": {}},
        "9": {"class_type": "EncryptExtra", "inputs": {}},
        "4": {"class_type": "LoadSAM3Model",
              "inputs": {"sam_model_name": "sam3.safetensors"}},
        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "a.safetensors"}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": "a.safetensors"}},
        "7": {"class_type": "MysteryNode", "inputs": {}},
        "8": {"class_type": "SomeLoader", "inputs": {"ckpt_name": "missing.safetensors"}},
    })
    lookups = make_lookups(
        registry={
            "InpaintCropImproved": {"id": "comfyui-workflow-encrypt",
                                    "repository": "https://github.com/user/comfyui-workflow-encrypt"},
            "EncryptExtra": {"id": "comfyui-workflow-encrypt",
                             "repository": "https://github.com/user/comfyui-workflow-encrypt"},
            "LoadSAM3Model": {"id": "ComfyUI-SAM3",
                              "repository": "https://github.com/user/ComfyUI-SAM3.git"},
        },
        model_list=[{"filename": "sam3.safetensors",
                     "url": "https://example.com/sam3.safetensors",
                     "save_path": "sam"}])
    existing = {
        "custom_node_packs": [{"name": "old-pack", "git": "https://github.com/old/pack"}],
        "models": [{"name": "extra.safetensors",
                    "url": "https://example.com/extra.safetensors",
                    "directory": "loras"}],
        "notes": "keep me",
    }
    requires, unknowns = deps.record(graph, OBJECT_INFO, existing=existing,
                                     lookups=lookups)

    assert requires["nodes"] == {
        "KSampler": "core", "TextGenerate": "core", "UNETLoader": "core",
        "InpaintCropImproved": "comfyui-workflow-encrypt",
        "EncryptExtra": "comfyui-workflow-encrypt",
        "LoadSAM3Model": "ComfyUI-SAM3",
        "MysteryNode": "unknown", "SomeLoader": "unknown",
    }
    # One pack entry per folder: two classes resolved to the encrypt pack.
    assert requires["custom_node_packs"] == [
        {"name": "comfyui-workflow-encrypt", "registry_id": "comfyui-workflow-encrypt",
         "git": "https://github.com/user/comfyui-workflow-encrypt"},
        {"name": "ComfyUI-SAM3", "registry_id": "ComfyUI-SAM3",
         "git": "https://github.com/user/ComfyUI-SAM3"},
    ]
    # a.safetensors is loaded by two nodes but recorded once; the existing
    # record's extra model is kept; both unplaceable models are unknown.
    assert requires["models"] == [
        {"directory": "sam", "name": "sam3.safetensors",
         "url": "https://example.com/sam3.safetensors"},
        {"directory": "diffusion_models", "name": "a.safetensors", "source": "unknown"},
        {"directory": "checkpoints", "name": "missing.safetensors", "source": "unknown"},
        {"directory": "loras", "name": "extra.safetensors",
         "url": "https://example.com/extra.safetensors"},
    ]
    assert requires["notes"] == "keep me"
    assert unknowns == ["node MysteryNode", "node SomeLoader",
                        "model a.safetensors", "model missing.safetensors"]


# -- case 9: replace_requires -----------------------------------------------------------

def test_replace_requires_swaps_the_block_and_keeps_the_rest_byte_identical():
    text = ('description: "wf"\n'
            'workflow: wf.json\n'
            'requires:\n'
            '  nodes:\n'
            '    KSampler: core\n'
            '  models:\n'
            '    - name: old.safetensors\n'
            '      directory: loras\n'
            'flags:\n'
            '  seed:\n'
            '    node: "3"\n')
    new = deps.replace_requires(text, {"nodes": {"KSampler": "core"}})
    assert new == ('description: "wf"\n'
                   'workflow: wf.json\n'
                   + deps.dump_requires({"nodes": {"KSampler": "core"}})
                   + 'flags:\n'
                   '  seed:\n'
                   '    node: "3"\n')


def test_replace_requires_adds_a_block_before_flags_when_none_exists():
    text = ('description: "wf"\n'
            'workflow: wf.json\n'
            'flags:\n'
            '  seed:\n'
            '    node: "3"\n')
    new = deps.replace_requires(text, {"nodes": {"KSampler": "core"}})
    assert new == ('description: "wf"\n'
                   'workflow: wf.json\n'
                   + deps.dump_requires({"nodes": {"KSampler": "core"}})
                   + 'flags:\n'
                   '  seed:\n'
                   '    node: "3"\n')


# -- case 10: plan() ---------------------------------------------------------------------

def test_plan_lists_packs_models_and_unplaced_classes():
    graph = Graph({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "b.safetensors"}},
        "2": {"class_type": "LoadSAM3Model",
              "inputs": {"sam_model_name": "sam3.safetensors"}},
        "3": {"class_type": "GhostLoader", "inputs": {"ckpt_name": "have.safetensors"}},
        "4": {"class_type": "MissingOne", "inputs": {}},
        "5": {"class_type": "MissingTwo", "inputs": {}},
        "6": {"class_type": "Mystery", "inputs": {}},
    })
    client = _FakePlanClient(
        object_info={"UNETLoader": {"input": {"required": {
            "unet_name": [["a.safetensors"], {}]}}}},
        listings={"sam": ["other.safetensors"], "loras": ["have.safetensors"]})

    class Manifest:
        name = "wf"
        requires = {
            "nodes": {"MissingOne": "pack-a", "MissingTwo": "pack-a"},
            "custom_node_packs": [{"name": "pack-a",
                                   "git": "https://github.com/x/pack-a"}],
            "models": [
                {"name": "b.safetensors", "url": "https://example.com/b.safetensors",
                 "directory": "unet"},
                {"name": "sam3.safetensors",
                 "url": "https://example.com/sam3.safetensors", "directory": "sam"},
                {"name": "have.safetensors",
                 "url": "https://example.com/have.safetensors", "directory": "loras"},
            ],
        }

    def fake_head(url):
        size = {"b.safetensors": "1024", "sam3.safetensors": "2048"}[url.rsplit("/", 1)[-1]]
        return _FakeResponse(headers={"content-length": size})

    result = deps.plan(Manifest(), graph, client, head=fake_head)

    assert result["workflow"] == "wf"
    assert result["machine"] == "http://fake:8188"
    # Two missing classes mapped to the same pack -> one entry holding both.
    assert result["packs"] == [{"name": "pack-a",
                                 "git": "https://github.com/x/pack-a",
                                 "classes": ["MissingOne", "MissingTwo"]}]
    # b.safetensors: the loader's combo lacks it, size from the fake HEAD.
    # sam3.safetensors: its class is missing, so it is checked against
    # GET /api/models/sam and the listing does not hold it.
    assert result["models"] == [
        {"directory": "unet", "name": "b.safetensors",
         "url": "https://example.com/b.safetensors", "size": 1024},
        {"directory": "sam", "name": "sam3.safetensors",
         "url": "https://example.com/sam3.safetensors", "size": 2048},
    ]
    assert result["unplaced_classes"] == ["GhostLoader", "LoadSAM3Model", "Mystery"]
    paths = [path for method, path, _kw in client.requests]
    assert "/api/models/sam" in paths and "/api/models/loras" in paths


# -- case 11: apply() through the v2 Manager ---------------------------------------------

def test_apply_queues_packs_and_models_on_the_v2_manager():
    client = _FakeManagerClient("v2", history={
        "pack:reg-pack": {"status": {"status_str": "done"}, "result": "ok"},
        "model:sub/model.safetensors": {"status": {"status_str": "installed"}},
    })
    the_plan = {
        "packs": [{"name": "reg-pack", "registry_id": "reg-pack",
                   "git": "https://github.com/x/reg-pack"}],
        "models": [{"name": "sub/model.safetensors",
                    "url": "https://example.com/model.safetensors",
                    "directory": "loras"}],
    }
    result = deps.apply(client, the_plan, lookups=make_lookups(), poll=0)

    assert result["api"] == "v2"
    assert result["queued"] == [
        {"kind": "pack", "name": "reg-pack", "ui_id": "pack:reg-pack"},
        {"kind": "model", "name": "sub/model.safetensors",
         "ui_id": "model:sub/model.safetensors"},
    ]
    assert result["results"] == {"pack:reg-pack": "done",
                                 "model:sub/model.safetensors": "installed"}
    assert result["restart"] is True

    posts = {path: kw["json"] for method, path, kw in client.requests
             if method == "POST" and "json" in kw}
    client_id = posts["/api/v2/manager/queue/task"]["client_id"]
    assert client_id.startswith("comfyrack-")
    # A registry pack goes to the task queue with id/selected_version latest.
    assert posts["/api/v2/manager/queue/task"] == {
        "ui_id": "pack:reg-pack", "client_id": client_id, "kind": "install",
        "params": {"id": "reg-pack", "version": "latest", "selected_version": "latest",
                   "mode": "remote", "channel": "default"}}
    assert posts["/api/v2/manager/queue/install_model"] == {
        "name": "model.safetensors", "filename": "model.safetensors",
        "url": "https://example.com/model.safetensors", "type": "loras", "base": "",
        "save_path": "loras", "ui_id": "model:sub/model.safetensors",
        "client_id": client_id}
    # start, then status polling with the client_id, then history.
    assert ("POST", "/api/v2/manager/queue/start",
            {"timeout": 60, "return_error_response": True}) in client.requests
    status_calls = [kw for method, path, kw in client.requests
                    if method == "GET" and path == "/api/v2/manager/queue/status"]
    assert status_calls and status_calls[0]["params"] == {"client_id": client_id}
    hist_calls = [kw for method, path, kw in client.requests
                  if method == "GET" and path == "/api/v2/manager/queue/history"]
    assert hist_calls and hist_calls[0]["params"] == {"client_id": client_id}


# -- case 12: apply() through the legacy Manager -----------------------------------------

def test_apply_on_the_legacy_manager_marks_a_git_only_pack_unknown_and_refuses():
    client = _FakeManagerClient("legacy", install_model_status=400)
    the_plan = {
        "packs": [{"name": "git-pack", "git": "https://github.com/x/git-pack"}],
        "models": [{"name": "model.safetensors",
                    "url": "https://example.com/model.safetensors",
                    "directory": "loras"}],
    }
    result = deps.apply(client, the_plan, lookups=make_lookups(), poll=0)

    assert result["api"] == "legacy"
    posts = {path: kw["json"] for method, path, kw in client.requests
             if method == "POST" and "json" in kw}
    # A git-only pack has no registry id: version "unknown" and the git URL as
    # its only file.
    assert posts["/api/manager/queue/install"] == {
        "id": "git-pack", "version": "unknown", "selected_version": "unknown",
        "files": ["https://github.com/x/git-pack"], "channel": "default",
        "mode": "remote", "ui_id": "pack:git-pack"}
    # Manager's 400 on install_model lands in refused with the url.
    assert result["refused"] == [
        {"kind": "model", "name": "model.safetensors", "status": 400,
         "reason": "bad model", "url": "https://example.com/model.safetensors"}]
    assert result["results"] == {}


# -- case 13: apply() with no Manager ------------------------------------------------------

def test_apply_without_a_manager_raises_naming_the_flag():
    client = _FakeManagerClient(None)
    with pytest.raises(errors.ComfyrackError) as ei:
        deps.apply(client, {"packs": [], "models": []})
    assert "--enable-manager" in str(ei.value)


def test_apply_without_a_manager_says_why_and_gives_the_pip_fix():
    client = _FakeManagerClient(None)
    with pytest.raises(errors.ComfyrackError) as ei:
        deps.apply(client, {"packs": [], "models": []})
    assert "comfyui-manager package" in str(ei.value)
    assert "manager_requirements.txt" in ei.value.help_text


def test_apply_progress_counts_installed_of_queued_and_skips_repeats():
    client = _FakeManagerClient("v2", status={"is_processing": False, "pending_count": 0,
                                              "done_count": 3, "total_count": 0})
    lines = []
    the_plan = {"packs": [], "models": [{"name": "m.safetensors", "directory": "loras",
                                         "url": "https://example.com/m.safetensors"}]}
    deps.apply(client, the_plan, lookups=make_lookups(), poll=0, on_progress=lines.append)
    assert len(lines) == 1
    assert lines[0].startswith("manager: 1 of 1 installed")


def test_plan_lists_missing_loras_by_name_without_a_source():
    graph = Graph({
        "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "mine.safetensors"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "b.safetensors"}},
    })
    client = _FakePlanClient(object_info={
        "LoraLoader": {"input": {"required": {"lora_name": [["other.safetensors"], {}]}}},
        "UNETLoader": {"input": {"required": {"unet_name": [["a.safetensors"], {}]}}}},
        listings={})

    class Manifest:
        name = "wf"
        requires = {"models": [
            {"name": "mine.safetensors", "url": "https://example.com/mine.safetensors",
             "directory": "loras"},
            {"name": "b.safetensors", "url": "https://example.com/b.safetensors",
             "directory": "unet"}]}

    heads = []
    result = deps.plan(Manifest(), graph, client,
                       head=lambda url: heads.append(url) or _FakeResponse(headers={}))

    assert result["loras"] == ["mine.safetensors"]
    assert [m["name"] for m in result["models"]] == ["b.safetensors"]
    assert heads == ["https://example.com/b.safetensors"]
