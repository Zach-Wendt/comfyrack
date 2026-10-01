"""Superflow gaps: describe form metadata, multi-output run, `requires:`."""
import json
from pathlib import Path

import requests

from comfyrack import Rack, discover, runner
from comfyrack.cli.main import main
from comfyrack.graph import Graph
from comfyrack.preflight import check
from comfyrack.registry import Flag, Manifest, Registry

GRAPH = {
    "1": {"class_type": "Prompt", "_meta": {"title": "Prompt"},
          "inputs": {"layout": "auto", "content": "x"}},
    "2": {"class_type": "Batch", "_meta": {"title": "Batch"}, "inputs": {"n": 6}},
    "3": {"class_type": "SaveImage", "_meta": {"title": "Sheet"}, "inputs": {}},
    "4": {"class_type": "SaveImage", "_meta": {"title": "Full"}, "inputs": {}},
    "5": {"class_type": "PreviewAny", "_meta": {"title": "Report"}, "inputs": {}},
}
INFO = {
    "Prompt": {"input": {"required": {
        "layout": [["auto", "bento", "flow"], {}],
        "content": ["STRING", {"multiline": True, "min_length": 1, "max_length": 4000}]}}},
    "Batch": {"input": {"required": {"n": ["INT", {"min": 1, "max": 64}]}}},
}
YAML = """description: d
workflow: w.json
output_node_title: Sheet
outputs:
  - {title: Sheet, kind: image}
  - {title: Full, kind: image}
  - {title: Report, kind: text}
requires:
  custom_node_packs:
    - {name: Pack, git: https://example.com/pack}
flags:
  layout: {node: '1', param: layout, static_ignored: 1, choices: [old]}
  content: {node: '1', param: content}
  n: {node: '2', param: n, type: int, min: 0}
"""


def project(tmp_path, monkeypatch):
    from comfyrack import config, registry
    monkeypatch.setattr(config, "USER_CONFIG", tmp_path / "none")
    monkeypatch.setattr(registry, "USER_DIR", tmp_path / "u")
    monkeypatch.setattr(registry, "BUILTIN_DIR", tmp_path / "b")
    d = tmp_path / ".comfyrack"
    d.mkdir()
    (d / "config.toml").write_text('[machines]\ndefault = "http://a:8188"\n')
    fam = d / "registry" / "f"
    fam.mkdir(parents=True)
    (fam / "w.json").write_text(json.dumps(GRAPH))
    (fam / "wf.manifest.yaml").write_text(YAML)
    monkeypatch.chdir(tmp_path)


def test_manifest_parses_outputs_requires_and_static_form_keys(tmp_path, monkeypatch):
    project(tmp_path, monkeypatch)
    m = Rack().describe("wf")
    assert m.outputs[2] == {"title": "Report", "kind": "text"}
    assert m.requires["custom_node_packs"][0]["git"] == "https://example.com/pack"
    assert m.flags["layout"].choices == ["old"] and m.flags["n"].min == 0


def test_describe_json_without_comfyui_is_manifest_only(tmp_path, monkeypatch, capsys):
    project(tmp_path, monkeypatch)

    def down(self, *a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests.Session, "get", down)
    assert main(["--json", "describe", "wf"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["live"] is False
    assert data["flags"]["layout"]["choices"] == ["old"]      # static fallback
    assert "max" not in data["flags"]["n"]
    assert data["requires"]["custom_node_packs"][0]["name"] == "Pack"
    assert data["outputs"][0]["title"] == "Sheet"


def test_describe_json_merges_live_object_info_over_static(tmp_path, monkeypatch, capsys):
    project(tmp_path, monkeypatch)
    seen = []

    class Resp:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def get(self, url, **kw):
        seen.append(url)
        cls = url.rsplit("/", 1)[1]
        return Resp({cls: INFO[cls]})
    monkeypatch.setattr(requests.Session, "get", get)
    assert main(["--json", "describe", "wf"]) == 0
    data = json.loads(capsys.readouterr().out)
    fl = data["flags"]
    assert data["live"] is True
    assert fl["layout"]["choices"] == ["auto", "bento", "flow"]   # live beats static
    assert (fl["content"]["min_length"], fl["content"]["max_length"]) == (1, 4000)
    assert (fl["n"]["min"], fl["n"]["max"]) == (1, 64)             # live beats static min 0
    assert all("/object_info/" in u for u in seen)                 # scoped, never the full dump


def test_form_meta_reads_the_newer_combo_shape():
    class C:
        def object_info(self, ct, timeout=0):
            return {ct: {"input": {"required": {"p": ["COMBO", {"options": ["a", "b"]}]}}}}
    m = Manifest(name="x", family="f", layer="b", description="", workflow_path=Path("w"),
                 output_node_title="t", flags={"p": Flag(param="p", node="1")})
    g = Graph({"1": {"class_type": "N", "inputs": {"p": "a"}}})
    assert discover.form_meta(m, g, C()) == {"p": {"choices": ["a", "b"]}}


class Client:
    url = "http://box"

    def __init__(self, entry):
        self.entry = entry

    def queue_prompt(self, graph, client_id=None, prompt_id=None):
        return prompt_id or "p"

    def wait(self, pid, timeout_s=1800):
        return self.entry

    def outputs(self, entry, node_id):
        from comfyrack.http import ComfyClient
        return ComfyClient.outputs(entry, node_id)

    def view(self, filename, subfolder="", type_="output"):
        return b"IMG"


def test_submit_returns_every_listed_output_keyed_by_title(tmp_path, monkeypatch):
    project(tmp_path, monkeypatch)
    m = Rack().describe("wf")
    img = lambda n: {"images": [{"filename": n, "subfolder": "", "type": "output"}]}
    client = Client({"outputs": {"3": img("sheet.png"),
                                 "4": {"images": [{"filename": "c1.png"}, {"filename": "c2.png"}]},
                                 "5": {"text": ["all words ok\n"]}}})
    res = runner.submit(client, Graph(GRAPH).copy(), m, results_dir=tmp_path / "r")
    assert [p.name for p in res.paths] == ["sheet.png"]
    assert res.outputs["Sheet"]["paths"] == [str(p) for p in res.paths]
    assert [Path(p).name for p in res.outputs["Full"]["paths"]] == ["c1.png", "c2.png"]
    assert res.outputs["Report"] == {"text": "all words ok"}
    assert all(Path(p).is_file() for p in res.outputs["Full"]["paths"])


def test_submit_without_outputs_key_is_unchanged(tmp_path):
    m = Manifest(name="x", family="f", layer="b", description="", workflow_path=Path("w"),
                 output_node_title="Sheet")
    client = Client({"outputs": {"3": {"images": [{"filename": "s.png"}]}}})
    res = runner.submit(client, Graph(GRAPH).copy(), m, results_dir=tmp_path)
    assert res.outputs == {} and len(res.paths) == 1


def test_preflight_names_the_missing_pack_by_git_url():
    req = {"custom_node_packs": [{"name": "Pack", "git": "https://example.com/pack"}]}
    problems = check(Graph({"1": {"class_type": "Nope", "inputs": {}}}), {}, requires=req)
    assert "Pack (https://example.com/pack)" in problems[0].detail
    assert "needs" not in check(Graph({"1": {"class_type": "Nope", "inputs": {}}}), {})[0].detail
