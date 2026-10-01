import json
from pathlib import Path
from comfyrack.graph import Graph
from comfyrack.registry import Manifest, Flag
from comfyrack.provenance import content_hash, write_sidecar, build_record

GRAPH = Graph({"3": {"class_type": "KSampler", "inputs": {"seed": 1}}})


def a_manifest():
    return Manifest(name="character-scene", family="anima", layer="builtin",
                    description="", workflow_path=Path("w.json"),
                    output_node_title="Save Image",
                    flags={"seed": Flag(param="seed", node="3", type="int")})


def test_content_hash_is_stable_for_identical_bytes(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"same"); b.write_bytes(b"same")
    assert content_hash(a) == content_hash(b)


def test_content_hash_differs_for_different_bytes(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"one"); b.write_bytes(b"two")
    assert content_hash(a) != content_hash(b)


def test_record_captures_workflow_family_layer_machine_and_resolved_values():
    rec = build_record(a_manifest(), {"seed": 42}, "http://box:8188", GRAPH,
                       recipe_name="anima_v1")
    assert rec["workflow"] == "character-scene"
    assert rec["family"] == "anima"
    assert rec["layer"] == "builtin"
    assert rec["machine"] == "http://box:8188"
    assert rec["recipe"] == "anima_v1"
    assert rec["values"] == {"seed": 42}


def test_record_includes_the_graph_hash_so_a_changed_workflow_is_detectable():
    rec = build_record(a_manifest(), {"seed": 42}, "http://box:8188", GRAPH)
    assert rec["graph_hash"] == GRAPH.content_hash()


def test_record_has_an_iso_utc_timestamp():
    rec = build_record(a_manifest(), {}, "http://box:8188", GRAPH)
    assert rec["timestamp"].endswith("+00:00")


def test_sidecar_is_written_next_to_the_output_with_a_json_suffix(tmp_path):
    out = tmp_path / "render.png"
    out.write_bytes(b"img")
    side = write_sidecar(out, {"workflow": "character-scene"})
    assert side == tmp_path / "render.json"
    assert json.loads(side.read_text())["workflow"] == "character-scene"


def test_a_json_output_does_not_get_overwritten_by_its_own_sidecar(tmp_path):
    out = tmp_path / "verdict.json"
    out.write_text('{"verdict": "ok"}', encoding="utf-8")
    side = write_sidecar(out, {"workflow": "vl-judge"})
    assert side != out
    assert json.loads(out.read_text())["verdict"] == "ok"   # render survived
    assert json.loads(side.read_text())["workflow"] == "vl-judge"


def test_sidecar_records_the_output_content_hash(tmp_path):
    out = tmp_path / "render.png"
    out.write_bytes(b"img")
    side = write_sidecar(out, {"workflow": "w"})
    assert json.loads(side.read_text())["output_sha256"] == content_hash(out)
