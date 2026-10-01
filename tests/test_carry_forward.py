"""Plan 1's three carry-forwards, each of which was code wired to nothing.

Every test here drives the wiring end to end -- through runner.run() or
discover.embeddings() -- rather than calling the underlying unit, because "written
and tested but connected to nothing" is precisely the state these tests exist to
make impossible to return to.
"""
import json
import struct
import sys
import threading
from pathlib import Path

import pytest

from comfyrack import discover, errors, runner
from comfyrack.graph import Graph
from comfyrack.http import ComfyClient
from comfyrack.registry import Flag, Manifest

# -- fixtures ---------------------------------------------------------------

OBJECT_INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {
        "ckpt_name": [["krea2.safetensors"], {}]}}},
    "LoraLoader": {"input": {"required": {
        "lora_name": [["your_character_lora.safetensors", "second.safetensors",
                       "third.safetensors"], {}]}}},
    "SaveImage": {"input": {"required": {"filename_prefix": ["STRING", {}]}}},
}

LORA_GRAPH = {
    "1": {"class_type": "CheckpointLoaderSimple", "_meta": {"title": "ckpt"},
          "inputs": {"ckpt_name": "krea2.safetensors"}},
    "2": {"class_type": "LoraLoader", "_meta": {"title": "lora"},
          "inputs": {"lora_name": "your_character_lora.safetensors", "model": ["1", 0]}},
    "9": {"class_type": "SaveImage", "_meta": {"title": "Save Image"},
          "inputs": {"filename_prefix": "out"}},
}


class FakeClient:
    url = "http://box:8188"

    def __init__(self, entry=None):
        self.entry = entry or {"outputs": {"9": {"images": [
            {"filename": "r.png", "subfolder": "", "type": "output"}]}}}
        self.submitted = None

    def object_info(self, class_type=None):
        return OBJECT_INFO

    def queue_prompt(self, graph, client_id=None, prompt_id=None):
        self.submitted = graph
        self.prompt_id = prompt_id or "p1"
        return self.prompt_id

    def wait(self, prompt_id, timeout_s=1800):
        return self.entry

    def outputs(self, entry, node_id):
        return ComfyClient.outputs(entry, node_id)

    def view(self, filename, subfolder="", type_="output"):
        return b"IMAGEBYTES"

    def upload_image(self, path):
        return "uploaded/ref.png"


def a_manifest(**over):
    base = dict(name="character-scene", family="anima", layer="builtin",
                description="", workflow_path=Path("w.json"),
                output_node_title="Save Image", output_type="image",
                flags={"seed": Flag(param="seed", node="9", type="int")})
    base.update(over)
    return Manifest(**base)


def a_registry(lineage="flux", manifest=None):
    m = manifest or a_manifest()
    fam = type("F", (), {"base_model_family": lineage})() if lineage else None
    return type("R", (), {"load": staticmethod(lambda n: m),
                          "family": staticmethod(lambda n: fam)})()


def write_lora(path, base_model_version):
    """Minimal valid safetensors carrying a kohya __metadata__ block."""
    header = {"__metadata__": {"ss_base_model_version": base_model_version,
                               "ss_output_name": path.stem,
                               "ss_num_train_images": "180",
                               "ss_tag_frequency": json.dumps(
                                   {"img": {"trig_a": 10, "1girl": 10}})},
              "lora_down.weight": {"dtype": "F16", "shape": [1, 1],
                                   "data_offsets": [0, 2]}}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00")


def a_loras_dir(tmp_path, lineage="anima", name="your_character_lora.safetensors"):
    d = tmp_path / "loras"
    d.mkdir(exist_ok=True)
    write_lora(d / name, lineage)
    return d


# -- step 1: the lineage check runs against real LoRA metadata --------------

def test_lineage_mismatch_fires_through_run_for_an_anima_lora_on_a_flux_checkpoint(
        tmp_path, monkeypatch):
    """The whole point of the wiring: until now this problem could only be produced
    by a unit test handing preflight.check a literal dict."""
    loras = a_loras_dir(tmp_path)
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    client = FakeClient()

    with pytest.raises(errors.ComfyrackError) as ei:
        runner.run(a_registry("flux"), client, "character-scene",
                   results_dir=tmp_path, loras_dir=loras)

    msg = str(ei.value)
    assert "lineage_mismatch" in msg
    assert "your_character_lora.safetensors" in msg   # which LoRA
    assert "illustrious" in msg                    # the LoRA's lineage
    assert "flux" in msg                           # the checkpoint's lineage
    assert client.submitted is None                # and nothing was queued


def test_an_anima_lora_on_an_illustrious_family_is_not_a_mismatch(tmp_path, monkeypatch):
    """A LoRA header records the CHECKPOINT it trained against ("anima"); a
    family.yaml records that checkpoint's LINEAGE ("illustrious"). Comparing the two
    raw strings would hard-fail every anima LoRA on this project's primary family."""
    loras = a_loras_dir(tmp_path)
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    client = FakeClient()

    res = runner.run(a_registry("illustrious"), client, "character-scene",
                     results_dir=tmp_path, loras_dir=loras)
    assert res.paths[0].read_bytes() == b"IMAGEBYTES"
    assert client.submitted is not None


def test_a_versioned_lineage_string_is_recognised_rather_than_reported_as_a_mismatch(
        tmp_path, monkeypatch):
    """Trainers write ss_base_model_version freely: "flux1-dev", "sdxl_base_v1-0",
    a Civitai baseModel string. A flux LoRA on a flux family is not a mismatch
    however the trainer spelled flux."""
    loras = a_loras_dir(tmp_path, lineage="flux1-dev")
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    client = FakeClient()
    runner.run(a_registry("flux"), client, "character-scene",
               results_dir=tmp_path, loras_dir=loras)
    assert client.submitted is not None


def test_a_versioned_string_from_a_different_lineage_still_fires(tmp_path, monkeypatch):
    """The other half of that tolerance: recognising the spelling must not stop it
    discriminating."""
    loras = a_loras_dir(tmp_path, lineage="sdxl_base_v1-0")
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())

    with pytest.raises(errors.ComfyrackError) as ei:
        runner.run(a_registry("flux"), FakeClient(), "character-scene",
                   results_dir=tmp_path, loras_dir=loras)
    assert "sdxl" in str(ei.value)


def test_an_unrecognised_lineage_string_is_unknown_not_different(tmp_path, monkeypatch):
    """Unknown must never mean mismatched. Hard-failing a valid run over a string
    this package has no mapping for is worse than the degraded render the check
    exists to prevent."""
    loras = a_loras_dir(tmp_path, lineage="some-private-trainer-build")
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    client = FakeClient()
    runner.run(a_registry("flux"), client, "character-scene",
               results_dir=tmp_path, loras_dir=loras)
    assert client.submitted is not None


def test_the_lineage_check_stays_dormant_when_no_loras_dir_is_configured(
        tmp_path, monkeypatch):
    """[paths] loras is optional. An unconfigured project must keep running exactly
    as it did before this wiring existed, not acquire a new hard failure."""
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    client = FakeClient()

    res = runner.run(a_registry("flux"), client, "character-scene",
                     results_dir=tmp_path, loras_dir=None)
    assert res.paths[0].read_bytes() == b"IMAGEBYTES"
    assert client.submitted is not None


def test_a_configured_but_missing_loras_dir_is_dormant_too(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    client = FakeClient()
    res = runner.run(a_registry("flux"), client, "character-scene",
                     results_dir=tmp_path, loras_dir=tmp_path / "nope")
    assert res.paths[0].read_bytes() == b"IMAGEBYTES"


def test_the_lora_directory_is_scanned_once_per_run_not_once_per_node(
        tmp_path, monkeypatch):
    """scan() opens every safetensors header under the directory. A 205-LoRA
    directory rescanned per node is 615 file reads for one three-node graph."""
    loras = a_loras_dir(tmp_path)
    write_lora(loras / "second.safetensors", "anima")
    write_lora(loras / "third.safetensors", "anima")
    three_loras = dict(LORA_GRAPH)
    three_loras["3"] = {"class_type": "LoraLoader", "_meta": {"title": "l2"},
                        "inputs": {"lora_name": "second.safetensors"}}
    three_loras["4"] = {"class_type": "LoraLoader", "_meta": {"title": "l3"},
                        "inputs": {"lora_name": "third.safetensors"}}

    calls = []
    real_scan = runner.lora_meta.scan
    monkeypatch.setattr(runner.lora_meta, "scan",
                        lambda d: (calls.append(d), real_scan(d))[1])
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(three_loras).copy())

    runner.run(a_registry("illustrious"), FakeClient(), "character-scene",
               results_dir=tmp_path, loras_dir=loras)
    assert len(calls) == 1


def test_rack_run_resolves_the_loras_dir_from_config(tmp_path, monkeypatch):
    """The only place that can resolve [paths] loras is the layer holding Config."""
    from comfyrack import Rack

    proj = tmp_path / "proj"
    (proj / ".comfyrack").mkdir(parents=True)
    (proj / "models" / "loras").mkdir(parents=True)
    (proj / ".comfyrack" / "config.toml").write_text(
        '[paths]\nloras = "models/loras"\n', encoding="utf-8")

    seen = {}

    def mock_run(registry, client, name, **kw):
        seen.update(kw)
        return runner.RunResult()

    monkeypatch.setattr("comfyrack.runner.run", mock_run)
    (tmp_path / "b").mkdir(exist_ok=True)
    (tmp_path / "u").mkdir(exist_ok=True)
    rack = Rack(start_dir=proj, builtin_dir=tmp_path / "b", user_dir=tmp_path / "u",
                user_config=tmp_path / "absent.toml")
    rack.run("w")
    assert seen["loras_dir"] == proj / "models" / "loras"


# -- step 2: progress reporting ---------------------------------------------

EVENTS = [
    {"type": "progress", "data": {"value": 10, "max": 20, "prompt_id": "p1"}},
    {"type": "progress", "data": {"value": 20, "max": 20, "prompt_id": "p1"}},
    {"type": "executing", "data": {"node": None, "prompt_id": "p1"}},
]


class ProgressClient(FakeClient):
    """A client whose websocket channel replays scripted events, like the real one
    does for a job that is actually rendering."""

    def __init__(self, events=None, **kw):
        super().__init__(**kw)
        self.events = events if events is not None else EVENTS
        self.progress_calls = 0

    def progress(self, prompt_id, on_event, ws_factory=None, client_id=None):
        self.progress_calls += 1
        for event in self.events:
            on_event(event)


def _stderr(text):
    print(text, file=sys.stderr)


def test_progress_events_are_written_to_stderr_and_stdout_stays_empty(
        tmp_path, monkeypatch, capsys):
    """stdout carries the machine-readable result and must stay parseable, which is
    the entire reason this cannot go there."""
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    client = ProgressClient()

    res = runner.run(a_registry("illustrious"), client, "character-scene",
                     results_dir=tmp_path, on_progress=_stderr)

    cap = capsys.readouterr()
    assert res.paths[0].read_bytes() == b"IMAGEBYTES"
    assert client.progress_calls == 1
    assert "10/20" in cap.err
    assert cap.out == ""


def test_stdout_is_byte_identical_with_and_without_a_progress_callback(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())

    runner.run(a_registry("illustrious"), ProgressClient(), "character-scene",
               results_dir=tmp_path / "loud", on_progress=_stderr)
    loud = capsys.readouterr()

    runner.run(a_registry("illustrious"), ProgressClient(), "character-scene",
               results_dir=tmp_path / "quiet")
    quiet = capsys.readouterr()

    assert loud.out == quiet.out == ""
    assert loud.err != ""
    assert quiet.err == ""


def test_without_a_callback_the_websocket_is_never_opened_at_all(
        tmp_path, monkeypatch, capsys):
    """Default silence is not "connect and discard": library and test callers must
    not pay for a websocket they are not reading."""
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    client = ProgressClient()
    runner.run(a_registry("illustrious"), client, "character-scene",
               results_dir=tmp_path)
    assert client.progress_calls == 0
    assert capsys.readouterr().err == ""


def test_progress_streams_while_wait_polls_rather_than_before_it(tmp_path, monkeypatch):
    """progress() and wait() BOTH block until the job finishes. Running them in
    sequence would double a five-minute render's wall clock, so this pins that they
    overlap: progress() here cannot emit until wait() has started polling."""
    polling = threading.Event()
    observed = {}

    class ConcurrentClient(FakeClient):
        def progress(self, prompt_id, on_event, ws_factory=None, client_id=None):
            observed["overlapped"] = polling.wait(timeout=5)
            on_event({"type": "executing", "data": {"node": None, "prompt_id": prompt_id}})

        def wait(self, prompt_id, timeout_s=1800):
            polling.set()
            return self.entry

    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    lines = []
    runner.run(a_registry("illustrious"), ConcurrentClient(), "character-scene",
               results_dir=tmp_path, on_progress=lines.append)

    assert observed.get("overlapped") is True
    assert lines  # and the events still arrived before submit() returned


def test_progress_output_cannot_bleed_into_the_following_run(tmp_path, monkeypatch):
    """submit() gives up on the progress thread once the job is done, so a socket
    that never saw its terminal frame leaves the thread parked in recv() with a
    live socket for up to its full 60s timeout. Capping the join at 2s only stops
    submit() WAITING for it -- the thread keeps going, and in a batch, shot N's
    "step 10/20" then prints to stderr in the middle of shot N+1. The join has to
    be paired with a stop signal the pump honours."""
    monkeypatch.setattr(runner, "PROGRESS_JOIN_TIMEOUT_S", 0.05)
    released, finished = threading.Event(), threading.Event()
    attempts, delivered = [], []

    class LateClient(FakeClient):
        def progress(self, prompt_id, on_event, ws_factory=None, client_id=None):
            try:
                released.wait(timeout=5)      # still blocked in recv() when submit() gives up
                for i in (1, 2, 3):
                    attempts.append(i)
                    on_event({"type": "progress", "data": {"value": i, "max": 3}})
            finally:
                finished.set()

    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    res = runner.run(a_registry("illustrious"), LateClient(), "character-scene",
                     results_dir=tmp_path, on_progress=delivered.append)

    assert res.paths[0].read_bytes() == b"IMAGEBYTES"
    assert delivered == []               # submit() returned without waiting on it
    released.set()                       # the abandoned socket finally yields frames
    assert finished.wait(timeout=5)
    assert attempts == [1]               # the pump unwound instead of draining the socket
    assert delivered == []               # and not one line escaped into the next run


def test_a_failing_progress_channel_does_not_fail_a_successful_render(
        tmp_path, monkeypatch):
    """websocket-client missing, a proxy refusing the upgrade, a dropped socket --
    none of it may destroy work the GPU already did."""
    class BrokenClient(FakeClient):
        def progress(self, prompt_id, on_event, ws_factory=None, client_id=None):
            raise RuntimeError("no websocket for you")

    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    res = runner.run(a_registry("illustrious"), BrokenClient(), "character-scene",
                     results_dir=tmp_path, on_progress=_stderr)
    assert res.paths[0].read_bytes() == b"IMAGEBYTES"


def test_a_client_without_a_progress_channel_still_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(LORA_GRAPH).copy())
    res = runner.run(a_registry("illustrious"), FakeClient(), "character-scene",
                     results_dir=tmp_path, on_progress=_stderr)
    assert res.paths[0].read_bytes() == b"IMAGEBYTES"


def test_progress_lines_name_the_step_and_the_finish():
    assert "10/20" in runner.format_progress(
        {"type": "progress", "data": {"value": 10, "max": 20}})
    assert runner.format_progress(
        {"type": "executing", "data": {"node": None}}) is not None
    assert runner.format_progress({"type": "status", "data": {}}) is None


def test_rack_run_threads_the_progress_callback_through(tmp_path, monkeypatch):
    from comfyrack import Rack

    proj = tmp_path / "proj"
    (proj / ".comfyrack").mkdir(parents=True)
    (proj / ".comfyrack" / "config.toml").write_text("[paths]\n", encoding="utf-8")

    seen = {}

    def mock_run(registry, client, name, **kw):
        seen.update(kw)
        return runner.RunResult()

    monkeypatch.setattr("comfyrack.runner.run", mock_run)
    (tmp_path / "b").mkdir(exist_ok=True)
    (tmp_path / "u").mkdir(exist_ok=True)
    rack = Rack(start_dir=proj, builtin_dir=tmp_path / "b", user_dir=tmp_path / "u",
                user_config=tmp_path / "absent.toml")
    rack.run("w", on_progress=_stderr)
    assert seen["on_progress"] is _stderr


# -- step 3: the embeddings endpoint ----------------------------------------

class EmbeddingsClient:
    """Serves /embeddings the way ComfyUI's server.py does: a flat list of names
    with the extension already stripped."""

    def __init__(self, names=None, oi=None, raises=None):
        self.names = names
        self.oi = oi or {}
        self.raises = raises
        self.oi_calls = 0

    def embeddings(self):
        if self.raises:
            raise self.raises
        return self.names

    def object_info(self, class_type=None):
        self.oi_calls += 1
        return self.oi


def test_embeddings_reads_the_real_endpoint_and_sorts_it():
    c = EmbeddingsClient(names=["easynegative", "bad_hands_5", "aidv1-neg"])
    assert discover.embeddings(c) == ["aidv1-neg", "bad_hands_5", "easynegative"]
    assert c.oi_calls == 0     # no multi-MB /object_info fetch for a list of names


def test_an_empty_endpoint_response_is_authoritative_not_a_fallback_trigger():
    """`[]` from a build that HAS the route means that machine has no embeddings.
    Re-asking /object_info there would cost megabytes to learn the same thing."""
    c = EmbeddingsClient(names=[], oi={"_embeddings": ["ghost"]})
    assert discover.embeddings(c) == []
    assert c.oi_calls == 0


def test_embeddings_falls_back_to_object_info_when_the_route_is_missing():
    """Older builds answer 404, which _request turns into a ComfyrackError."""
    c = EmbeddingsClient(raises=errors.ComfyrackError("ComfyUI returned HTTP 404"),
                         oi={"_embeddings": ["zzz", "aaa"]})
    assert discover.embeddings(c) == ["aaa", "zzz"]
    assert c.oi_calls == 1


def test_embeddings_falls_back_for_a_transport_without_the_method():
    class OldClient:
        def object_info(self, class_type=None):
            return {"_embeddings": ["bad_hands_5"]}

    assert discover.embeddings(OldClient()) == ["bad_hands_5"]


def test_embeddings_returns_empty_when_neither_source_has_anything():
    c = EmbeddingsClient(raises=errors.ComfyrackError("HTTP 404"), oi={})
    assert discover.embeddings(c) == []


class _Resp:
    status_code = 200

    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        if self._payload is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


class _Session:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        return self.resp


def test_client_embeddings_gets_the_documented_path():
    """/api/models/{folder} -- the route that actually serves JSON on a live build.
    The bare /embeddings path is a frontend HTML page there, not an API."""
    s = _Session(_Resp(["bad_hands_5.pt"]))
    c = ComfyClient("http://box:8188", session=s)
    assert c.embeddings() == ["bad_hands_5.pt"]
    assert s.calls == ["http://box:8188/api/models/embeddings"]


def test_an_html_body_becomes_a_comfyrack_error_not_a_raw_json_traceback():
    """Observed on a live ComfyUI 0.28.0: GET /embeddings answers 200 with a
    185KB "Embedding Models" HTML page. r.json() then raises JSONDecodeError from
    OUTSIDE _request, escaping this module's contract that no raw requests
    exception reaches a caller."""
    c = ComfyClient("http://box:8188",
                    session=_Session(_Resp(None, text="<!DOCTYPE html>")))
    with pytest.raises(errors.ComfyrackError) as ei:
        c.embeddings()
    assert "/api/models/embeddings" in ei.value.message
    assert ei.value.help_text


def test_a_build_that_serves_html_falls_back_instead_of_crashing():
    """The whole point of turning that into a typed error: discover can catch it."""
    class HtmlThenObjectInfo(ComfyClient):
        def object_info(self, class_type=None):
            return {"_embeddings": ["bad_hands_5"]}

    c = HtmlThenObjectInfo("http://box:8188",
                           session=_Session(_Resp(None, text="<!DOCTYPE html>")))
    assert discover.embeddings(c) == ["bad_hands_5"]


# -- subfoldered LoRAs reach the lineage check (final review, Important 6) ---
#
# The silent half of the subfolder bug. `lora_meta.scan()` rglob'd
# subdirectories but keyed results by the bare `path.name`, while a workflow's
# `lora_name` widget holds the reference the machine uses -- the subpath. The
# lineage check's `value in lora_meta` therefore missed, and the check simply
# DROPPED OUT for those 13-of-206 files: no mismatch reported, no error, no
# signal of any kind. Its loud twin (dispatch._missing calling the same file
# unroutable) is covered in test_dispatch.py; keeping both consistent is the
# point, because the loud/silent asymmetry is what made the bug survive.

SUBFOLDER_REF = "Anima\\AnimaEditV1.safetensors"

SUBFOLDER_OBJECT_INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {
        "ckpt_name": [["krea2.safetensors"], {}]}}},
    "LoraLoader": {"input": {"required": {"lora_name": [[SUBFOLDER_REF], {}]}}},
    "SaveImage": {"input": {"required": {"filename_prefix": ["STRING", {}]}}},
}

SUBFOLDER_GRAPH = {
    "1": {"class_type": "CheckpointLoaderSimple", "_meta": {"title": "ckpt"},
          "inputs": {"ckpt_name": "krea2.safetensors"}},
    "2": {"class_type": "LoraLoader", "_meta": {"title": "lora"},
          "inputs": {"lora_name": SUBFOLDER_REF, "model": ["1", 0]}},
    "9": {"class_type": "SaveImage", "_meta": {"title": "Save Image"},
          "inputs": {"filename_prefix": "out"}},
}


class SubfolderClient(FakeClient):
    def object_info(self, class_type=None):
        return SUBFOLDER_OBJECT_INFO


def a_subfoldered_loras_dir(tmp_path, lineage="anima"):
    d = tmp_path / "loras" / "Anima"
    d.mkdir(parents=True, exist_ok=True)
    write_lora(d / "AnimaEditV1.safetensors", lineage)
    return tmp_path / "loras"


def test_lora_lineages_keys_a_subfoldered_lora_by_its_relative_reference(tmp_path):
    loras = a_subfoldered_loras_dir(tmp_path)
    assert runner.lora_lineages(loras) == {
        "Anima/AnimaEditV1.safetensors": {"base_model_family": "illustrious"}}


def test_lineage_mismatch_fires_through_run_for_a_subfoldered_lora(
        tmp_path, monkeypatch):
    """The required proof: the check FIRES on a real mismatch involving a
    subfoldered LoRA instead of quietly skipping it. The graph spells the
    reference the way a Windows ComfyUI reports it; the scan spells it from
    disk. Those must resolve to the same file."""
    loras = a_subfoldered_loras_dir(tmp_path)
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(SUBFOLDER_GRAPH).copy())
    client = SubfolderClient()

    with pytest.raises(errors.ComfyrackError) as ei:
        runner.run(a_registry("flux"), client, "character-scene",
                   results_dir=tmp_path, loras_dir=loras)

    msg = str(ei.value)
    assert "lineage_mismatch" in msg
    assert "AnimaEditV1.safetensors" in msg    # which LoRA
    assert "illustrious" in msg                # the LoRA's lineage
    assert "flux" in msg                       # the checkpoint's lineage
    assert client.submitted is None            # and nothing was queued


def test_a_matching_subfoldered_lora_is_not_a_mismatch(tmp_path, monkeypatch):
    """The positive control: normalising must not make every subfoldered LoRA
    fire. An anima-trained LoRA on an illustrious family still renders."""
    loras = a_subfoldered_loras_dir(tmp_path)
    monkeypatch.setattr(runner, "load_graph", lambda m, c: Graph(SUBFOLDER_GRAPH).copy())
    client = SubfolderClient()

    res = runner.run(a_registry("illustrious"), client, "character-scene",
                     results_dir=tmp_path, loras_dir=loras)

    assert res.paths[0].read_bytes() == b"IMAGEBYTES"
    assert client.submitted is not None
