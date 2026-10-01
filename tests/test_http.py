import uuid
import pytest
import requests
from comfyrack.http import ComfyClient
from comfyrack import errors


class FakeResponse:
    def __init__(self, json_data=None, content=b"", status=200, text=""):
        self._json = json_data
        self.content = content
        self.status_code = status
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            # A real requests.HTTPError carries the response, which is what
            # _request's return_error_response opt-in hands back to the caller.
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Records calls and replays scripted responses, so no test touches a network."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def _pop(self, key):
        seq = self.responses[key]
        return seq.pop(0) if isinstance(seq, list) else seq

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        for key in self.responses:
            if key in url:
                return self._pop(key)
        raise AssertionError(f"unscripted GET {url}")

    def post(self, url, json=None, files=None, data=None, timeout=None):
        self.calls.append(("POST", url, json or data))
        for key in self.responses:
            if key in url:
                return self._pop(key)
        raise AssertionError(f"unscripted POST {url}")


class DeadSession:
    """Every call fails the way a dead port really fails."""

    def __init__(self, exc=None):
        self.exc = exc or requests.ConnectionError("connection refused")

    def get(self, url, **kw):
        raise self.exc

    def post(self, url, **kw):
        raise self.exc


def test_a_connection_failure_becomes_a_comfyrack_error_naming_the_url():
    c = ComfyClient("http://box:8188", session=DeadSession())
    with pytest.raises(errors.ComfyrackError) as ei:
        c.system_stats()
    assert "http://box:8188/api/system_stats" in ei.value.message
    assert "ConnectionError" in ei.value.message
    assert ei.value.help_text and "comfyrack stats" in ei.value.help_text


@pytest.mark.parametrize("call", [
    lambda c: c.queue_prompt({"1": {}}),
    lambda c: c.wait("abc-123", timeout_s=5),
    lambda c: c.view("a.png"),
    lambda c: c.object_info(),
    lambda c: c.queue_status(),
    lambda c: c.system_stats(),
    lambda c: c.interrupt(),
])
def test_every_transport_method_wraps_connection_failures(call):
    """One unwrapped call site is enough to put a traceback on stderr, so pin all
    of them rather than a representative sample."""
    c = ComfyClient("http://box:8188", session=DeadSession(), poll_interval=0)
    with pytest.raises(errors.ComfyrackError):
        call(c)


def test_upload_image_wraps_connection_failures(tmp_path):
    src = tmp_path / "pic.png"
    src.write_bytes(b"PNGBYTES")
    c = ComfyClient("http://box:8188", session=DeadSession())
    with pytest.raises(errors.ComfyrackError) as ei:
        c.upload_image(src)
    assert "/upload/image" in ei.value.message


def test_an_http_status_error_reports_the_status_and_stays_in_the_hierarchy():
    """A 500 from a live-but-unhappy ComfyUI is a different diagnosis than an
    unreachable box, so it must not collapse into the same message."""
    exc = requests.HTTPError("boom")
    exc.response = type("R", (), {"status_code": 500})()
    c = ComfyClient("http://box:8188", session=DeadSession(exc))
    with pytest.raises(errors.ComfyrackError) as ei:
        c.queue_status()
    assert "HTTP 500" in ei.value.message
    assert "cannot reach" not in ei.value.message


def test_wrapping_does_not_swallow_the_typed_errors_raised_above_the_transport():
    """The transport wrapper must not become the vocabulary for everything: an
    execution error reported inside a finished job keeps its own message."""
    s = FakeSession({"/api/jobs": FakeResponse({
        "status": "failed",
        "outputs": {},
        "execution_status": {"messages": [["execution_error", {"exception_message": "OOM"}]]},
    })})
    c = ComfyClient("http://box:8188", session=s, poll_interval=0)
    with pytest.raises(errors.ComfyrackError) as ei:
        c.wait("abc-123", timeout_s=5)
    assert "OOM" in ei.value.message
    assert "cannot reach" not in ei.value.message


def test_the_client_generates_a_client_id_and_sends_it_with_every_prompt():
    """ComfyUI scopes events and history by client_id; sending our own (instead
    of a shared constant) is what keeps one Rack's job from another's stream."""
    s = FakeSession({"/api/prompt": FakeResponse({"prompt_id": "server-echo"})})
    c = ComfyClient("http://box:8188", session=s)
    assert str(uuid.UUID(c.client_id)) == c.client_id  # canonical lowercase uuid
    c.queue_prompt({"1": {"class_type": "KSampler"}})
    body = s.calls[0][2]
    assert body["client_id"] == c.client_id


def test_queue_prompt_posts_our_own_prompt_id_and_returns_what_it_sent():
    """The caller polls the id it generated, so the return value must be the id
    that went out on the wire -- not whatever the server echoed back."""
    s = FakeSession({"/api/prompt": FakeResponse({"prompt_id": "server-echo"})})
    c = ComfyClient("http://box:8188", session=s)
    returned = c.queue_prompt({"1": {"class_type": "KSampler"}})
    method, url, body = s.calls[0]
    assert url == "http://box:8188/api/prompt"
    assert body["prompt"] == {"1": {"class_type": "KSampler"}}
    assert body["prompt_id"] == returned
    assert returned != "server-echo"


def test_queue_prompt_generates_a_fresh_uuid_prompt_id_per_call():
    s = FakeSession({"/api/prompt": FakeResponse({"prompt_id": "x"})})
    c = ComfyClient("http://box:8188", session=s)
    first, second = c.queue_prompt({}), c.queue_prompt({})
    assert first != second
    assert str(uuid.UUID(first)) == first  # canonical lowercase uuid


def test_queue_prompt_honours_explicit_client_and_prompt_ids():
    s = FakeSession({"/api/prompt": FakeResponse({"prompt_id": "x"})})
    c = ComfyClient("http://box:8188", session=s)
    returned = c.queue_prompt({"1": {}}, client_id="cli-run", prompt_id="pid-9")
    assert returned == "pid-9"
    assert s.calls[0][2] == {"prompt": {"1": {}}, "client_id": "cli-run",
                             "prompt_id": "pid-9"}


def test_queue_prompt_400_raises_with_the_server_message_and_each_node_error():
    """A 400 body is the only place ComfyUI names the offending node; flattening
    it into the message is the difference between "fix node 3" and "retry"."""
    body = {
        "error": {"message": "prompt validation failed", "type": "prompt_validation"},
        "node_errors": {
            "3": {"errors": [{"message": "seed must be an integer", "details": "got 'lots'"}],
                  "class_type": "KSampler"},
            "9": {"errors": [{"message": "unknown input 'seedx'", "details": ""}],
                  "class_type": "SaveImage"},
        },
    }
    s = FakeSession({"/api/prompt": FakeResponse(body, status=400)})
    c = ComfyClient("http://box:8188", session=s)
    with pytest.raises(errors.ComfyrackError) as ei:
        c.queue_prompt({"1": {}})
    msg = ei.value.message
    assert "prompt validation failed" in msg
    assert "node 3 (KSampler): seed must be an integer got 'lots'" in msg
    assert "node 9 (SaveImage): unknown input 'seedx'" in msg


def test_queue_prompt_400_without_a_json_body_still_raises_a_comfyrack_error():
    """The opt-in returns the response, but a body that is not JSON must not
    turn into a raw JSONDecodeError escaping queue_prompt."""
    s = FakeSession({"/api/prompt": FakeResponse(None, status=400,
                                                  text="<html>bad request</html>")})
    c = ComfyClient("http://box:8188", session=s)
    with pytest.raises(errors.ComfyrackError) as ei:
        c.queue_prompt({"1": {}})
    assert "rejected" in ei.value.message


def test_wait_returns_the_history_shaped_entry_from_the_jobs_route():
    """The entry wait() builds must be exactly the /history shape the rest of
    the code reads: outputs plus the status dict with its messages."""
    s = FakeSession({"/api/jobs": FakeResponse({
        "status": "completed",
        "outputs": {"9": {"images": [{"filename": "a.png"}]}},
        "execution_status": {"messages": []},
    })})
    c = ComfyClient("http://box:8188", session=s, poll_interval=0)
    entry = c.wait("abc-123", timeout_s=5)
    assert entry["outputs"]["9"]["images"][0]["filename"] == "a.png"
    assert entry["status"] == {"messages": []}
    assert s.calls[0][1] == "http://box:8188/api/jobs/abc-123"


def test_wait_polls_until_the_job_reaches_a_terminal_status():
    s = FakeSession({"/api/jobs": [
        FakeResponse({"status": "pending"}),
        FakeResponse({"status": "in_progress", "outputs": {}}),
        FakeResponse({"status": "completed",
                      "outputs": {"9": {"images": [{"filename": "a.png"}]}},
                      "execution_status": {"messages": []}}),
    ]})
    c = ComfyClient("http://box:8188", session=s, poll_interval=0)
    entry = c.wait("abc-123", timeout_s=5)
    assert entry["outputs"]["9"]["images"][0]["filename"] == "a.png"
    assert len(s.calls) == 3


def test_wait_raises_the_execution_error_when_a_job_failed():
    s = FakeSession({"/api/jobs": FakeResponse({
        "status": "failed",
        "outputs": {},
        "execution_status": {"messages": [["execution_error", {"exception_message": "OOM"}]]},
    })})
    c = ComfyClient("http://box:8188", session=s, poll_interval=0)
    with pytest.raises(errors.ComfyrackError) as ei:
        c.wait("abc-123", timeout_s=5)
    assert "OOM" in str(ei.value)


def test_wait_says_the_job_failed_when_there_is_no_execution_error():
    """A failed job with no execution_error message still has to fail the run --
    silently returning empty outputs would look like a successful render."""
    s = FakeSession({"/api/jobs": FakeResponse({
        "status": "failed", "outputs": {}, "execution_status": {"messages": []}})})
    c = ComfyClient("http://box:8188", session=s, poll_interval=0)
    with pytest.raises(errors.ComfyrackError) as ei:
        c.wait("abc-123", timeout_s=5)
    assert "failed" in str(ei.value)


def test_wait_says_the_job_was_cancelled():
    s = FakeSession({"/api/jobs": FakeResponse({
        "status": "cancelled", "outputs": {}, "execution_status": {"messages": []}})})
    c = ComfyClient("http://box:8188", session=s, poll_interval=0)
    with pytest.raises(errors.ComfyrackError) as ei:
        c.wait("abc-123", timeout_s=5)
    assert "cancelled" in str(ei.value)


def test_wait_falls_back_to_history_when_the_jobs_route_404s():
    """Older ComfyUI builds predate /api/jobs; the fallback keeps wait() working
    on them without a version probe."""
    s = FakeSession({
        "/api/jobs": FakeResponse({"detail": "Not Found"}, status=404),
        "/api/history": FakeResponse({"abc-123": {
            "outputs": {"9": {"images": [{"filename": "a.png"}]}},
            "status": {"messages": []}}}),
    })
    c = ComfyClient("http://box:8188", session=s, poll_interval=0)
    entry = c.wait("abc-123", timeout_s=5)
    assert entry["outputs"]["9"]["images"][0]["filename"] == "a.png"
    # The fallback is sticky: later polls go straight to history, not via a 404.
    assert [u for _m, u, _p in s.calls] == [
        "http://box:8188/api/jobs/abc-123",
        "http://box:8188/api/history/abc-123",
    ]


def test_wait_times_out_with_an_actionable_message():
    s = FakeSession({"/api/jobs": FakeResponse({"status": "pending"})})
    c = ComfyClient("http://box:8188", session=s, poll_interval=0)
    with pytest.raises(errors.ComfyrackError) as ei:
        c.wait("abc-123", timeout_s=0)
    assert "comfyrack queue" in (ei.value.help_text or "")
    # A zero timeout must still poll ONCE before giving up. A deadline pre-check
    # would make zero requests and raise an identical error -- this assertion is
    # what distinguishes the two.
    assert len(s.calls) == 1


def test_wait_a_non_404_http_error_from_the_jobs_route_raises_the_standard_error():
    """A 500 is not a missing route: it must surface as the usual HTTP-error
    ComfyrackError -- not poll forever on an error body, and not let a non-JSON
    error page raise a raw JSONDecodeError out of wait()."""
    s = FakeSession({"/api/jobs": FakeResponse(None, status=500,
                                                  text="<html>upstream exploded</html>")})
    c = ComfyClient("http://box:8188", session=s, poll_interval=0)
    with pytest.raises(errors.ComfyrackError) as ei:
        c.wait("abc-123", timeout_s=5)
    assert "HTTP 500" in ei.value.message
    assert "http://box:8188/api/jobs/abc-123" in ei.value.message
    assert len(s.calls) == 1  # it did not retry the error as a pending job


def test_history_returns_the_entry_for_a_prompt_id():
    s = FakeSession({"/api/history": FakeResponse({"abc-123": {
        "outputs": {"9": {"images": [{"filename": "a.png"}]}},
        "status": {"messages": []}}})})
    c = ComfyClient("http://box:8188", session=s)
    entry = c.history("abc-123")
    assert entry["outputs"]["9"]["images"][0]["filename"] == "a.png"
    assert s.calls[0][1] == "http://box:8188/api/history/abc-123"


def test_history_raises_not_found_naming_the_id_when_the_server_has_no_record():
    """An unknown id comes back as an empty dict, not a 404. Returning it bare
    would read like a prompt that produced no outputs, so the id is named in
    the error instead."""
    s = FakeSession({"/api/history": FakeResponse({})})
    c = ComfyClient("http://box:8188", session=s)
    with pytest.raises(errors.NotFoundError) as ei:
        c.history("abc-123")
    assert "abc-123" in ei.value.message


def test_outputs_accepts_images_gifs_audio_and_text():
    c = ComfyClient("http://box:8188", session=FakeSession({}))
    assert c.outputs({"outputs": {"9": {"images": [{"filename": "a.png"}]}}}, "9") == \
        [{"filename": "a.png"}]
    assert c.outputs({"outputs": {"9": {"gifs": [{"filename": "a.mp4"}]}}}, "9") == \
        [{"filename": "a.mp4"}]
    assert c.outputs({"outputs": {"9": {"string": ["verdict"]}}}, "9") == ["verdict"]


def test_outputs_raises_naming_the_keys_it_actually_found():
    c = ComfyClient("http://box:8188", session=FakeSession({}))
    with pytest.raises(errors.ComfyrackError) as ei:
        c.outputs({"outputs": {"9": {"latents": []}}}, "9")
    assert "latents" in str(ei.value)


def test_upload_image_returns_subfolder_qualified_name():
    s = FakeSession({"/api/upload/image": FakeResponse({"name": "ref.png", "subfolder": "clipspace"})})
    c = ComfyClient("http://box:8188", session=s)
    import tempfile, pathlib
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"x")
        tmp = f.name
    assert c.upload_image(tmp) == "clipspace/ref.png"
    pathlib.Path(tmp).unlink()


def test_object_info_for_one_class_hits_the_scoped_endpoint():
    s = FakeSession({"/api/object_info/KSampler": FakeResponse({"KSampler": {"input": {}}})})
    c = ComfyClient("http://box:8188", session=s)
    assert "KSampler" in c.object_info("KSampler")


# -- /object_info is fetched once (final review, Important 6) ----------------

def _oi_session(n=1):
    return FakeSession({"/api/object_info": [FakeResponse({"KSampler": {"input": {}}})] * n})


def test_repeated_object_info_calls_hit_the_transport_once():
    """On a real ComfyUI with a normal custom-node load-out this response is
    multiple MB, and one `comfyrack run` fetched it twice: load_graph, then
    preflight_graph."""
    s = _oi_session()
    c = ComfyClient("http://box:8188", session=s)
    first, second, third = c.object_info(), c.object_info(), c.object_info()
    assert first == second == third
    assert [u for _m, u, _p in s.calls] == ["http://box:8188/api/object_info"]


def test_a_scoped_call_is_served_from_the_warm_full_cache():
    s = _oi_session()
    c = ComfyClient("http://box:8188", session=s)
    c.object_info()
    # No "/object_info/KSampler" is scripted, so a second request would raise
    # AssertionError("unscripted GET") rather than quietly pass.
    assert c.object_info("KSampler") == {"KSampler": {"input": {}}}
    assert len(s.calls) == 1


def test_refresh_object_info_forces_the_next_call_to_refetch():
    """A cache with no invalidation is a bug waiting for someone to install a
    custom node."""
    s = FakeSession({"/api/object_info": [FakeResponse({"A": {}}), FakeResponse({"B": {}})]})
    c = ComfyClient("http://box:8188", session=s)
    assert c.object_info() == {"A": {}}
    c.refresh_object_info()
    assert c.object_info() == {"B": {}}
    assert len(s.calls) == 2


# -- every route is the /api-prefixed form (spec: maintained routes) ----------


def test_every_route_uses_the_api_prefix():
    """ComfyUI registers every core route bare AND /api-prefixed; the /api form
    is the documented one (the bare /embeddings is even shadowed by the
    frontend), so pin the prefix across the whole client in one pass."""
    import tempfile, pathlib
    s = FakeSession({
        "/api/view": FakeResponse(content=b"IMGBYTES"),
        "/api/upload/image": FakeResponse({"name": "ref.png"}),
        "/api/object_info": FakeResponse({"KSampler": {"input": {}}}),
        "/api/models/embeddings": FakeResponse(["e.safetensors"]),
        "/api/queue": FakeResponse({"queue_running": [], "queue_pending": []}),
        "/api/system_stats": FakeResponse({"system": {}, "devices": []}),
        "/api/interrupt": FakeResponse({}),
    })
    c = ComfyClient("http://box:8188", session=s)
    assert c.view("a.png") == b"IMGBYTES"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"x")
        tmp = f.name
    assert c.upload_image(tmp) == "ref.png"
    pathlib.Path(tmp).unlink()
    assert "KSampler" in c.object_info()
    assert c.embeddings() == ["e.safetensors"]
    assert c.queue_status() == {"queue_running": [], "queue_pending": []}
    assert c.system_stats() == {"system": {}, "devices": []}
    c.interrupt()
    assert all(u.startswith("http://box:8188/api/") for _m, u, _p in s.calls)
