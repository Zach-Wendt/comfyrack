import json
from comfyrack.http import ComfyClient


class FakeWS:
    """Replays a scripted frame sequence. Binary frames are included deliberately:
    the contract is that progress() ignores them completely."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False

    def recv(self):
        if not self.frames:
            raise ConnectionError("closed")
        return self.frames.pop(0)

    def close(self):
        self.closed = True


def test_progress_yields_text_events_and_stops_on_execution_success():
    frames = [
        json.dumps({"type": "progress", "data": {"value": 1, "max": 20}}),
        json.dumps({"type": "progress", "data": {"value": 20, "max": 20}}),
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "p1"}}),
    ]
    seen = []
    c = ComfyClient("http://box:8188", session=object())
    c.progress("p1", on_event=seen.append, ws_factory=lambda url: FakeWS(frames))
    assert [e["type"] for e in seen] == ["progress", "progress", "executing"]


def test_progress_discards_binary_frames_entirely():
    frames = [
        b"\x00\x01\x02 binary preview payload",
        json.dumps({"type": "progress", "data": {"value": 5, "max": 20}}),
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "p1"}}),
    ]
    seen = []
    c = ComfyClient("http://box:8188", session=object())
    c.progress("p1", on_event=seen.append, ws_factory=lambda url: FakeWS(frames))
    assert all(isinstance(e, dict) for e in seen)
    assert [e["type"] for e in seen] == ["progress", "executing"]


def test_progress_ignores_events_for_a_different_prompt():
    frames = [
        json.dumps({"type": "progress", "data": {"prompt_id": "other", "value": 1}}),
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "p1"}}),
    ]
    seen = []
    c = ComfyClient("http://box:8188", session=object())
    c.progress("p1", on_event=seen.append, ws_factory=lambda url: FakeWS(frames))
    assert [e["type"] for e in seen] == ["executing"]


def test_progress_closes_the_socket_even_when_the_connection_drops():
    ws = FakeWS([])
    c = ComfyClient("http://box:8188", session=object())
    c.progress("p1", on_event=lambda e: None, ws_factory=lambda url: ws)
    assert ws.closed


def test_progress_survives_a_library_specific_close_exception():
    """websocket-client raises WebSocketConnectionClosedException, which inherits from
    Exception rather than OSError. A guard catching only OSError would let it escape."""
    class ClosedWS:
        def __init__(self):
            self.closed = False

        def recv(self):
            raise RuntimeError("Connection to remote host was lost.")

        def close(self):
            self.closed = True

    ws = ClosedWS()
    c = ComfyClient("http://box:8188", session=object())
    c.progress("p1", on_event=lambda e: None, ws_factory=lambda url: ws)
    assert ws.closed


def test_progress_connects_to_the_unprefixed_ws_route_with_our_own_client_id():
    """/ws is the one route that is NOT /api-prefixed, and the clientId query
    parameter is how ComfyUI scopes events to this client -- so the default must
    be the id this client generated, not a shared constant."""
    seen = []
    c = ComfyClient("http://box:8188", session=object())
    c.progress("p1", on_event=lambda e: None,
               ws_factory=lambda url: seen.append(url) or FakeWS([]))
    assert seen == [f"ws://box:8188/ws?clientId={c.client_id}"]


def test_progress_honours_an_explicit_client_id():
    seen = []
    c = ComfyClient("http://box:8188", session=object())
    c.progress("p1", on_event=lambda e: None, client_id="other-client",
               ws_factory=lambda url: seen.append(url) or FakeWS([]))
    assert seen == ["ws://box:8188/ws?clientId=other-client"]


def test_progress_upgrades_an_https_url_to_wss():
    seen = []
    c = ComfyClient("https://box:8188", session=object())
    c.progress("p1", on_event=lambda e: None,
               ws_factory=lambda url: seen.append(url) or FakeWS([]))
    assert seen == [f"wss://box:8188/ws?clientId={c.client_id}"]
