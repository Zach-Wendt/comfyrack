"""Live tests. Excluded by default; they need a running ComfyUI.

    python -m pytest tests/integration -m integration -v

Set COMFYRACK_TEST_URL to point at a specific instance."""
import os
import pytest

from comfyrack import Rack
from comfyrack.http import ComfyClient

pytestmark = pytest.mark.integration

URL = os.environ.get("COMFYRACK_TEST_URL", "http://127.0.0.1:8188")


@pytest.fixture(scope="module")
def client():
    c = ComfyClient(URL)
    try:
        c.system_stats()
    except Exception as exc:
        pytest.skip(f"no ComfyUI at {URL}: {exc}")
    return c


def test_object_info_returns_a_populated_node_catalog(client):
    oi = client.object_info()
    assert "KSampler" in oi
    assert len(oi) > 50


def test_scoped_object_info_returns_just_that_class(client):
    assert list(client.object_info("KSampler")) == ["KSampler"]


def test_system_stats_reports_a_device(client):
    from comfyrack.discover import stats_summary
    s = stats_summary(client)
    assert s["device"]
    assert s["vram_total_gb"] > 0


def test_queue_status_is_readable(client):
    from comfyrack.discover import queue_summary
    q = queue_summary(client)
    assert set(q) == {"running", "pending"}


def test_models_lists_at_least_one_checkpoint(client):
    from comfyrack.discover import models
    assert models(client).get("checkpoints")


def test_the_bare_embeddings_path_is_not_a_json_api_on_this_build(client):
    """The mistake this pair of tests exists to keep from recurring.

    server.py defines `@routes.get("/embeddings")` returning JSON, so reading the
    source says it is the endpoint. On a real build (0.28.0) the frontend shadows
    it and answers 200 with an HTML page, and .json() raised a bare
    JSONDecodeError. A route decorator is not evidence about a running server.

    This is written to record the shadowing where it exists and pass anyway where
    it does not, so it never becomes a test that has to be deleted on a build that
    reverts the shadowing.
    """
    import requests

    r = requests.get(f"{URL}/embeddings", timeout=30)
    ctype = r.headers.get("Content-Type", "")
    print(f"GET /embeddings -> {r.status_code} {ctype} {len(r.content)} bytes")
    if "json" not in ctype:
        assert "html" in ctype, f"unexpected non-JSON content type {ctype!r}"
        # ...and the supported route must still work on that same build.
        assert isinstance(client.embeddings(), list)


def test_embeddings_reads_the_models_folder_route_on_this_build(client):
    # Plan 1 Task 18 found `_embeddings` absent from /object_info on a real
    # ComfyUI, and 0.28.0 confirms it, so discover.embeddings reads
    # GET /models/embeddings -- the `/models/{folder}` route that genuinely
    # serves JSON -- and keeps /object_info only as a fallback. This correlates
    # the result against whichever source this particular server actually
    # serves, so it fails if either path or the sort regresses, and prints what
    # the build had so the assumption stays checkable rather than assumed.
    from comfyrack.discover import embeddings
    from comfyrack.errors import ComfyrackError

    try:
        direct = client.embeddings()
    except ComfyrackError as exc:
        direct = None
        print(f"GET /models/embeddings unavailable on this build: {exc}")
    oi_key = client.object_info().get("_embeddings")
    result = embeddings(client)
    print(f"/models/embeddings: {direct}; /object_info _embeddings: {oi_key}; "
          f"returned {len(result)} embedding(s): {result}")

    if isinstance(direct, list):
        assert result == sorted(str(n) for n in direct)
    elif isinstance(oi_key, list):
        assert result == sorted(str(n) for n in oi_key)
    else:
        assert result == []


def test_upload_image_round_trips(client, tmp_path):
    # 1x1 PNG
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000a49444154789c6360000002000100ffff0300000600"
        "0557bfabd40000000049454e44ae426082")
    p = tmp_path / "probe.png"
    p.write_bytes(png)
    assert client.upload_image(p)


def test_preflight_passes_for_a_registered_workflow_on_this_machine(client):
    # `client` is only a skip gate here (to satisfy the README's "skips
    # rather than fails" contract) -- Rack() resolves its own machine URL
    # from config, which may differ from COMFYRACK_TEST_URL. There is no
    # supported way to force a raw URL into Rack, so we don't try.
    rack = Rack()
    manifests = rack.list()
    if not manifests:
        pytest.skip("no workflows registered")
    target = next((m for m in manifests if m.family == "anima"), manifests[0])
    problems = rack.preflight(target.name)
    for p in problems:
        print(f"{p.kind} node {p.node_id}: {p.detail}")
    assert all(p.kind != "missing_node_class" for p in problems)
