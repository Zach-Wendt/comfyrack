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


# The ["TAG", {"options": [...]}] input shapes combo_choices() knows are combos.
# A string tag outside this set makes combo_choices() return None, and preflight
# then silently stops validating that input -- exactly what happened when V3
# nodes introduced COMBO and COMFY_DYNAMICCOMBO_V3.
KNOWN_COMBO_TAGS = {"COMBO", "COMFY_DYNAMICCOMBO_V3"}

# Tags that legitimately carry an "options" dict WITHOUT being a combo widget.
# Populated from a live run; each entry needs a one-line reason. A tag listed
# here is a known non-combo, not a silently-unchecked input.
NOT_COMBOS = {
    # comfyui-easy-use multi-select widget; its frontend fills the real options at
    # runtime (/object_info carries one placeholder), so there is nothing to check.
    "EASY_COMBO",
}


def test_every_option_bearing_type_is_known(client):
    """Fail loudly when ComfyUI invents a new option-bearing input tag.

    A combo input reaches comfyrack either as a bare list of choices (old
    shape, fine here) or as ["TAG", {"options": [...]}]. combo_choices()
    understands only the tags in KNOWN_COMBO_TAGS; any other string tag
    carrying "options" reads as "not a combo", so preflight silently stops
    checking that input. This walks every required and optional input spec
    on the live machine and fails ONCE with the full sorted list."""
    oi = client.object_info()
    offenders = set()
    for class_type, entry in oi.items():
        inputs = (entry or {}).get("input") or {}
        for section in ("required", "optional"):
            for name, val in (inputs.get(section) or {}).items():
                if not isinstance(val, list) or len(val) < 2:
                    continue
                tag, opts = val[0], val[1]
                if (isinstance(tag, str) and isinstance(opts, dict)
                        and "options" in opts
                        and tag not in KNOWN_COMBO_TAGS
                        and tag not in NOT_COMBOS):
                    offenders.add(f"{class_type}.{name}: {tag}")
    if offenders:
        lines = sorted(offenders)
        shown = "\n".join(lines[:50])
        if len(lines) > 50:
            shown += f"\n... and {len(lines) - 50} more"
        pytest.fail(
            "unknown option-bearing input tag(s): combo_choices() returns None "
            "for these, so preflight silently stops checking them:\n" + shown)


def test_registry_preflights_without_crashing(client):
    """Every builtin workflow must survive preflight against the live machine.

    Problems are EXPECTED -- the corpus targets many model families and no
    single machine has them all installed -- so counts are printed, not
    asserted. What must never happen is check() raising: that would mean the
    live object_info has a shape preflight cannot walk, which is the same
    drift signal test 1 catches at the tag level."""
    import json
    from collections import Counter

    from comfyrack.graph import Graph
    from comfyrack.preflight import check
    from comfyrack.registry import BUILTIN_DIR

    oi = client.object_info()
    counts = Counter()
    checked = 0
    for path in sorted(BUILTIN_DIR.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("nodes"), list):
            continue  # UI-format workflow, not API-format
        for problem in check(Graph(data), oi):
            counts[problem.kind] += 1
        checked += 1
    print(f"preflight.check ran on {checked} builtin workflow(s) without raising")
    for kind, n in sorted(counts.items()):
        print(f"  {kind}: {n}")
