import pathlib
import pytest
import yaml
from comfyrack.config import Config
from comfyrack.registry import Registry

ROOT = pathlib.Path(__file__).resolve().parent.parent
REG = ROOT / "registry"
KNOWN_DIALECTS = {"booru", "natural"}


def builtin_registry(tmp_path):
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    return Registry(cfg, builtin_dir=REG, user_dir=tmp_path / "nouser")


def test_the_test_family_was_not_imported():
    assert not (REG / "_test").exists()


# -- BUILTIN_DIR resolves in both layouts -----------------------------------
#
# The default Registry (the one the CLI builds) reads registry.BUILTIN_DIR. If that
# constant points somewhere that does not exist, _iter_paths skips it silently and
# every listing command reports a clean, definitive, wrong zero.

def test_builtin_dir_points_at_a_real_directory_holding_manifests():
    from comfyrack.registry import BUILTIN_DIR
    assert BUILTIN_DIR.is_dir(), f"BUILTIN_DIR does not exist: {BUILTIN_DIR}"
    found = list(BUILTIN_DIR.glob("*/*.manifest.yaml"))
    assert found, f"BUILTIN_DIR holds no manifests: {BUILTIN_DIR}"


def test_a_default_registry_discovers_the_builtin_corpus(tmp_path):
    """Not just that the path exists -- that the layer the CLI actually reads loads
    the corpus. builtin_dir is deliberately NOT overridden here."""
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    reg = Registry(cfg, user_dir=tmp_path / "nouser")
    names = {m.name for m in reg.discover()}
    assert len(names) == sum(EXPECTED_FAMILY_COUNTS.values())
    assert "harmonize" in names


def test_builtin_dir_prefers_the_package_adjacent_copy_when_it_exists():
    """The installed-wheel layout. The src checkout has no src/comfyrack/registry,
    so exercise the preference rule against the same two candidates the module
    computes rather than asserting on whichever one this checkout happens to hit."""
    from comfyrack import registry as reg_mod
    pkg, repo = reg_mod._PKG_REGISTRY, reg_mod._REPO_REGISTRY
    assert pkg != repo
    assert pkg.name == "registry" and pkg.parent.name == "comfyrack"
    expected = pkg if pkg.is_dir() else repo
    assert reg_mod.BUILTIN_DIR == expected


# Deliberate tripwire over committed data, not a range: the worst outcome for this
# corpus is a manifest silently vanishing (a bad future edit to the import script, an
# accidentally-excluded family, a bad merge) while the suite stays green. Pinning the
# exact total AND the exact per-family split means a dropped manifest -- whether from
# one family or a whole missing family -- fails loudly and names where. When the
# corpus legitimately changes, these numbers get updated as part of that change.
EXPECTED_FAMILY_COUNTS = {
    "anima": 7,
    "dataset-tools": 2,
    "detail": 1,
    "klein": 2,
    "krea2": 2,
    "ltx2.3": 75,
    "minimax-h3": 3,
    "qwen": 10,
    "superflow": 6,
    "wan": 1,
}


def test_every_manifest_parses(tmp_path):
    reg = builtin_registry(tmp_path)
    manifests = reg.discover()
    assert len(manifests) >= 90
    for m in manifests:
        assert m.output_node_title


def test_discovery_reconciles_with_the_manifests_committed_to_disk(tmp_path):
    """A parse-level drop (discovered a subset of what's on disk) is exactly the
    silent-data-loss failure mode this task exists to catch."""
    reg = builtin_registry(tmp_path)
    manifests = reg.discover()
    on_disk = list(REG.glob("*/*.manifest.yaml"))
    assert len(manifests) == len(on_disk)


def test_every_family_has_its_exact_committed_manifest_count(tmp_path):
    reg = builtin_registry(tmp_path)
    observed = {}
    for m in reg.discover():
        observed[m.family] = observed.get(m.family, 0) + 1
    assert observed == EXPECTED_FAMILY_COUNTS


def test_every_manifest_points_at_a_workflow_json_that_exists(tmp_path):
    for m in builtin_registry(tmp_path).discover():
        assert m.workflow_path.is_file(), f"{m.name}: missing {m.workflow_path}"


def test_no_name_collides_across_families(tmp_path):
    assert builtin_registry(tmp_path).shadowed() == []


def test_every_family_declares_metadata():
    families = [d for d in REG.iterdir() if d.is_dir()]
    assert families
    for d in families:
        assert (d / "family.yaml").is_file(), f"{d.name} has no family.yaml"


def test_every_family_declares_a_known_prompt_dialect():
    for d in (p for p in REG.iterdir() if p.is_dir()):
        raw = yaml.safe_load((d / "family.yaml").read_text(encoding="utf-8"))
        assert raw["prompt_dialect"] in KNOWN_DIALECTS, d.name
        assert raw["base_model_family"], d.name


def test_anima_and_illustrious_are_distinct_families_sharing_one_lineage():
    anima = yaml.safe_load((REG / "anima" / "family.yaml").read_text(encoding="utf-8"))
    assert anima["base_model_family"] == "illustrious"
    assert anima["prompt_dialect"] == "booru"


def test_every_flag_carries_a_node_id_or_a_node_title(tmp_path):
    for m in builtin_registry(tmp_path).discover():
        for key, f in m.flags.items():
            assert f.node or f.node_title, f"{m.name}.{key} addresses no node"


def test_canonical_names_resolve_across_every_family(tmp_path):
    """The concept run_batch drives must be reachable on every workflow that has it,
    whatever that manifest happens to call it (Task 8b)."""
    reg = builtin_registry(tmp_path)
    for m in reg.discover():
        for canon, aliases in [("positive_text", ["prompt", "positive_prompt"]),
                               ("negative_text", ["negative", "negative_prompt"])]:
            present = [k for k in m.flags if k in ([canon] + aliases)]
            if not present:
                continue
            assert m.resolve_key(canon) in m.flags, \
                f"{m.family}/{m.name} declares {present} but {canon!r} does not resolve"


def test_alias_keyed_manifests_are_reported_by_the_lint(tmp_path):
    """Not a failure -- aliases keep working. This surfaces the list so keys can be
    canonicalised over time."""
    from comfyrack.registry import canonical_of, lint_manifest
    manifests = builtin_registry(tmp_path).discover()
    problems = [(m, p) for m in manifests for p in lint_manifest(m)]
    for _, line in problems:
        print(line)
    # A lint returning [] for everything, and one flagging canonical keys too, would
    # both satisfy `isinstance(problems, list)`. Pin the actual relationship: the set
    # the lint reports must equal the set that really has a non-canonical flag key.
    for m, line in problems:
        assert line.startswith(f"{m.family}/{m.name}: ")
    assert {(m.family, m.name) for m, _ in problems} == \
        {(m.family, m.name) for m in manifests
         if any(canonical_of(k) != k for k in m.flags)}


# -- mode taxonomy ---------------------------------------
#
# TOKEN_MAP/parse_modes mirror the parsing scripts/_scratch_mode_apply.py used to
# backfill mode: into every manifest, kept here (not imported -- that script is
# scratch and gets deleted) so (b) below is a real regression check against the
# corpus's own filenames, not a hand-picked fixture.
_MODE_TOKEN_MAP = {
    "i2v": "i2v", "t2v": "t2v", "v2v": "v2v", "i2i": "i2i", "t2i": "t2i",
    "r2v": "r2v", "r2i": "r2i",
    "txt2img": "t2i", "txt2vid": "t2v",
    "flf2v": "r2v", "fml2v": "r2v",
    "iv2v": "v2v", "tv2v": "v2v", "f2f": "v2v",
}
_MODE_TOKENS_BY_LEN = sorted(_MODE_TOKEN_MAP.keys(), key=len, reverse=True)


def _parse_modes_from_filename(filename: str) -> set:
    import re
    stem = filename.lower().replace(".json", "")
    parts = re.split(r"[-_]", stem)
    found = set()
    for part in parts:
        for tok in _MODE_TOKENS_BY_LEN:
            if tok in part:
                found.add(_MODE_TOKEN_MAP[tok])
    return found


def test_every_manifest_has_a_non_empty_mode(tmp_path):
    reg = builtin_registry(tmp_path)
    for m in reg.discover():
        assert m.mode, f"{m.family}/{m.name} has no mode"
        for mode in m.mode:
            assert mode in {"t2i", "t2v", "i2i", "i2v", "v2v", "r2v", "r2i"}, \
                f"{m.family}/{m.name}: unknown mode {mode!r}"


def test_mode_i2v_matches_exactly_the_manifests_whose_workflow_filename_encodes_i2v(tmp_path):
    reg = builtin_registry(tmp_path)
    manifests = reg.discover()
    expected = {m.name for m in manifests
                if "i2v" in _parse_modes_from_filename(m.workflow_path.name)}
    actual = {m.name for m in manifests if "i2v" in m.mode}
    assert expected, "sanity: the real corpus must have at least one i2v workflow"
    assert actual == expected


def test_a_non_video_family_is_tagged_t2i_or_i2i_by_the_real_registry(tmp_path):
    """krea2 and qwen are both text-to-image/image-editing families -- neither
    should ever carry a video mode."""
    reg = builtin_registry(tmp_path)
    manifests = [m for m in reg.discover() if m.family in ("krea2", "qwen")]
    assert manifests
    for m in manifests:
        assert m.mode, f"{m.family}/{m.name} has no mode"
        assert set(m.mode) <= {"t2i", "i2i"}, \
            f"{m.family}/{m.name}: expected only t2i/i2i, got {m.mode}"
    # And at least one of each, so this isn't vacuously true.
    all_modes = {mode for m in manifests for mode in m.mode}
    assert "t2i" in all_modes or "i2i" in all_modes


def test_every_flag_addresses_an_input_that_actually_exists_on_its_node(tmp_path):
    """A flag whose `param` is not a real input key on its node can never be set:
    patch() writes it, the graph carries a stray key, and the workflow renders with
    the baked-in value as if nothing was passed. Found one such flag (`param:
    UNKNOWN` over a `String` node whose input key is `String`) among 1,170 imported
    flags, whose own help text claimed it had been verified against the JSON."""
    import json
    from comfyrack.graph import Graph

    problems = []
    for m in builtin_registry(tmp_path).discover():
        graph = Graph(json.loads(m.workflow_path.read_text(encoding="utf-8")))
        for key, f in m.flags.items():
            try:
                node_id = graph.resolve(node_id=f.node, title=f.node_title)
            except Exception as exc:            # unresolvable node is its own bug
                problems.append(f"{m.family}/{m.name}.{key}: {exc}")
                continue
            inputs = graph.data[node_id].get("inputs", {})
            if f.param not in inputs:
                problems.append(
                    f"{m.family}/{m.name}.{key}: param {f.param!r} is not an input on "
                    f"node {node_id} ({graph.data[node_id]['class_type']}); "
                    f"has {sorted(inputs)}")
    assert problems == []
