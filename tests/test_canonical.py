import pytest
from pathlib import Path
from comfyrack.graph import Graph
from comfyrack.registry import Manifest, Flag, canonical_of, lint_manifest
from comfyrack.recipes import parse_set, resolve_values
from comfyrack import errors

GRAPH = Graph({
    "6": {"class_type": "CLIPTextEncode", "_meta": {"title": "Positive"},
          "inputs": {"text": "baked"}},
    "3": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
          "inputs": {"seed": 1, "steps": 20}},
})


def manifest(**flags):
    return Manifest(name="w", family="f", layer="builtin", description="",
                    workflow_path=Path("w.json"), output_node_title="Save Image",
                    flags=flags)


def anima_style():
    """Uses the canonical name as its key, like anima/character-scene."""
    return manifest(positive_text=Flag(param="text", node="6"),
                    seed=Flag(param="seed", node="3", type="int"))


def wan_style():
    """Uses an alias as its key, like wan/txt2vid."""
    return manifest(positive_prompt=Flag(param="text", node="6"),
                    seed=Flag(param="seed", node="3", type="int"))


def test_canonical_of_maps_a_known_alias_to_its_canonical_name():
    assert canonical_of("prompt") == "positive_text"
    assert canonical_of("positive_prompt") == "positive_text"
    assert canonical_of("negative") == "negative_text"


def test_canonical_of_returns_a_canonical_name_unchanged():
    assert canonical_of("positive_text") == "positive_text"


def test_canonical_of_returns_an_unknown_name_unchanged():
    assert canonical_of("finishing_noise_strength") == "finishing_noise_strength"


def test_a_manifests_own_key_always_resolves_to_itself():
    assert anima_style().resolve_key("positive_text") == "positive_text"
    assert wan_style().resolve_key("positive_prompt") == "positive_prompt"


def test_an_alias_resolves_to_the_manifests_actual_key():
    assert anima_style().resolve_key("prompt") == "positive_text"
    assert anima_style().resolve_key("positive_prompt") == "positive_text"


def test_the_canonical_name_resolves_on_a_manifest_that_uses_an_alias_as_its_key():
    # This is the case that unblocks run_batch: it drives positive_text, wan declares
    # positive_prompt.
    assert wan_style().resolve_key("positive_text") == "positive_prompt"


def test_an_explicit_manifest_alias_resolves():
    m = manifest(finishing=Flag(param="strength", node="3",
                                aliases=["finishing_noise_strength"]))
    assert m.resolve_key("finishing_noise_strength") == "finishing"


def test_an_unknown_key_resolves_to_itself_so_the_caller_reports_it():
    assert anima_style().resolve_key("nonsense") == "nonsense"


def test_two_flags_claiming_one_canonical_name_do_not_register_an_ambiguous_alias():
    m = manifest(positive_text=Flag(param="text", node="6"),
                 prompt=Flag(param="text", node="6"))
    # Both are the same concept; neither wins the shared alias, and each key still
    # resolves to itself.
    assert m.resolve_key("positive_text") == "positive_text"
    assert m.resolve_key("prompt") == "prompt"
    assert m.resolve_key("positive_prompt") == "positive_prompt"   # unresolved


def test_parse_set_accepts_an_alias_and_stores_the_actual_key():
    assert parse_set(["prompt=hello"], anima_style()) == {"positive_text": "hello"}


def test_parse_set_accepts_the_canonical_name_on_an_alias_keyed_manifest():
    assert parse_set(["positive_text=hello"], wan_style()) == \
        {"positive_prompt": "hello"}


def test_parse_set_still_coerces_types_through_an_alias():
    m = manifest(cfg=Flag(param="cfg", node="3", type="float"))
    assert parse_set(["cfg_scale=7.5"], m) == {"cfg": 7.5}


def test_parse_set_rejects_a_genuinely_unknown_key_listing_valid_keys():
    with pytest.raises(errors.UsageError) as ei:
        parse_set(["nonsense=1"], anima_style())
    assert "positive_text" in ei.value.help_text


def test_a_recipe_keyed_canonically_applies_to_an_alias_keyed_manifest():
    values = resolve_values(wan_style(), GRAPH, recipe={"positive_text": "from recipe"})
    assert values["positive_prompt"] == "from recipe"


def test_an_override_beats_a_recipe_even_across_alias_and_canonical_spellings():
    values = resolve_values(wan_style(), GRAPH,
                            recipe={"positive_text": "from recipe"},
                            overrides={"positive_prompt": "from override"})
    assert values["positive_prompt"] == "from override"


def test_lint_flags_a_manifest_using_an_alias_as_its_key():
    problems = lint_manifest(wan_style())
    assert any("positive_prompt" in p and "positive_text" in p for p in problems)


def test_lint_is_silent_on_a_manifest_using_canonical_names():
    assert lint_manifest(anima_style()) == []


def test_aliases_declared_in_a_manifest_yaml_are_loaded_and_resolve(tmp_path):
    """The alias feature is unreachable unless Registry._parse reads `aliases:`."""
    from comfyrack.config import Config
    from comfyrack.registry import Registry

    fam = tmp_path / "builtin" / "krea2"
    fam.mkdir(parents=True)
    (fam / "w.json").write_text("{}", encoding="utf-8")
    (fam / "w.manifest.yaml").write_text(
        'description: "d"\n'
        "workflow: w.json\n"
        'output_node_title: "Save Image"\n'
        "flags:\n"
        "  finishing:\n"
        '    node: "3"\n'
        "    param: strength\n"
        "    type: float\n"
        "    aliases: [finishing_noise_strength]\n",
        encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    m = Registry(cfg, builtin_dir=tmp_path / "builtin",
                 user_dir=tmp_path / "nouser").load("w")
    assert m.flags["finishing"].aliases == ["finishing_noise_strength"]
    assert m.resolve_key("finishing_noise_strength") == "finishing"


def test_a_bare_aliases_key_in_yaml_does_not_crash(tmp_path):
    """`aliases:` with nothing after it parses as None; set.update(None) would raise."""
    from comfyrack.config import Config
    from comfyrack.registry import Registry

    fam = tmp_path / "builtin" / "krea2"
    fam.mkdir(parents=True)
    (fam / "w.json").write_text("{}", encoding="utf-8")
    (fam / "w.manifest.yaml").write_text(
        'description: "d"\n'
        "workflow: w.json\n"
        'output_node_title: "Save Image"\n'
        "flags:\n"
        "  finishing:\n"
        '    node: "3"\n'
        "    param: strength\n"
        "    aliases:\n",
        encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    m = Registry(cfg, builtin_dir=tmp_path / "builtin",
                 user_dir=tmp_path / "nouser").load("w")
    assert m.flags["finishing"].aliases == []
