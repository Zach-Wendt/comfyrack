import pytest
from comfyrack.graph import Graph
from comfyrack.registry import Manifest, Flag
from comfyrack.recipes import parse_set, resolve_values, load_recipe
from comfyrack.config import Config
from comfyrack import errors
from pathlib import Path

GRAPH = Graph({
    "3": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
          "inputs": {"seed": 1, "steps": 20, "cfg": 4.0}},
})


def manifest(**flags):
    return Manifest(name="w", family="anima", layer="builtin", description="",
                    workflow_path=Path("w.json"), output_node_title="Save Image",
                    flags=flags)


def test_parse_set_coerces_using_the_declared_flag_type():
    m = manifest(seed=Flag(param="seed", node="3", type="int"),
                 cfg=Flag(param="cfg", node="3", type="float"))
    assert parse_set(["seed=42", "cfg=7.5"], m) == {"seed": 42, "cfg": 7.5}


def test_parse_set_rejects_an_unknown_key_listing_valid_keys():
    m = manifest(seed=Flag(param="seed", node="3", type="int"))
    with pytest.raises(errors.UsageError) as ei:
        parse_set(["sed=42"], m)
    assert "seed" in ei.value.help_text
    assert ei.value.exit_code == 2


def test_parse_set_rejects_a_pair_without_an_equals_sign():
    m = manifest(seed=Flag(param="seed", node="3", type="int"))
    with pytest.raises(errors.UsageError):
        parse_set(["seed 42"], m)


def test_parse_set_keeps_equals_signs_inside_the_value():
    m = manifest(prompt=Flag(param="text", node="3", type="str"))
    assert parse_set(["prompt=a=b"], m) == {"prompt": "a=b"}


def test_parse_set_reports_a_non_numeric_value_for_an_int_flag():
    m = manifest(seed=Flag(param="seed", node="3", type="int"))
    with pytest.raises(errors.UsageError) as ei:
        parse_set(["seed=abc"], m)
    assert "int" in str(ei.value)


def test_override_beats_recipe_beats_default_beats_baked_value():
    m = manifest(
        seed=Flag(param="seed", node="3", type="int"),
        steps=Flag(param="steps", node="3", type="int"),
        cfg=Flag(param="cfg", node="3", type="float", default=6.0),
    )
    values = resolve_values(m, GRAPH, recipe={"steps": 30, "seed": 99},
                            overrides={"seed": 7})
    assert values["seed"] == 7      # override wins
    assert values["steps"] == 30    # recipe wins over baked 20
    assert values["cfg"] == 6.0     # manifest default wins over baked 4.0


def test_baked_workflow_value_is_used_when_nothing_else_supplies_one():
    m = manifest(steps=Flag(param="steps", node="3", type="int"))
    assert resolve_values(m, GRAPH)["steps"] == 20


def test_all_missing_required_flags_are_reported_in_one_error():
    m = manifest(
        zulu=Flag(param="nope_zulu", node="3", required=True),
        yankee=Flag(param="nope_yankee", node="3", required=True),
    )
    with pytest.raises(errors.UsageError) as ei:
        resolve_values(m, GRAPH)
    # Assert on the list AFTER the colon, not on substrings of the whole message.
    # The template reads "missing required flags for 'w': ..." -- the word "flags"
    # contains "a", so `"a" in str(exc)` is true for ANY message and proves nothing.
    listed = str(ei.value).split(": ", 1)[1]
    assert [n.strip() for n in listed.split(",")] == ["yankee", "zulu"]


def test_resolve_values_rejects_an_override_key_the_manifest_does_not_declare():
    """The enabling bug behind the id_strength Critical: resolve_values() only ever
    iterates the manifest's OWN flags, so an override naming a key it does not
    declare used to fall off the end of the loop and vanish -- the caller believed
    it had set a value and nothing said otherwise. parse_set() has always rejected
    the same condition loudly; resolve_values() now does too."""
    m = manifest(seed=Flag(param="seed", node="3", type="int"))
    with pytest.raises(errors.UsageError) as ei:
        resolve_values(m, GRAPH, overrides={"id_strength_model": 0.45})
    assert "id_strength_model" in str(ei.value)
    assert "seed" in ei.value.help_text
    assert ei.value.exit_code == 2


def test_resolve_values_reports_every_unrecognised_override_key_at_once():
    m = manifest(seed=Flag(param="seed", node="3", type="int"))
    with pytest.raises(errors.UsageError) as ei:
        resolve_values(m, GRAPH, overrides={"zulu": 1, "yankee": 2})
    listed = str(ei.value).split(": ", 1)[1]
    assert [n.strip() for n in listed.split(",")] == ["yankee", "zulu"]


def test_a_valid_override_still_resolves_after_the_unknown_key_check():
    m = manifest(seed=Flag(param="seed", node="3", type="int"),
                 steps=Flag(param="steps", node="3", type="int"))
    assert resolve_values(m, GRAPH, overrides={"seed": 7})["seed"] == 7


def test_an_override_spelled_as_an_alias_is_accepted_not_rejected():
    """resolve_key() maps an accepted spelling onto the manifest's own flag key
    BEFORE the unknown-key check, so the new check cannot reject a legitimate
    alias -- `cfg_scale` is a declared alias of `cfg`."""
    m = manifest(cfg=Flag(param="cfg", node="3", type="float"))
    assert resolve_values(m, GRAPH, overrides={"cfg_scale": 7.5})["cfg"] == 7.5


def test_a_recipe_key_the_manifest_does_not_declare_is_still_tolerated():
    """A recipe is shared across sibling workflows in a family, so a key one of
    them does not declare is normal. Only overrides -- per-call instructions aimed
    at THIS workflow -- are held to the loud rule."""
    m = manifest(seed=Flag(param="seed", node="3", type="int"))
    assert resolve_values(m, GRAPH, recipe={"seed": 5, "not_a_flag": 1}) == {"seed": 5}


def test_load_recipe_reads_from_the_configured_recipes_path(tmp_path):
    (tmp_path / ".comfyrack").mkdir()
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[paths]\nrecipes = "myrecipes"\n', encoding="utf-8")
    (tmp_path / "myrecipes").mkdir()
    (tmp_path / "myrecipes" / "anima_v1.yaml").write_text(
        "steps: 20\ncfg: 4.0\n", encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    assert load_recipe(cfg, "anima_v1") == {"steps": 20, "cfg": 4.0}


def test_load_recipe_raises_naming_the_directory_it_searched(tmp_path):
    (tmp_path / ".comfyrack").mkdir()
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[paths]\nrecipes = "myrecipes"\n', encoding="utf-8")
    (tmp_path / "myrecipes").mkdir()
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    with pytest.raises(errors.NotFoundError) as ei:
        load_recipe(cfg, "nope")
    assert "myrecipes" in ei.value.help_text
