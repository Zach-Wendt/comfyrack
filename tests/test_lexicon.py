import pytest
import yaml
from comfyrack.config import Config
from comfyrack.lexicon import Lexicon, Entry
from comfyrack import errors


def make_lexicon(tmp_path):
    (tmp_path / ".comfyrack").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[paths]\nlexicon = "lex"\n', encoding="utf-8")
    (tmp_path / "lex").mkdir(exist_ok=True)
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    return Lexicon(cfg)


def write_entry(tmp_path, term, kind="character", renders=None):
    renders = renders or {
        "illustrious": {"trigger": "trig_a",
                        "tags": "1girl, (straight blonde hair:1.2), fair skin",
                        "lora": "your_character_lora.safetensors"},
        "flux": {"text": "a young girl with straight blonde hair and fair skin"},
    }
    (tmp_path / "lex" / f"{term}.yaml").write_text(
        yaml.safe_dump({"term": term, "kind": kind, "renders": renders}, sort_keys=False),
        encoding="utf-8")


def test_all_loads_every_entry_in_the_lexicon_directory(tmp_path):
    lex = make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a")
    write_entry(tmp_path, "char_b")
    assert set(make_lexicon(tmp_path).all()) == {"char_a", "char_b"}


def test_get_returns_a_typed_entry(tmp_path):
    make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a")
    e = make_lexicon(tmp_path).get("char_a")
    assert isinstance(e, Entry) and e.kind == "character"


def test_get_unknown_term_raises_with_the_add_command_as_the_fix(tmp_path):
    lex = make_lexicon(tmp_path)
    with pytest.raises(errors.NotFoundError) as ei:
        lex.get("harborview")
    assert "comfyrack lex add" in ei.value.help_text


def test_render_for_a_booru_lineage_puts_the_trigger_first(tmp_path):
    make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a")
    rendered = make_lexicon(tmp_path).render("char_a", "illustrious")
    assert rendered.startswith("trig_a, ")
    assert "(straight blonde hair:1.2)" in rendered


def test_render_for_a_natural_lineage_returns_prose(tmp_path):
    make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a")
    rendered = make_lexicon(tmp_path).render("char_a", "flux")
    assert rendered.startswith("a young girl")
    assert "trig_a" not in rendered


def test_render_for_a_lineage_with_no_entry_raises_naming_the_lineages_present(tmp_path):
    make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a")
    with pytest.raises(errors.NotFoundError) as ei:
        make_lexicon(tmp_path).render("char_a", "wan")
    assert "illustrious" in ei.value.help_text and "flux" in ei.value.help_text


def test_has_is_true_only_for_known_terms(tmp_path):
    make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a")
    lex = make_lexicon(tmp_path)
    assert lex.has("char_a") and not lex.has("harborview")


def test_term_lookup_is_case_insensitive(tmp_path):
    make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a")
    assert make_lexicon(tmp_path).has("Char_a")


def test_save_writes_a_round_trippable_entry(tmp_path):
    lex = make_lexicon(tmp_path)
    e = Entry(term="harborview-coast", kind="location",
              renders={"illustrious": {"tags": "ocean, wooden dock, overcast"}})
    lex.save(e)
    assert make_lexicon(tmp_path).get("harborview-coast").kind == "location"


def test_lexicon_with_no_configured_path_is_empty_rather_than_an_error(tmp_path):
    (tmp_path / ".comfyrack").mkdir(parents=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\n', encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    assert Lexicon(cfg).all() == {}


def test_render_prepends_trigger_even_when_text_is_present(tmp_path):
    # Natural-language LoRAs routinely have trigger words too. Dropping the
    # trigger because "text" is present produces a fluent prompt with no
    # LoRA activation -- the exact plausible-and-wrong render this ring
    # exists to prevent.
    make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a", renders={
        "krea2": {"trigger": "trig_a", "text": "a young girl with blonde hair",
                  "lora": "mira_natural.safetensors"},
    })
    rendered = make_lexicon(tmp_path).render("char_a", "krea2")
    assert rendered.startswith("trig_a, ")
    assert "a young girl with blonde hair" in rendered


def test_render_raises_on_empty_render_block_naming_term_lineage_and_keys(tmp_path):
    make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a", renders={"wan": {}})
    with pytest.raises(errors.UsageError) as ei:
        make_lexicon(tmp_path).render("char_a", "wan")
    assert "char_a" in str(ei.value) and "wan" in str(ei.value)


def test_render_raises_on_mis_keyed_render_block(tmp_path):
    # "tag:" instead of "tags:" -- a typo'd lexicon entry must not silently
    # render as "" and make the subject vanish from the assembled prompt.
    make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a", renders={"wan": {"tag": "1girl"}})
    with pytest.raises(errors.UsageError) as ei:
        make_lexicon(tmp_path).render("char_a", "wan")
    assert "tag" in ei.value.message


def test_lora_field_survives_loading(tmp_path):
    # Per the plan's Global Constraints: a trigger token is only meaningful
    # when its own LoRA is loaded, so Entry.renders[lineage]["lora"] must
    # survive load -- it is consumed later for machine routing.
    make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a")
    entry = make_lexicon(tmp_path).get("char_a")
    assert entry.renders["illustrious"]["lora"] == "your_character_lora.safetensors"


def test_save_lowercases_the_filename_to_match_the_all_key(tmp_path):
    # Windows filesystems are case-insensitive, so Path.exists() can't tell
    # "char_a.yaml" from "Char_a.yaml" apart -- assert on the actual name
    # returned by save() and the glob listing instead.
    lex = make_lexicon(tmp_path)
    path = lex.save(Entry(term="Char_a", kind="character",
                           renders={"illustrious": {"tags": "1girl"}}))
    assert path.name == "char_a.yaml"
    assert [p.name for p in (tmp_path / "lex").glob("*.yaml")] == ["char_a.yaml"]


def test_save_of_differently_cased_terms_does_not_silently_collide(tmp_path):
    lex = make_lexicon(tmp_path)
    lex.save(Entry(term="Char_a", kind="character",
                    renders={"illustrious": {"tags": "1girl"}}))
    lex.save(Entry(term="char_a", kind="character",
                    renders={"illustrious": {"tags": "1girl, redo"}}))
    # Both writes must have landed on the SAME file (same lowercased key) --
    # if they didn't, all() would silently see only one of them via
    # last-glob-wins with two files present.
    assert len(list((tmp_path / "lex").glob("*.yaml"))) == 1
    assert make_lexicon(tmp_path).all().keys() == {"char_a"}


def test_all_returns_a_copy_mutation_does_not_affect_the_cache(tmp_path):
    lex = make_lexicon(tmp_path)
    write_entry(tmp_path, "char_a")
    first = lex.all()
    first["char_a"].renders["illustrious"]["tags"] = "MUTATED"
    first["mallory"] = Entry(term="mallory")
    second = lex.all()
    assert "mallory" not in second
    assert second["char_a"].renders["illustrious"]["tags"] != "MUTATED"


def test_all_raises_a_comfyrack_error_naming_the_file_when_yaml_is_not_a_mapping(tmp_path):
    make_lexicon(tmp_path)
    (tmp_path / "lex" / "broken.yaml").write_text("- not\n- a\n- mapping\n",
                                                    encoding="utf-8")
    with pytest.raises(errors.ComfyrackError) as ei:
        make_lexicon(tmp_path).all()
    assert "broken.yaml" in ei.value.message


def test_save_with_no_configured_path_raises_usage_error(tmp_path):
    (tmp_path / ".comfyrack").mkdir(parents=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\n', encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    with pytest.raises(errors.UsageError):
        Lexicon(cfg).save(Entry(term="char_a"))


def test_subject_and_scene_kinds_supported_in_lexicon(tmp_path):
    lex = make_lexicon(tmp_path)
    lex.save(Entry(term="cyber-car", kind="subject", renders={"flux": {"text": "a futuristic vehicle"}}))
    lex.save(Entry(term="neon-alley", kind="scene", renders={"flux": {"text": "a dark alley illuminated by neon signs"}}))

    car = lex.get("cyber-car")
    assert car.kind == "subject"
    assert lex.render("cyber-car", "flux") == "a futuristic vehicle"

    alley = lex.get("neon-alley")
    assert alley.kind == "scene"
    assert lex.render("neon-alley", "flux") == "a dark alley illuminated by neon signs"
