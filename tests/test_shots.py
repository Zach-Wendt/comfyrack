import pytest
import yaml
from comfyrack.shots import load_shots, validate
from comfyrack import errors

# Fixtures below follow the shape of a real shot list -- `characters:` is a
# mapping of name -> {text: ...}, and `composite:` is a mapping with seed/refs,
# not a list. The text is realistic prose so a silently-dropped description would
# show up as a fixture mismatch, not just a shape mismatch.
SAMPLE_CHARACTERS = {
    "char_d": {
        "text": "an elderly man with short silver hair, calm unhurried bearing, "
                "simple earth-tone teacher's robes, weathered kind face"
    },
    "char_c": {
        "text": "an adult woman in her mid thirties, athletic build, dark hair "
                "pulled back, alert focused face, fitted sea-blue waterbending "
                "movement wrap with teal arm-wraps, attentive watching expression"
    },
}

SAMPLE_S05_COMPOSITE = {
    "seed": 80315,
    "refs": [
        {
            "character": "char_b",
            "clause": "Add the man from Picture 2, pulling the girl into a tight "
                      "embrace right where she has arrived near the dock.",
        }
    ],
    "extra_positive": "(long straight blonde hair:1.3) and (green eyes:1.2) on the girl",
    "extra_negative": "blue arrow tattoo, forehead arrow marking",
}

# Modeled on a real shot list's s01_char_b_net_platform -- the one shape where
# `base.kind` is `location_plate` rather than `native`: no `action_text`
# anywhere (composites onto a pre-rendered plate, no text-conditioned render
# stage of its own) and no seed anywhere except `composite.seed`.
LOCATION_PLATE_SHOT = {
    "name": "s01_char_b_net_platform",
    "via": "shot_builder",
    "base": {"kind": "location_plate", "location": "harborview_coast"},
    "composite": {
        "seed": 90301,
        "refs": [
            {
                "character": "char_b",
                "clause": "Place the man from Picture 2 standing braced on a "
                          "floating wooden platform at the edge of the water, "
                          "both hands hauling a heavy net full of glinting fish.",
            }
        ],
        "extra_positive": "an adult man with a bald head and full dark brown "
                           "beard, broad shoulders",
        "extra_negative": "hair on the man's head, clean-shaven man, beardless",
    },
}

SAMPLE = {
    "sequence": 8,
    "primary_character": "char_a",
    "comfy_url": None,
    "extra_negative": "modern clothing",
    "characters": SAMPLE_CHARACTERS,
    "shots": [
        {"name": "s01_spiral", "character": "char_a", "people": "two_person",
         "positive_prefix": "trig_a, (straight blonde hair:1.4)",
         "action_text": "mid-dance, {char_c} watching closely",
         "location_text": "a wooden training platform over a calm lake",
         "seed": 80201, "id_strength": 0.45},
        {"name": "s05_reunion", "via": "shot_builder",
         "base": {"kind": "native", "character": "char_a", "people": "solo",
                  "positive_prefix": "trig_a", "action_text": "running toward the dock",
                  "location_text": "a coastal settlement", "seed": 80305,
                  "id_strength": 0.45},
         "composite": SAMPLE_S05_COMPOSITE},
    ],
}


def write(tmp_path, data=None):
    p = tmp_path / "seq01_shots.yaml"
    p.write_text(yaml.safe_dump(data or SAMPLE, sort_keys=False), encoding="utf-8")
    return p


def test_load_reads_the_top_level_metadata(tmp_path):
    sl = load_shots(write(tmp_path))
    assert sl.sequence == 8
    assert sl.primary_character == "char_a"
    assert sl.extra_negative == "modern clothing"
    assert sl.characters == SAMPLE_CHARACTERS


def test_chapter_key_is_not_read_as_sequence(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    del data["sequence"]
    data["chapter"] = 12
    assert load_shots(write(tmp_path, data)).sequence is None


def test_characters_mapping_preserves_each_description_not_just_the_names(tmp_path):
    # The real defect: `characters=list(raw.get("characters") or [])` kept only
    # the dict keys and silently discarded every `{text: ...}` description --
    # exactly the prose that grounds `{char_d}` / `{char_c}` placeholders in
    # action_text. Assert the description text itself survives, not just the key.
    sl = load_shots(write(tmp_path))
    assert sl.characters["char_d"]["text"] == SAMPLE_CHARACTERS["char_d"]["text"]
    assert sl.characters["char_c"]["text"] == SAMPLE_CHARACTERS["char_c"]["text"]


def test_a_missing_characters_key_normalizes_to_an_empty_dict(tmp_path):
    # Some real sequence files have no `characters:` key at all (parses as None).
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    del data["characters"]
    assert load_shots(write(tmp_path, data)).characters == {}


def test_a_list_of_character_names_is_tolerated_as_a_dict_of_empty_entries(tmp_path):
    # Not present anywhere in practice, but cheap to tolerate rather than
    # reject outright.
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["characters"] = ["char_a", "char_c"]
    assert load_shots(write(tmp_path, data)).characters == {"char_a": {}, "char_c": {}}


def test_a_non_mapping_non_list_characters_value_is_a_usage_error(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["characters"] = "char_a"
    with pytest.raises(errors.UsageError):
        load_shots(write(tmp_path, data))


def test_load_reads_every_shot_in_order(tmp_path):
    assert [s.name for s in load_shots(write(tmp_path)).shots] == \
        ["s01_spiral", "s05_reunion"]


def test_a_plain_shot_carries_its_prompt_fields_and_seed(tmp_path):
    s = load_shots(write(tmp_path)).shots[0]
    assert s.positive_prefix == "trig_a, (straight blonde hair:1.4)"
    assert s.seed == 80201 and s.id_strength == 0.45
    assert s.via is None


def test_a_shot_builder_shot_exposes_its_base_and_composite(tmp_path):
    s = load_shots(write(tmp_path)).shots[1]
    assert s.via == "shot_builder"
    assert s.base["kind"] == "native"
    assert s.composite["refs"][0]["character"] == "char_b"
    assert s.composite["seed"] == 80315


def test_a_shot_builder_shot_inherits_seed_from_its_base(tmp_path):
    assert load_shots(write(tmp_path)).shots[1].seed == 80305


def test_a_plain_shot_with_no_composite_still_gets_a_normalized_empty_composite(tmp_path):
    s = load_shots(write(tmp_path)).shots[0]
    assert s.composite == {"seed": None, "refs": []}


def test_a_non_mapping_composite_value_is_a_usage_error(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"][1]["composite"] = [{"character": "char_b"}]
    with pytest.raises(errors.UsageError):
        load_shots(write(tmp_path, data))


@pytest.mark.parametrize("field_name,top_value", [
    ("character", "char_a-top"),
    ("people", "two_person-top"),
    ("positive_prefix", "trig_a-top"),
    ("action_text", "top-level action"),
    ("location_text", "top-level location"),
    ("extra_negative", "top-level negative"),
])
def test_a_shot_builder_shot_falls_back_to_a_top_level_field_missing_from_base(
    tmp_path, field_name, top_value
):
    # A hand-authored `via` shot may write a prompt field top-level instead of
    # under `base:`. Every prompt field must read through identically --
    # previously only `extra_negative` did, so `character`/`people`/`seed`/
    # `positive_prefix` written top-level silently yielded "" / None.
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"][1]["base"].pop(field_name, None)
    data["shots"][1][field_name] = top_value
    s = load_shots(write(tmp_path, data)).shots[1]
    assert getattr(s, field_name) == top_value


def test_a_shot_builder_shot_falls_back_to_a_top_level_seed_missing_from_base(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    del data["shots"][1]["base"]["seed"]
    data["shots"][1]["seed"] = 99999
    s = load_shots(write(tmp_path, data)).shots[1]
    assert s.seed == 99999


def test_a_shot_builder_shot_falls_back_to_a_top_level_id_strength_missing_from_base(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    del data["shots"][1]["base"]["id_strength"]
    data["shots"][1]["id_strength"] = 0.9
    s = load_shots(write(tmp_path, data)).shots[1]
    assert s.id_strength == 0.9


def test_comfy_url_becomes_the_machine_target(tmp_path):
    data = dict(SAMPLE, comfy_url="http://192.0.2.10:8188")
    assert load_shots(write(tmp_path, data)).machine == "http://192.0.2.10:8188"


def test_the_raw_shot_dict_is_preserved_for_fields_not_yet_modelled(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"][0]["future_field"] = "value"
    assert load_shots(write(tmp_path, data)).shots[0].raw["future_field"] == "value"


def test_a_missing_shots_key_is_a_usage_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"sequence": 1}), encoding="utf-8")
    with pytest.raises(errors.UsageError) as ei:
        load_shots(p)
    assert "shots" in str(ei.value)


def test_a_none_shots_key_is_a_usage_error_not_a_raw_traceback(tmp_path):
    # A hand-edited file with an empty `shots:` key parses as None. Previously
    # `raw["shots"]` was iterated unconditionally, raising a bare
    # `TypeError: 'NoneType' is not iterable`.
    p = tmp_path / "empty_shots.yaml"
    p.write_text("sequence: 1\nshots:\n", encoding="utf-8")
    with pytest.raises(errors.UsageError) as ei:
        load_shots(p)
    assert "shots" in str(ei.value)


def test_a_non_list_shots_key_is_a_usage_error(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"] = {"name": "not a list"}
    with pytest.raises(errors.UsageError):
        load_shots(write(tmp_path, data))


def test_a_non_mapping_document_root_is_a_usage_error(tmp_path):
    # A list root that happens to contain the string "shots" as an element would
    # slip past a bare `"shots" not in raw` check (list membership, not a key
    # lookup) and hit `raw.get(...)` next with a bare AttributeError -- this must
    # be caught by an explicit root-shape check, not accidentally by the
    # membership test.
    p = tmp_path / "list_root.yaml"
    p.write_text(yaml.safe_dump(["shots", "not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(errors.UsageError):
        load_shots(p)


def test_validate_reports_a_shot_with_no_name(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    del data["shots"][0]["name"]
    problems = validate(load_shots(write(tmp_path, data)))
    assert "shot #0: missing `name`" in problems


def test_validate_reports_duplicate_shot_names(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"][1] = dict(data["shots"][0])
    problems = validate(load_shots(write(tmp_path, data)))
    assert "s01_spiral: duplicate shot name" in problems


def test_validate_reports_a_shot_with_no_seed(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    del data["shots"][0]["seed"]
    problems = validate(load_shots(write(tmp_path, data)))
    assert any("seed" in p for p in problems)


def test_validate_reports_a_via_shot_with_no_composite_refs(tmp_path):
    # A `via: shot_builder` shot with no `refs` previously validated clean --
    # `validate()` skipped `action_text` for `via` shots and never looked at
    # `composite` at all.
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"][1]["composite"] = {"seed": 80315, "refs": []}
    problems = validate(load_shots(write(tmp_path, data)))
    assert "s05_reunion: `via: shot_builder` shot has no `composite.refs`" in problems


def test_a_valid_shot_list_validates_clean(tmp_path):
    assert validate(load_shots(write(tmp_path))) == []


def test_a_location_plate_shot_needs_no_action_text(tmp_path):
    # seq02_shots.yaml's s01_char_b_net_platform is a real, shipped shot: a
    # `via: shot_builder` shot whose `base.kind` is `location_plate` --
    # composited straight onto a pre-rendered plate, with no text-conditioned
    # render stage and therefore no `action_text` anywhere. `validate()` must
    # not reject this, or `comfyrack batch --validate` would refuse to render
    # a real sequence.
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"] = [LOCATION_PLATE_SHOT]
    problems = validate(load_shots(write(tmp_path, data)))
    assert not any("action_text" in p for p in problems)


def test_a_native_via_shot_still_needs_action_text(tmp_path):
    # The other direction: a `via: shot_builder` shot whose base kind is
    # `native` (the sample shot list's s05 is exactly this) still has a real text-conditioned
    # render stage, so a missing action_text there is a genuine problem.
    # `location_plate` is a specific exemption, not "via shots never need
    # action_text."
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    del data["shots"][1]["base"]["action_text"]
    problems = validate(load_shots(write(tmp_path, data)))
    assert any("action_text" in p for p in problems)


def test_a_full_location_plate_shot_validates_clean(tmp_path):
    # The exact real shape end to end: no action_text, no seed except on
    # composite, refs present -- should report zero problems.
    data = {"sequence": 9, "shots": [LOCATION_PLATE_SHOT]}
    assert validate(load_shots(write(tmp_path, data))) == []


def test_shot_seed_falls_back_to_the_composite_seed_when_shot_and_base_have_none(tmp_path):
    data = {"sequence": 9, "shots": [LOCATION_PLATE_SHOT]}
    s = load_shots(write(tmp_path, data)).shots[0]
    assert s.seed == 90301


def test_shot_seed_does_not_fall_back_when_the_base_already_has_its_own_seed(tmp_path):
    # the sample shot list's s05/s06/s09 all carry two distinct seeds -- one for the native
    # render (`base.seed`), a different one for the compositing step
    # (`composite.seed`). Falling back must never shadow the shot's own seed
    # with the composite step's different seed.
    s = load_shots(write(tmp_path)).shots[1]
    assert s.seed == 80305
    assert s.composite["seed"] == 80315
    assert s.seed != s.composite["seed"]


def test_validate_still_reports_a_shot_with_no_seed_anywhere_composite_included(tmp_path):
    # The composite-seed fallback must not mask a genuinely seedless shot.
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"] = [dict(LOCATION_PLATE_SHOT)]
    data["shots"][0]["composite"] = dict(LOCATION_PLATE_SHOT["composite"])
    del data["shots"][0]["composite"]["seed"]
    problems = validate(load_shots(write(tmp_path, data)))
    assert any("seed" in p for p in problems)
