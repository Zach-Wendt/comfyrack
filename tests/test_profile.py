import pytest
import yaml
from comfyrack.config import Config
from comfyrack.profile import Profile, Character
from comfyrack import errors


def setup(tmp_path):
    (tmp_path / ".comfyrack").mkdir(parents=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[paths]\ncharacters = "chars"\nlocations = "locs"\nrecipes = "recs"\n',
        encoding="utf-8")
    ch = tmp_path / "chars" / "char_a"
    ch.mkdir(parents=True)
    (ch / "recipe.yaml").write_text(yaml.safe_dump({
        "character": "char_a",
        "base_model_family": "anima_illustrious",
        "id_lora": {"name": "your_character_lora.safetensors", "strength": 0.8,
                    "trigger": "trig_a", "locked_date": "2026-07-05"},
        "style_lora": {"name": "your_style_lora.safetensors", "strength": 1.0},
        "sampler": {"sampler_name": "er_sde", "scheduler": "simple", "steps": 20},
        "age_bands": {"default": 0.6, "close_shot": 0.45},
    }), encoding="utf-8")
    loc = tmp_path / "locs" / "bedroom"
    loc.mkdir(parents=True)
    (loc / "reference_plate.png").write_bytes(b"PNG")
    recs = tmp_path / "recs"
    recs.mkdir()
    (recs / "anima_v1.yaml").write_text("steps: 20\ncfg: 4.0\n", encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    return Profile(cfg)


def test_characters_are_discovered_by_directory(tmp_path):
    assert list(setup(tmp_path).characters()) == ["char_a"]


def test_character_carries_the_id_lora_and_its_strength(tmp_path):
    c = setup(tmp_path).character("char_a")
    assert c.id_lora == "your_character_lora.safetensors"
    assert c.id_strength == 0.8
    assert c.trigger == "trig_a"


def test_character_carries_the_style_lora_and_sampler(tmp_path):
    c = setup(tmp_path).character("char_a")
    assert c.style_lora == "your_style_lora.safetensors"
    assert c.style_strength == 1.0
    assert c.sampler["sampler_name"] == "er_sde"


def test_unknown_character_raises_listing_the_known_ones(tmp_path):
    with pytest.raises(errors.NotFoundError) as ei:
        setup(tmp_path).character("nobody")
    assert "char_a" in ei.value.help_text


def test_id_strength_defaults_to_the_age_band_default(tmp_path):
    assert setup(tmp_path).resolve_id_strength("char_a") == 0.6


def test_close_shot_id_strength_uses_the_close_shot_band(tmp_path):
    assert setup(tmp_path).resolve_id_strength("char_a", close_shot=True) == 0.45


def test_a_per_sequence_band_overrides_the_default(tmp_path):
    p = setup(tmp_path)
    path = tmp_path / "chars" / "char_a" / "recipe.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["age_bands"]["sequences"] = {12: 0.7}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    assert Profile(cfg).resolve_id_strength("char_a", sequence=12) == 0.7


def test_a_character_without_age_bands_falls_back_to_the_id_lora_strength(tmp_path):
    p = setup(tmp_path)
    path = tmp_path / "chars" / "char_a" / "recipe.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["age_bands"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    assert Profile(cfg).resolve_id_strength("char_a") == 0.8


def test_locations_are_discovered_by_directory_with_their_plate(tmp_path):
    loc = setup(tmp_path).location("bedroom")
    assert loc.plate_path.name == "reference_plate.png"
    assert loc.plate_path.is_file()


def test_a_location_directory_without_a_plate_is_still_listed(tmp_path):
    p = setup(tmp_path)
    (tmp_path / "locs" / "quarry").mkdir()
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    prof = Profile(cfg)
    assert "quarry" in prof.locations()
    assert prof.location("quarry").plate_path is None


def test_recipes_are_listed_by_name(tmp_path):
    assert setup(tmp_path).recipes() == ["anima_v1"]


def _reload(tmp_path) -> Profile:
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    return Profile(cfg)


def _rewrite_recipe(tmp_path, mutate) -> Profile:
    setup(tmp_path)
    path = tmp_path / "chars" / "char_a" / "recipe.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return _reload(tmp_path)


# -- malformed YAML is a ComfyrackError, not a traceback (finding 4) ----------

def test_a_character_recipe_that_is_not_a_mapping_raises_comfyrack_error(tmp_path):
    setup(tmp_path)
    (tmp_path / "chars" / "char_a" / "recipe.yaml").write_text(
        "- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(errors.ComfyrackError) as ei:
        _reload(tmp_path).characters()
    assert "recipe.yaml" in str(ei.value)


def test_an_unparseable_character_recipe_raises_comfyrack_error(tmp_path):
    setup(tmp_path)
    (tmp_path / "chars" / "char_a" / "recipe.yaml").write_text(
        "id_lora: {name: x\n", encoding="utf-8")
    with pytest.raises(errors.ComfyrackError) as ei:
        _reload(tmp_path).characters()
    assert "recipe.yaml" in str(ei.value)


def test_an_id_lora_block_that_is_not_a_mapping_raises_comfyrack_error(tmp_path):
    p = _rewrite_recipe(tmp_path, lambda raw: raw.__setitem__("id_lora", "just a name"))
    with pytest.raises(errors.ComfyrackError) as ei:
        p.characters()
    assert "id_lora" in str(ei.value)


def test_a_location_file_that_is_not_a_mapping_raises_comfyrack_error(tmp_path):
    setup(tmp_path)
    (tmp_path / "locs" / "bedroom" / "location.yaml").write_text(
        "- a\n- list\n", encoding="utf-8")
    with pytest.raises(errors.ComfyrackError) as ei:
        _reload(tmp_path).locations()
    assert "location.yaml" in str(ei.value)


# -- None, not 0.0, is the "unset" sentinel (findings 5 and 8) ----------------

def test_an_omitted_id_strength_is_unset_not_zero(tmp_path):
    p = _rewrite_recipe(tmp_path, lambda raw: raw["id_lora"].pop("strength"))
    assert p.character("char_a").id_strength is None


def test_an_unset_id_strength_fails_loudly_instead_of_defaulting(tmp_path):
    def mutate(raw):
        raw["id_lora"].pop("strength")
        del raw["age_bands"]
    p = _rewrite_recipe(tmp_path, mutate)
    with pytest.raises(errors.UsageError) as ei:
        p.resolve_id_strength("char_a")
    assert "char_a" in str(ei.value)


def test_an_explicit_zero_id_lora_strength_survives_as_zero(tmp_path):
    def mutate(raw):
        raw["id_lora"]["strength"] = 0
        del raw["age_bands"]
    p = _rewrite_recipe(tmp_path, mutate)
    assert p.character("char_a").id_strength == 0.0
    assert p.resolve_id_strength("char_a") == 0.0


def test_an_explicit_zero_age_band_is_not_treated_as_absent(tmp_path):
    p = _rewrite_recipe(tmp_path, lambda raw: raw["age_bands"].__setitem__("default", 0))
    # 0.8 would mean the falsy zero fell through to the LoRA's locked strength.
    assert p.resolve_id_strength("char_a") == 0.0


def test_an_explicit_zero_sequence_band_is_not_treated_as_absent(tmp_path):
    p = _rewrite_recipe(
        tmp_path, lambda raw: raw["age_bands"].__setitem__("sequences", {12: 0}))
    assert p.resolve_id_strength("char_a", sequence=12) == 0.0


def test_an_omitted_style_strength_is_unset_not_zero(tmp_path):
    p = _rewrite_recipe(tmp_path, lambda raw: raw["style_lora"].pop("strength"))
    assert p.character("char_a").style_strength is None


# -- non-numeric values are a ComfyrackError, not a ValueError (finding 6) ----

def test_a_non_numeric_id_strength_raises_comfyrack_error(tmp_path):
    p = _rewrite_recipe(tmp_path, lambda raw: raw["id_lora"].__setitem__("strength", "high"))
    with pytest.raises(errors.ComfyrackError) as ei:
        p.characters()
    assert "strength" in str(ei.value) and "recipe.yaml" in str(ei.value)


def test_a_non_numeric_age_band_raises_comfyrack_error(tmp_path):
    p = _rewrite_recipe(tmp_path, lambda raw: raw["age_bands"].__setitem__("default", "high"))
    with pytest.raises(errors.ComfyrackError) as ei:
        p.resolve_id_strength("char_a")
    assert "age_bands.default" in str(ei.value)


# -- recipes() and recipe() agree on what exists (finding 7) ------------------

def test_a_yml_recipe_is_listed_like_a_yaml_one(tmp_path):
    p = setup(tmp_path)
    (tmp_path / "recs" / "flux_v2.yml").write_text("steps: 8\n", encoding="utf-8")
    p = _reload(tmp_path)
    assert p.recipes() == ["anima_v1", "flux_v2"]
    assert p.recipe("flux_v2") == {"steps": 8}


def test_recipes_falls_back_to_the_default_dir_like_load_recipe_does(tmp_path):
    (tmp_path / ".comfyrack" / "recipes").mkdir(parents=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\n', encoding="utf-8")
    (tmp_path / ".comfyrack" / "recipes" / "fallback.yaml").write_text(
        "steps: 20\n", encoding="utf-8")
    p = _reload(tmp_path)
    # recipe() finds it via load_recipe's fallback, so list must report it too.
    assert p.recipe("fallback") == {"steps": 20}
    assert p.recipes() == ["fallback"]


def test_a_profile_with_no_configured_paths_is_empty_not_an_error(tmp_path):
    (tmp_path / ".comfyrack").mkdir(parents=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[machines]\ndefault = "http://a:8188"\n', encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    p = Profile(cfg)
    assert p.characters() == {} and p.locations() == {} and p.recipes() == []
