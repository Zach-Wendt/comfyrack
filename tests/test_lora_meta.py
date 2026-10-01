import json
import struct
import pytest
from comfyrack.lora_meta import read_header, read_meta, scan
from comfyrack import errors


def write_safetensors(path, metadata: dict):
    """Minimal valid safetensors: 8-byte little-endian header length, then JSON."""
    header = {"__metadata__": metadata,
              "lora_down.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00")


def mira_metadata():
    return {
        "ss_output_name": "mira_cel_v2_anima_lora",
        "ss_base_model_version": "anima",
        "ss_sd_model_name": "anima_baseV10.safetensors",
        "ss_num_train_images": "180",
        "ss_tag_frequency": json.dumps({"img": {
            "trig_a": 10, "1girl": 10, "young teenager": 10,
            "slim athletic dancer build": 10,
            "(straight blonde hair:1.2)": 10,
            "(light sprinkle of freckles across nose:0.9)": 10,
            "fair skin": 10, "blue-green eyes": 10, "earth-tone tunic": 10,
            "smiling": 3,
        }}),
    }


def test_read_header_returns_the_metadata_dict(tmp_path):
    p = tmp_path / "a.safetensors"
    write_safetensors(p, mira_metadata())
    assert read_header(p)["ss_base_model_version"] == "anima"


def test_read_header_on_a_non_safetensors_file_errors_clearly(tmp_path):
    p = tmp_path / "junk.safetensors"
    p.write_bytes(b"not a safetensors file at all")
    with pytest.raises(errors.ComfyrackError) as ei:
        read_header(p)
    assert "safetensors" in str(ei.value).lower()


def test_meta_extracts_lineage_and_train_image_count(tmp_path):
    p = tmp_path / "your_character_lora.safetensors"
    write_safetensors(p, mira_metadata())
    m = read_meta(p)
    assert m.base_model_family == "anima"
    assert m.train_images == 180
    assert m.output_name == "mira_cel_v2_anima_lora"


def test_meta_identifies_the_trigger_token(tmp_path):
    p = tmp_path / "your_character_lora.safetensors"
    write_safetensors(p, mira_metadata())
    assert read_meta(p).trigger == "trig_a"


def test_meta_returns_trained_tags_ordered_by_frequency(tmp_path):
    p = tmp_path / "a.safetensors"
    write_safetensors(p, mira_metadata())
    tags = read_meta(p).tags
    # Every tag but `smiling` shares frequency 10, so position 0 is decided by the
    # alphabetical tie-break -- "(light sprinkle...)" wins it, because "(" sorts
    # before digits and letters. Asserting on tags[0] would pin that artifact and
    # fail. Assert the property that matters: frequency dominates the sort.
    assert tags[-1] == "smiling"        # lowest frequency last
    assert tags.index("trig_a") < tags.index("smiling")
    assert tags.index("1girl") < tags.index("smiling")
    assert len(tags) == 10


def test_weighted_tokens_are_preserved_verbatim(tmp_path):
    p = tmp_path / "a.safetensors"
    write_safetensors(p, mira_metadata())
    assert "(straight blonde hair:1.2)" in read_meta(p).tags


def test_meta_without_tag_frequency_yields_no_trigger_and_no_tags(tmp_path):
    p = tmp_path / "b.safetensors"
    write_safetensors(p, {"ss_base_model_version": "flux"})
    m = read_meta(p)
    assert m.trigger is None and m.tags == []
    assert m.base_model_family == "flux"


def test_scan_maps_filename_to_meta_for_every_lora_in_a_directory(tmp_path):
    write_safetensors(tmp_path / "a.safetensors", mira_metadata())
    write_safetensors(tmp_path / "b.safetensors", {"ss_base_model_version": "flux"})
    got = scan(tmp_path)
    assert set(got) == {"a.safetensors", "b.safetensors"}
    assert got["a.safetensors"].trigger == "trig_a"


def test_scan_skips_unreadable_files_rather_than_aborting(tmp_path):
    write_safetensors(tmp_path / "good.safetensors", mira_metadata())
    (tmp_path / "bad.safetensors").write_bytes(b"garbage")
    got = scan(tmp_path)
    assert "good.safetensors" in got
    assert "bad.safetensors" not in got


def write_raw_header(path, blob: bytes):
    """A file with a VALID length prefix and valid JSON of the wrong shape."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)


def test_a_json_header_that_is_not_an_object_is_a_clear_error(tmp_path):
    p = tmp_path / "arr.safetensors"
    write_raw_header(p, b"[1, 2, 3]")
    with pytest.raises(errors.ComfyrackError) as ei:
        read_header(p)
    assert "not an object" in str(ei.value)


def test_a_non_object_metadata_block_is_a_clear_error(tmp_path):
    p = tmp_path / "weird.safetensors"
    write_raw_header(p, json.dumps({"__metadata__": ["not", "a", "dict"]}).encode())
    with pytest.raises(errors.ComfyrackError) as ei:
        read_header(p)
    assert "__metadata__" in str(ei.value)


def test_scan_survives_a_header_whose_json_root_is_not_an_object(tmp_path):
    """The reason the isinstance guards exist. scan() catches ComfyrackError only,
    so an unguarded .get() on a JSON array raises AttributeError and takes the whole
    directory walk down -- exactly what scan() is supposed to make impossible."""
    write_safetensors(tmp_path / "good.safetensors", mira_metadata())
    write_raw_header(tmp_path / "arr.safetensors", b"[1, 2, 3]")
    assert set(scan(tmp_path)) == {"good.safetensors"}


def test_a_malformed_tag_frequency_block_raises_rather_than_reporting_no_tags(tmp_path):
    """Present-but-corrupt must not be indistinguishable from never-captioned."""
    p = tmp_path / "bad.safetensors"
    write_safetensors(p, {"ss_base_model_version": "anima",
                          "ss_tag_frequency": "{not json at all"})
    with pytest.raises(errors.ComfyrackError) as ei:
        read_meta(p)
    assert "ss_tag_frequency" in str(ei.value)


def test_scan_keys_a_subfoldered_lora_by_its_relative_reference(tmp_path):
    """scan() rglob's subdirectories, so it must key by the path RELATIVE to the
    LoRA root -- the same shape ComfyUI's own combo lists and a `lora_name`
    widget use. Keying by the bare `path.name` made this mapping disagree with
    every machine-side source for the 13 of 206 real LoRAs that sit in a
    subfolder, and disagree asymmetrically: dispatch called them unroutable
    (loud) while the lineage check dropped them (silent)."""
    sub = tmp_path / "characters"
    sub.mkdir()
    write_safetensors(sub / "a.safetensors", mira_metadata())
    got = scan(tmp_path)
    assert set(got) == {"characters/a.safetensors"}
    assert got["characters/a.safetensors"].trigger == "trig_a"


def test_scan_keys_a_top_level_lora_by_its_bare_filename(tmp_path):
    """A LoRA that is not in a subfolder has no subpath to carry, so its key is
    unchanged -- the 193 of 206 unaffected files stay exactly as they were."""
    write_safetensors(tmp_path / "a.safetensors", mira_metadata())
    assert set(scan(tmp_path)) == {"a.safetensors"}
