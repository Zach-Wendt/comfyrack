import json
import struct
import yaml
from comfyrack.config import Config
from comfyrack.lexicon import Lexicon, Entry


def write_safetensors(path, metadata: dict):
    header = {"__metadata__": metadata,
              "w": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00")


def mira_md():
    return {"ss_output_name": "mira_cel_v2_anima_lora",
            "ss_base_model_version": "anima",
            "ss_num_train_images": "180",
            "ss_tag_frequency": json.dumps({"img": {
                "trig_a": 10, "1girl": 10, "(straight blonde hair:1.2)": 10,
                "fair skin": 10, "blue-green eyes": 10, "smiling": 2}})}


def make(tmp_path):
    (tmp_path / ".comfyrack").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[paths]\nlexicon = "lex"\n', encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    return Lexicon(cfg)


def test_sync_creates_an_entry_per_lora(tmp_path):
    loras = tmp_path / "loras"; loras.mkdir()
    write_safetensors(loras / "your_character_lora.safetensors", mira_md())
    result = make(tmp_path).sync(loras)
    assert ("your_character_lora", "created") in result


def test_synced_entry_uses_the_trigger_and_trained_tags(tmp_path):
    loras = tmp_path / "loras"; loras.mkdir()
    write_safetensors(loras / "your_character_lora.safetensors", mira_md())
    make(tmp_path).sync(loras)
    e = make(tmp_path).get("your_character_lora")
    block = e.renders["illustrious"]
    assert block["trigger"] == "trig_a"
    assert "(straight blonde hair:1.2)" in block["tags"]
    assert block["lora"] == "your_character_lora.safetensors"


def test_lineage_map_translates_the_checkpoint_name_to_a_lineage(tmp_path):
    loras = tmp_path / "loras"; loras.mkdir()
    write_safetensors(loras / "a.safetensors", mira_md())
    make(tmp_path).sync(loras, lineage_map={"anima": "illustrious"})
    assert "illustrious" in make(tmp_path).get("a").renders


def test_unmapped_lineage_is_kept_verbatim_rather_than_dropped(tmp_path):
    loras = tmp_path / "loras"; loras.mkdir()
    md = mira_md(); md["ss_base_model_version"] = "some_new_base"
    write_safetensors(loras / "a.safetensors", md)
    make(tmp_path).sync(loras, lineage_map={"anima": "illustrious"})
    assert "some_new_base" in make(tmp_path).get("a").renders


def test_sync_skips_an_existing_entry_by_default(tmp_path):
    loras = tmp_path / "loras"; loras.mkdir()
    write_safetensors(loras / "a.safetensors", mira_md())
    lex = make(tmp_path)
    lex.save(Entry(term="a", kind="character",
                   renders={"illustrious": {"trigger": "HAND_EDITED"}}))
    result = make(tmp_path).sync(loras)
    assert ("a", "skipped") in result
    assert make(tmp_path).get("a").renders["illustrious"]["trigger"] == "HAND_EDITED"


def test_sync_with_overwrite_replaces_an_existing_entry(tmp_path):
    loras = tmp_path / "loras"; loras.mkdir()
    write_safetensors(loras / "a.safetensors", mira_md())
    lex = make(tmp_path)
    lex.save(Entry(term="a", renders={"illustrious": {"trigger": "HAND_EDITED"}}))
    result = make(tmp_path).sync(loras, overwrite=True)
    assert ("a", "updated") in result
    assert make(tmp_path).get("a").renders["illustrious"]["trigger"] == "trig_a"


def test_a_lora_with_no_trigger_is_reported_as_skipped(tmp_path):
    loras = tmp_path / "loras"; loras.mkdir()
    write_safetensors(loras / "style.safetensors", {"ss_base_model_version": "anima"})
    assert ("style", "skipped") in make(tmp_path).sync(loras)


def test_sync_marks_entries_as_derived_so_hand_edits_are_distinguishable(tmp_path):
    loras = tmp_path / "loras"; loras.mkdir()
    write_safetensors(loras / "a.safetensors", mira_md())
    make(tmp_path).sync(loras)
    raw = yaml.safe_load((tmp_path / "lex" / "a.yaml").read_text(encoding="utf-8"))
    assert raw["renders"]["illustrious"]["derived_from"] == "a.safetensors"


def test_sync_with_overwrite_preserves_other_lineages_on_the_same_entry(tmp_path):
    loras = tmp_path / "loras"; loras.mkdir()
    write_safetensors(loras / "a.safetensors", mira_md())
    lex = make(tmp_path)
    lex.save(Entry(term="a", kind="character", renders={
        "illustrious": {"trigger": "OLD_TRIGGER"},
        "flux": {"trigger": "FLUX_TRIGGER", "text": "hand-authored flux render"},
    }))
    result = make(tmp_path).sync(loras, overwrite=True)
    assert ("a", "updated") in result
    entry = make(tmp_path).get("a")
    # the targeted lineage was updated from the LoRA
    assert entry.renders["illustrious"]["trigger"] == "trig_a"
    # an unrelated lineage on the same entry must survive untouched
    assert entry.renders["flux"] == {
        "trigger": "FLUX_TRIGGER", "text": "hand-authored flux render"}
