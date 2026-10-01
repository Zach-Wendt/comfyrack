import json
import pathlib

from comfyrack import discover


_fixtures = pathlib.Path(__file__).parent / "fixtures"
_OI_037 = json.loads((_fixtures / "object_info_0_37.json").read_text(encoding="utf-8"))


class FakeClient:
    def __init__(self, oi=None, queue=None, stats=None):
        self._oi = oi or {}
        self._queue = queue or {}
        self._stats = stats or {}

    def object_info(self, class_type=None):
        if class_type:
            return {class_type: self._oi[class_type]}
        return self._oi

    def queue_status(self):
        return self._queue

    def system_stats(self):
        return self._stats


OI = {
    "KSampler": {"input": {"required": {"model": ["MODEL"], "steps": ["INT", {}],
                                        "sampler_name": [["euler", "er_sde"], {}]}},
                 "output": ["LATENT"]},
    "CheckpointLoaderSimple": {"input": {"required": {
        "ckpt_name": [["anima_baseV10.safetensors", "krea2.safetensors"], {}]}},
        "output": ["MODEL", "CLIP", "VAE"]},
    "LoraLoader": {"input": {"required": {
        "lora_name": [["your_character_lora.safetensors"], {}]}}, "output": ["MODEL"]},
}


def test_node_types_lists_every_class_sorted():
    assert discover.node_types(FakeClient(OI)) == ["CheckpointLoaderSimple", "KSampler", "LoraLoader"]


def test_node_types_filters_case_insensitively_by_pattern():
    assert discover.node_types(FakeClient(OI), "lora") == ["LoraLoader"]


def test_node_types_with_no_match_returns_an_empty_list():
    assert discover.node_types(FakeClient(OI), "zzz") == []


def test_node_info_separates_widgets_from_link_inputs():
    info = discover.node_info(FakeClient(OI), "KSampler")
    assert info["widgets"] == ["steps", "sampler_name"]
    assert "model" in info["inputs"]
    assert "model" not in info["widgets"]


def test_node_info_includes_output_types():
    assert discover.node_info(FakeClient(OI), "CheckpointLoaderSimple")["outputs"] == \
        ["MODEL", "CLIP", "VAE"]


def test_models_groups_filenames_by_the_loader_that_offers_them():
    got = discover.models(FakeClient(OI))
    assert got["checkpoints"] == ["anima_baseV10.safetensors", "krea2.safetensors"]
    assert got["loras"] == ["your_character_lora.safetensors"]


def test_models_filters_to_one_type_when_asked():
    assert discover.models(FakeClient(OI), "loras") == {"loras": ["your_character_lora.safetensors"]}


# -- V3 combo model discovery against the recorded 0.37 fixture -----------------

def test_models_lists_v3_upscale_models():
    """UpscaleModelLoader in 0.37 emits a V3 COMBO; models() must read its
    options and list them under upscale_models."""
    got = discover.models(FakeClient(_OI_037))
    expected = _OI_037["UpscaleModelLoader"]["input"]["required"]["model_name"][1]["options"]
    assert got["upscale_models"] == expected


def test_models_filters_to_upscale_type():
    got = discover.models(FakeClient(_OI_037), "upscale_models")
    expected = _OI_037["UpscaleModelLoader"]["input"]["required"]["model_name"][1]["options"]
    assert got == {"upscale_models": expected}


def test_queue_summary_counts_running_and_pending():
    c = FakeClient(queue={"queue_running": [["x"]], "queue_pending": [["y"], ["z"]]})
    assert discover.queue_summary(c) == {"running": 1, "pending": 2}


def test_stats_summary_extracts_device_name_and_vram_in_gb():
    c = FakeClient(stats={"devices": [{"name": "NVIDIA RTX 3060",
                                       "vram_total": 12884901888,
                                       "vram_free": 6442450944}]})
    s = discover.stats_summary(c)
    assert s["device"] == "NVIDIA RTX 3060"
    assert s["vram_total_gb"] == 12.0
    assert s["vram_free_gb"] == 6.0


# -- unknown node type is a clean error (final review, Important 7) ----------

class LenientClient(FakeClient):
    """ComfyUI answers /object_info/<unknown> with an empty object rather than a
    404, so this is what the real transport hands node_info for a bad name."""

    def object_info(self, class_type=None):
        if class_type:
            return {class_type: self._oi[class_type]} if class_type in self._oi else {}
        return self._oi


def test_node_info_for_an_unknown_class_raises_not_found_not_a_key_error():
    """`comfyrack node NoSuchNode` gave a bare KeyError traceback where the sibling
    `comfyrack nodes NoSuch` gives a clean empty state."""
    import pytest
    from comfyrack import errors
    with pytest.raises(errors.NotFoundError) as ei:
        discover.node_info(LenientClient(OI), "NoSuchNode")
    assert "NoSuchNode" in ei.value.message
    assert "comfyrack nodes" in (ei.value.help_text or "")


def test_node_info_still_returns_a_known_class_through_the_same_path():
    """The guard must not reject valid names -- `in` on the wrong dict would."""
    info = discover.node_info(LenientClient(OI), "KSampler")
    assert info["class_type"] == "KSampler"
    assert info["outputs"] == ["LATENT"]


# -- model reference normalisation (final review, Important 6) ---------------
#
# ComfyUI names a model by its path RELATIVE to the models dir, using the HOST's
# separator: a Windows box reports `Anima\x.safetensors` where a Linux box
# reports `Anima/x.safetensors` for the same file. Routing compares references
# that came from different machines and different sources (a hand-written
# character recipe, a scanned directory, an /object_info combo list), so the
# separator and an omitted subfolder both have to stop mattering before anything
# is compared. Both consumers -- dispatch._missing() and preflight's lineage
# check -- go through these, which is what keeps them from disagreeing.

def test_normalize_ref_unifies_separators_for_the_same_file():
    assert (discover.normalize_ref("Anima\\x.safetensors")
            == discover.normalize_ref("Anima/x.safetensors")
            == "Anima/x.safetensors")


def test_normalize_ref_tolerates_none_and_a_leading_separator():
    assert discover.normalize_ref(None) == ""
    assert discover.normalize_ref("/Anima/x.safetensors") == "Anima/x.safetensors"


def test_ref_lookup_resolves_a_bare_name_against_a_subfoldered_entry():
    index = discover.ref_index({"Anima\\x.safetensors": "meta"})
    assert discover.ref_lookup(index, "x.safetensors") == "meta"
    assert discover.ref_lookup(index, "Anima/x.safetensors") == "meta"
    assert discover.ref_lookup(index, "Anima\\x.safetensors") == "meta"


def test_ref_lookup_resolves_a_subfoldered_name_against_a_bare_entry():
    index = discover.ref_index({"x.safetensors": "meta"})
    assert discover.ref_lookup(index, "Anima/x.safetensors") == "meta"


def test_ref_lookup_returns_none_for_a_name_nothing_holds():
    index = discover.ref_index({"Anima/x.safetensors": "meta"})
    assert discover.ref_lookup(index, "y.safetensors") is None


def test_ref_index_refuses_to_resolve_an_ambiguous_bare_name():
    """A bare name claimed by two subfolders is registered by NEITHER -- the
    same rule Manifest._build_alias_map follows, because silently picking one
    resolves to the wrong file."""
    index = discover.ref_index({"Anima/x.safetensors": "a", "Krea/x.safetensors": "b"})
    assert discover.ref_lookup(index, "x.safetensors") is None
    assert discover.ref_lookup(index, "Anima/x.safetensors") == "a"
    assert discover.ref_lookup(index, "Krea/x.safetensors") == "b"


def test_ref_index_accepts_a_plain_list_of_names():
    """dispatch passes a machine's enumerated model list, not a mapping."""
    index = discover.ref_index(["Anima/x.safetensors"])
    assert discover.ref_lookup(index, "x.safetensors") == "Anima/x.safetensors"
