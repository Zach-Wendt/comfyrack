import json
import pathlib

import pytest
from comfyrack.graph import Graph, combo_choices, widget_names
from comfyrack import errors

SAMPLE = {
    "3": {"class_type": "KSampler", "_meta": {"title": "KSampler"},
           "inputs": {"seed": 1, "steps": 20, "model": ["4", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "_meta": {"title": "Load Checkpoint"},
           "inputs": {"ckpt_name": "anima_baseV10.safetensors"}},
}

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"
_OI_037 = json.loads((_FIXTURES / "object_info_0_37.json").read_text(encoding="utf-8"))


def test_resolve_prefers_node_id_over_title():
    g = Graph(SAMPLE)
    assert g.resolve(node_id="3", title="Load Checkpoint") == "3"


def test_resolve_falls_back_to_title_when_id_is_absent():
    g = Graph(SAMPLE)
    assert g.resolve(node_id="999", title="Load Checkpoint") == "4"


def test_resolve_raises_when_neither_id_nor_title_matches():
    g = Graph(SAMPLE)
    with pytest.raises(errors.NotFoundError) as ei:
        g.resolve(node_id="999", title="No Such Node")
    assert "No Such Node" in str(ei.value)


def test_resolve_raises_when_a_title_is_ambiguous():
    data = {
        "1": {"class_type": "CLIPTextEncode", "_meta": {"title": "Prompt"}, "inputs": {}},
        "2": {"class_type": "CLIPTextEncode", "_meta": {"title": "Prompt"}, "inputs": {}},
    }
    with pytest.raises(errors.NotFoundError) as ei:
        Graph(data).resolve(title="Prompt")
    assert "ambiguous" in str(ei.value).lower()


def test_set_param_mutates_only_the_named_input():
    g = Graph(SAMPLE).copy()
    g.set_param("3", "seed", 42)
    assert g.data["3"]["inputs"]["seed"] == 42
    assert g.data["3"]["inputs"]["steps"] == 20


def test_set_param_on_an_unknown_param_is_a_usage_error_listing_valid_params():
    g = Graph(SAMPLE).copy()
    with pytest.raises(errors.UsageError) as ei:
        g.set_param("3", "cfgg", 7)
    assert "steps" in ei.value.help_text


def test_copy_is_deep_so_mutations_do_not_leak():
    g1 = Graph(SAMPLE)
    g2 = g1.copy()
    g2.set_param("3", "seed", 999)
    assert g1.data["3"]["inputs"]["seed"] == 1


def test_nodes_lists_id_title_and_class_type():
    # Assert on node "4", whose title and class_type DIFFER. Node "3" has both set to
    # "KSampler", so a tuple built in the wrong field order would satisfy it either way
    # and the assertion would prove nothing about ordering.
    assert ("4", "Load Checkpoint", "CheckpointLoaderSimple") in Graph(SAMPLE).nodes()


def test_class_types_returns_the_distinct_set():
    assert Graph(SAMPLE).class_types() == {"KSampler", "CheckpointLoaderSimple"}


def test_content_hash_is_stable_across_key_order_and_changes_with_values():
    a = Graph({"1": {"class_type": "X", "inputs": {"p": 1, "q": 2}}})
    b = Graph({"1": {"class_type": "X", "inputs": {"q": 2, "p": 1}}})
    c = Graph({"1": {"class_type": "X", "inputs": {"p": 9, "q": 2}}})
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()


def test_widget_names_returns_widget_inputs_in_declaration_order():
    entry = {"input": {
        "required": {"model": ["MODEL"], "seed": ["INT", {}], "steps": ["INT", {}],
                     "sampler_name": [["euler", "er_sde"], {}]},
        "optional": {"denoise": ["FLOAT", {}]},
    }}
    assert widget_names(entry) == ["seed", "steps", "sampler_name", "denoise"]


def test_combo_choices_recognises_old_list_shape():
    assert combo_choices([["euler", "er_sde"], {"tooltip": "x"}]) == ["euler", "er_sde"]


def test_combo_choices_recognises_v3_combo_shape():
    spec = ["COMBO", {"multiselect": False, "options": ["4x-UltraSharp.pth", "4x-UltraSharpV2.pth"]}]
    assert combo_choices(spec) == ["4x-UltraSharp.pth", "4x-UltraSharpV2.pth"]


def test_combo_choices_recognises_dynamic_combo_shape():
    spec = ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "a", "inputs": {}}, {"key": "b", "inputs": {}}]}]
    assert combo_choices(spec) == ["a", "b"]


def test_combo_choices_returns_none_for_primitives_and_link_types():
    assert combo_choices(["INT", {"min": 0}]) is None
    assert combo_choices(["FLOAT", {}]) is None
    assert combo_choices(["STRING", {}]) is None
    assert combo_choices(["BOOLEAN", {}]) is None
    assert combo_choices(["MODEL", {}]) is None
    assert combo_choices(["CLIP", {}]) is None


def test_combo_choices_returns_none_for_non_list_or_empty():
    assert combo_choices(None) is None
    assert combo_choices([]) is None
    assert combo_choices("COMBO") is None


def test_widget_names_includes_v3_combo_inputs_from_fixture():
    assert "model_name" in widget_names(_OI_037["UpscaleModelLoader"])
    assert "sampler_name" in widget_names(_OI_037["KSamplerSelect"])


def test_widget_names_includes_dynamic_combo_from_fixture():
    # ResizeImageMaskNode.resize_type is a COMFY_DYNAMICCOMBO_V3; scale_method
    # is a V3 COMBO. Neither was detected as a widget before the fix.
    widgets = widget_names(_OI_037["ResizeImageMaskNode"])
    assert "resize_type" in widgets
    assert "scale_method" in widgets


def test_widget_names_excludes_matchtype_link_inputs_from_fixture():
    # ResizeImageMaskNode.input is a COMFY_MATCHTYPE_V3 link type, not a widget.
    widgets = widget_names(_OI_037["ResizeImageMaskNode"])
    assert "input" not in widgets


def test_ensure_titles_injects_the_node_id_where_no_title_is_set():
    g = Graph({"7": {"class_type": "KSampler", "inputs": {}}}).ensure_titles()
    assert g.data["7"]["_meta"]["title"] == "7"


def test_ensure_titles_leaves_an_existing_title_alone():
    g = Graph({"7": {"class_type": "KSampler", "_meta": {"title": "Sampler"},
                     "inputs": {}}}).ensure_titles()
    assert g.data["7"]["_meta"]["title"] == "Sampler"
