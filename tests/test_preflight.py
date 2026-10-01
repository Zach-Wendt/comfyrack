import json
import pathlib

import pytest
from comfyrack.graph import Graph
from comfyrack.preflight import check, raise_if_problems
from comfyrack import errors

_fixtures = pathlib.Path(__file__).parent / "fixtures"
_OI_037 = json.loads((_fixtures / "object_info_0_37.json").read_text(encoding="utf-8"))

OBJECT_INFO = {
    "KSampler": {"input": {"required": {
        "sampler_name": [["euler", "er_sde"], {}],
        "steps": ["INT", {}],
    }}},
    "LoraLoader": {"input": {"required": {
        "lora_name": [["your_character_lora.safetensors", "your_style_lora.safetensors"], {}],
        "strength_model": ["FLOAT", {}],
    }}},
}


def g(data):
    return Graph(data)


def test_missing_node_class_is_reported_with_the_node_id():
    problems = check(g({"1": {"class_type": "SomeCustomNode", "inputs": {}}}), OBJECT_INFO)
    assert [(p.kind, p.node_id) for p in problems] == [("missing_node_class", "1")]


def test_combo_value_not_in_the_enumerated_options_is_reported():
    problems = check(g({"1": {"class_type": "KSampler",
                              "inputs": {"sampler_name": "dpmpp_3m", "steps": 20}}}),
                     OBJECT_INFO)
    assert problems[0].kind == "invalid_combo_value"
    assert "dpmpp_3m" in problems[0].detail


def test_a_valid_graph_produces_no_problems():
    assert check(g({"1": {"class_type": "KSampler",
                          "inputs": {"sampler_name": "er_sde", "steps": 20}}}),
                 OBJECT_INFO) == []


# NOTE: `test_a_real_link_is_still_skipped` used to sit further down this file with
# an identical graph and an identical assertion to this test. Deleted as a duplicate;
# this is the one that survives.
def test_linked_inputs_are_not_validated_as_combo_values():
    problems = check(g({"1": {"class_type": "KSampler",
                              "inputs": {"sampler_name": ["2", 0], "steps": 20}},
                        "2": {"class_type": "KSampler", "inputs": {"sampler_name": "euler", "steps": 1}}}),
                     OBJECT_INFO)
    assert problems == []


def test_every_problem_is_reported_not_just_the_first():
    problems = check(g({
        "1": {"class_type": "Nope", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {"sampler_name": "bad", "steps": 1}},
    }), OBJECT_INFO)
    assert len(problems) == 2


def test_lora_lineage_mismatch_against_the_checkpoint_is_reported():
    problems = check(
        g({"1": {"class_type": "LoraLoader",
                 "inputs": {"lora_name": "your_character_lora.safetensors",
                            "strength_model": 0.8}}}),
        OBJECT_INFO,
        lora_meta={"your_character_lora.safetensors": {"base_model_family": "illustrious"}},
        checkpoint_lineage="flux",
    )
    assert problems[0].kind == "lineage_mismatch"
    assert "illustrious" in problems[0].detail and "flux" in problems[0].detail


def test_matching_lora_lineage_produces_no_problem():
    problems = check(
        g({"1": {"class_type": "LoraLoader",
                 "inputs": {"lora_name": "your_character_lora.safetensors",
                            "strength_model": 0.8}}}),
        OBJECT_INFO,
        lora_meta={"your_character_lora.safetensors": {"base_model_family": "illustrious"}},
        checkpoint_lineage="illustrious",
    )
    assert problems == []


def test_raise_if_problems_names_the_machine_and_lists_every_problem():
    problems = check(g({"1": {"class_type": "Nope", "inputs": {}}}), OBJECT_INFO)
    with pytest.raises(errors.ComfyrackError) as ei:
        raise_if_problems(problems, "http://remote:8188")
    assert "http://remote:8188" in str(ei.value)
    assert "Nope" in str(ei.value)


def test_raise_if_problems_is_a_no_op_when_there_are_none():
    raise_if_problems([], "http://remote:8188")


def test_a_multi_select_combo_value_is_validated_not_skipped_as_a_link():
    """A multi-select combo's value is a JSON array. Treating every list as a link
    would silently skip validating it."""
    oi = {"Picker": {"input": {"required": {"choices": [["a", "b", "c"], {}]}}}}
    problems = check(g({"1": {"class_type": "Picker",
                              "inputs": {"choices": ["a", "zzz"]}}}), oi)
    assert problems and problems[0].kind == "invalid_combo_value"


def test_lineage_is_checked_on_a_stacker_node_whose_param_is_not_lora_name():
    oi = {"LoraStacker": {"input": {"required": {
        "lora_01": [["your_character_lora.safetensors"], {}]}}}}
    problems = check(
        g({"1": {"class_type": "LoraStacker",
                 "inputs": {"lora_01": "your_character_lora.safetensors"}}}),
        oi,
        lora_meta={"your_character_lora.safetensors": {"base_model_family": "illustrious"}},
        checkpoint_lineage="flux",
    )
    assert any(p.kind == "lineage_mismatch" for p in problems)


# -- V3 combo validation (real-world /object_info shapes from the fixture) -----

def _upscale_oi():
    """UpscaleModelLoader in 0.37 emits a V3 COMBO, not an old list."""
    return {"UpscaleModelLoader": _OI_037["UpscaleModelLoader"]}


def test_preflight_flags_a_bad_v3_combo_value():
    oi = _upscale_oi()
    problems = check(g({"1": {"class_type": "UpscaleModelLoader",
                              "inputs": {"model_name": "does_not_exist.pth"}}}), oi)
    assert problems[0].kind == "invalid_combo_value"
    assert "does_not_exist.pth" in problems[0].detail


def test_preflight_passes_a_good_v3_combo_value():
    oi = _upscale_oi()
    good = _OI_037["UpscaleModelLoader"]["input"]["required"]["model_name"][1]["options"][0]
    problems = check(g({"1": {"class_type": "UpscaleModelLoader",
                              "inputs": {"model_name": good}}}), oi)
    assert problems == []


def test_multiselect_combo_checks_each_chosen_value():
    oi = {"Pick": {"input": {"required": {"m": ["COMBO", {"multiselect": True,
                                                          "options": ["a", "b"]}]}}}}
    ok = Graph({"1": {"class_type": "Pick", "inputs": {"m": ["a", "b"]}}})
    bad = Graph({"1": {"class_type": "Pick", "inputs": {"m": ["a", "zzz"]}}})
    assert check(ok, oi) == []
    assert [p.kind for p in check(bad, oi)] == ["invalid_combo_value"]
