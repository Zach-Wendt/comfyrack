import json
import pathlib
import pytest
from comfyrack.convert import is_ui_format, ui_to_api, load_workflow
from comfyrack import errors

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

OBJECT_INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["anima_baseV10.safetensors"], {}]}}},
    "KSampler": {"input": {"required": {
        "model": ["MODEL"],
        # seed declares control_after_generate, which is what licenses skipping the
        # "randomize" token that follows it in widgets_values.
        "seed": ["INT", {"control_after_generate": True}],
        "steps": ["INT", {}], "cfg": ["FLOAT", {}],
        "sampler_name": [["euler", "er_sde"], {}],
        "scheduler": [["simple", "fixed"], {}],
        "denoise": ["FLOAT", {}],
    }}},
}


def test_is_ui_format_detects_the_nodes_array_shape():
    assert is_ui_format(json.loads((FIXTURES / "ui_basic.json").read_text()))
    assert not is_ui_format({"3": {"class_type": "KSampler", "inputs": {}}})


def test_conversion_produces_node_id_keys_with_class_type():
    api = ui_to_api(json.loads((FIXTURES / "ui_basic.json").read_text()), OBJECT_INFO)
    assert api["1"]["class_type"] == "CheckpointLoaderSimple"
    assert api["2"]["class_type"] == "KSampler"


def test_widget_values_map_onto_named_inputs_in_declaration_order():
    api = ui_to_api(json.loads((FIXTURES / "ui_basic.json").read_text()), OBJECT_INFO)
    assert api["2"]["inputs"]["seed"] == 123
    assert api["2"]["inputs"]["steps"] == 20
    assert api["2"]["inputs"]["sampler_name"] == "er_sde"


def test_control_after_generate_tokens_are_skipped_not_consumed_as_a_widget():
    # "randomize" follows seed in widgets_values but is a UI-only control, not an input.
    api = ui_to_api(json.loads((FIXTURES / "ui_basic.json").read_text()), OBJECT_INFO)
    assert api["2"]["inputs"]["cfg"] == 4.0


def test_links_become_node_id_slot_pairs():
    api = ui_to_api(json.loads((FIXTURES / "ui_basic.json").read_text()), OBJECT_INFO)
    assert api["2"]["inputs"]["model"] == ["1", 0]


def test_titles_are_preserved_in_meta_and_default_to_the_class_name():
    api = ui_to_api(json.loads((FIXTURES / "ui_basic.json").read_text()), OBJECT_INFO)
    assert api["1"]["_meta"]["title"] == "Load Checkpoint"
    assert api["2"]["_meta"]["title"] == "KSampler"


def test_unknown_class_type_fails_loud_rather_than_half_converting():
    data = {"nodes": [{"id": 1, "type": "SomeCustomNode", "widgets_values": [1]}], "links": []}
    with pytest.raises(errors.ComfyrackError) as ei:
        ui_to_api(data, OBJECT_INFO)
    assert "SomeCustomNode" in str(ei.value)


def test_unrecognized_top_level_shape_fails_loud():
    with pytest.raises(errors.ComfyrackError):
        ui_to_api({"something": "else"}, OBJECT_INFO)


def test_a_legitimate_widget_value_matching_a_control_token_is_not_skipped():
    """scheduler="fixed" is a real combo choice. A value-based skip would eat it and
    shift denoise into scheduler, silently mis-assigning the rest of the node."""
    data = {"nodes": [{"id": 2, "type": "KSampler",
                       "widgets_values": [123, "randomize", 20, 4.0, "euler", "fixed", 0.9],
                       "inputs": [], "outputs": []}], "links": []}
    api = ui_to_api(data, OBJECT_INFO)
    assert api["2"]["inputs"]["scheduler"] == "fixed"
    assert api["2"]["inputs"]["denoise"] == 0.9


def test_too_few_widget_values_fails_loud_naming_the_unfilled_inputs():
    data = {"nodes": [{"id": 2, "type": "KSampler",
                       "widgets_values": [123, "randomize", 20],
                       "inputs": [], "outputs": []}], "links": []}
    with pytest.raises(errors.ComfyrackError) as ei:
        ui_to_api(data, OBJECT_INFO)
    assert "cfg" in str(ei.value)
    assert "sampler_name" in str(ei.value)


def test_is_ui_format_rejects_an_api_graph_containing_a_node_id_named_nodes():
    api_graph = {"nodes": {"class_type": "KSampler", "inputs": {}},
                 "3": {"class_type": "SaveImage", "inputs": {}}}
    assert not is_ui_format(api_graph)


def test_load_workflow_reads_an_api_format_file_without_object_info(tmp_path):
    p = tmp_path / "api.json"
    p.write_text(json.dumps({"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}),
                 encoding="utf-8")
    g = load_workflow(p)
    assert g.data["3"]["inputs"]["seed"] == 1
    assert g.data["3"]["_meta"]["title"] == "3"      # ensure_titles applied


def test_load_workflow_converts_a_ui_format_file_when_given_object_info(tmp_path):
    p = tmp_path / "ui.json"
    p.write_text((FIXTURES / "ui_basic.json").read_text(encoding="utf-8"), encoding="utf-8")
    g = load_workflow(p, object_info=OBJECT_INFO)
    assert g.data["2"]["class_type"] == "KSampler"


def test_load_workflow_on_a_ui_file_without_object_info_says_to_pass_machine(tmp_path):
    p = tmp_path / "ui.json"
    p.write_text((FIXTURES / "ui_basic.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(errors.ComfyrackError) as ei:
        load_workflow(p)
    assert "--machine" in (ei.value.help_text or "")


def test_a_widget_converted_to_a_linked_input_is_not_counted_as_positional():
    """ComfyUI's "Convert Widget to Input" drops the widget from widgets_values and adds
    a linked input slot. Counting it as positional makes a valid export look short."""
    data = {"nodes": [{"id": 2, "type": "KSampler",
                       "widgets_values": [20, 4.0, "euler", "simple", 0.9],
                       "inputs": [{"name": "seed", "link": 1}], "outputs": []},
                      {"id": 1, "type": "CheckpointLoaderSimple",
                       "widgets_values": ["anima_baseV10.safetensors"],
                       "inputs": [], "outputs": [{"name": "MODEL", "links": [1]}]}],
            "links": [[1, 1, 0, 2, 0, "INT"]]}
    api = ui_to_api(data, OBJECT_INFO)
    assert api["2"]["inputs"]["steps"] == 20        # first positional value, not seed's
    assert api["2"]["inputs"]["denoise"] == 0.9     # nothing shifted
    assert api["2"]["inputs"]["seed"] == ["1", 0]   # satisfied by the link


def test_the_unfilled_message_excludes_widgets_satisfied_by_links():
    data = {"nodes": [{"id": 2, "type": "KSampler",
                       "widgets_values": [20],
                       "inputs": [{"name": "seed", "link": 1}], "outputs": []}],
            "links": []}
    with pytest.raises(errors.ComfyrackError) as ei:
        ui_to_api(data, OBJECT_INFO)
    assert "seed" not in str(ei.value)
    assert "cfg" in str(ei.value)


# --- current frontend: widgets keep their slot in widgets_values (from a real export) ---

# TextGenerate.sampling_mode is a dynamic combo: choosing "on" adds sub-widgets.
# `mtp` is deliberately absent from this object_info: the target ComfyUI does not know
# it, so its value is dropped (the hand-converted registry/superflow/infographic.json
# omits it for the same reason).
CURRENT_OBJECT_INFO = {
    "InfographicPlatformSize": {"input": {"required": {"platform": [["A", "B"], {}]}}},
    "InfographicPrompt": {"input": {"required": {
        "content": ["STRING", {}], "layout": [["auto"], {}], "style": [["auto"], {}],
        "aspect_ratio": [["3:4"], {}], "language": [["English"], {}],
        "imagery": ["STRING", {}]}}},
    "KSampler": OBJECT_INFO["KSampler"],
    "TextGenerate": {"input": {
        "required": {
            "clip": ["CLIP"], "prompt": ["STRING", {}], "max_length": ["INT", {}],
            "sampling_mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                {"key": "off", "inputs": {"required": {}}},
                {"key": "on", "inputs": {"required": {
                    "temperature": ["FLOAT", {}], "top_k": ["INT", {}]}}},
            ]}],
            "thinking": ["BOOLEAN", {}], "use_default_template": ["BOOLEAN", {}]},
        "optional": {"image": ["IMAGE"], "system_prompt": ["STRING", {}]}}},
}


def _current():
    return json.loads((FIXTURES / "ui_current_frontend.json").read_text())


def _node(data, nid):
    return next(n for n in data["nodes"] if n["id"] == nid)


def test_current_frontend_linked_widgets_keep_their_slot():
    api = ui_to_api(_current(), CURRENT_OBJECT_INFO)
    p = api["3"]["inputs"]
    assert p["aspect_ratio"] == ["2", 0]              # linked: link wins over slot value
    assert p["language"] == "English"                 # value after the linked slot
    assert p["imagery"].startswith("syringe")
    assert p["layout"] == "auto" and p["style"] == "auto"
    t = api["13"]["inputs"]
    assert t["prompt"] == ["3", 2]
    assert t["max_length"] == 1024                    # not shifted by the linked prompt
    assert t["sampling_mode"] == "off"
    assert t["thinking"] is False and t["use_default_template"] is True
    assert t["clip"] == ["12", 0] and t["image"] == ["11", 0]


def test_current_frontend_control_token_and_widgets_after_it():
    k = ui_to_api(_current(), CURRENT_OBJECT_INFO)["9"]["inputs"]
    assert (k["seed"], k["steps"], k["cfg"]) == (42, 50, 3)
    assert (k["sampler_name"], k["scheduler"], k["denoise"]) == ("euler", "normal", 1)


def test_widget_unknown_to_the_target_is_dropped_but_keeps_its_slot():
    api = ui_to_api(_current(), CURRENT_OBJECT_INFO)
    assert "mtp" not in api["13"]["inputs"]


def test_matches_the_hand_converted_superflow_workflow_widget_values():
    known = json.loads((pathlib.Path(__file__).parent.parent
                        / "registry" / "superflow" / "infographic.json").read_text())
    api = ui_to_api(_current(), CURRENT_OBJECT_INFO)
    for nid, node in api.items():
        assert node["inputs"] == known[nid]["inputs"], nid


def test_dynamic_combo_sub_widgets_listed_as_inputs():
    data = _current()
    n = _node(data, 13)
    n["widgets_values"] = ["", 1024, "on", 0.7, 40, False, True, "auto"]
    i = next(i for i, s in enumerate(n["inputs"]) if s.get("widget", {}).get("name")
             == "sampling_mode")
    n["inputs"][i + 1:i + 1] = [
        {"name": "sampling_mode.temperature", "type": "FLOAT",
         "widget": {"name": "sampling_mode.temperature"}, "link": None},
        {"name": "sampling_mode.top_k", "type": "INT",
         "widget": {"name": "sampling_mode.top_k"}, "link": None}]
    t = ui_to_api(data, CURRENT_OBJECT_INFO)["13"]["inputs"]
    assert t["sampling_mode"] == "on"
    assert t["sampling_mode.temperature"] == 0.7 and t["sampling_mode.top_k"] == 40
    assert t["thinking"] is False and t["use_default_template"] is True


def test_dynamic_combo_sub_widgets_taken_from_node_definition_when_not_listed():
    data = _current()
    n = _node(data, 13)
    n["widgets_values"] = ["", 1024, "on", 0.7, 40, False, True, "auto"]
    t = ui_to_api(data, CURRENT_OBJECT_INFO)["13"]["inputs"]
    assert t["sampling_mode.temperature"] == 0.7 and t["sampling_mode.top_k"] == 40
    assert t["thinking"] is False and t["use_default_template"] is True


def test_current_frontend_too_few_values_fails_loud():
    data = _current()
    _node(data, 13)["widgets_values"] = ["", 1024]
    with pytest.raises(errors.ComfyrackError, match="unfilled"):
        ui_to_api(data, CURRENT_OBJECT_INFO)


# --- refusals: cases that would convert into a silently wrong graph ---

def _simple_ui(nodes, links=None, **extra):
    data = {"nodes": nodes, "links": links or []}
    data.update(extra)
    return data


def test_version_1_object_links_schema_is_refused():
    data = _simple_ui(
        [{"id": 1, "type": "CheckpointLoaderSimple",
          "widgets_values": ["anima_baseV10.safetensors"], "inputs": [], "outputs": []}],
        version=1)
    with pytest.raises(errors.ComfyrackError) as ei:
        ui_to_api(data, OBJECT_INFO)
    assert "version 1" in str(ei.value)
    assert "Export (API)" in ei.value.help_text


def test_object_link_entries_are_refused():
    data = _simple_ui(
        [{"id": 1, "type": "CheckpointLoaderSimple",
          "widgets_values": ["anima_baseV10.safetensors"], "inputs": [], "outputs": []},
         {"id": 2, "type": "KSampler",
          "widgets_values": [123, "randomize", 20, 4.0, "euler", "simple", 1.0],
          "inputs": [{"name": "model", "link": 1}], "outputs": []}],
        links=[{"id": 1, "origin": [1, 0], "target": [2, 0]}])
    with pytest.raises(errors.ComfyrackError) as ei:
        ui_to_api(data, OBJECT_INFO)
    assert "link" in str(ei.value).lower()
    assert "Export (API)" in ei.value.help_text


def test_muted_and_bypassed_nodes_are_refused_naming_the_node():
    for mode, word in ((2, "muted"), (4, "bypassed")):
        data = _simple_ui(
            [{"id": 1, "type": "CheckpointLoaderSimple",
              "widgets_values": ["anima_baseV10.safetensors"], "inputs": [], "outputs": []},
             {"id": 2, "type": "KSampler", "mode": mode,
              "widgets_values": [123, "randomize", 20, 4.0, "euler", "simple", 1.0],
              "inputs": [], "outputs": []}])
        with pytest.raises(errors.ComfyrackError) as ei:
            ui_to_api(data, OBJECT_INFO)
        assert "2" in str(ei.value) and word in str(ei.value)
        assert "Export (API)" in ei.value.help_text


def test_subgraphs_are_refused():
    data = _simple_ui(
        [{"id": 1, "type": "CheckpointLoaderSimple",
          "widgets_values": ["anima_baseV10.safetensors"], "inputs": [], "outputs": []}],
        definitions={"subgraphs": [{"id": "sg1", "nodes": []}]})
    with pytest.raises(errors.ComfyrackError) as ei:
        ui_to_api(data, OBJECT_INFO)
    assert "subgraph" in str(ei.value).lower()
    assert "Export (API)" in ei.value.help_text


def test_proxy_widgets_are_refused_naming_the_node():
    data = _simple_ui(
        [{"id": 1, "type": "CheckpointLoaderSimple",
          "widgets_values": ["anima_baseV10.safetensors"], "inputs": [], "outputs": []},
         {"id": 2, "type": "KSampler", "properties": {"proxyWidgets": ["seed"]},
          "widgets_values": [123, "randomize", 20, 4.0, "euler", "simple", 1.0],
          "inputs": [], "outputs": []}])
    with pytest.raises(errors.ComfyrackError) as ei:
        ui_to_api(data, OBJECT_INFO)
    assert "2" in str(ei.value) and "proxyWidgets" in str(ei.value)
    assert "Export (API)" in ei.value.help_text


def test_legacy_path_leftover_widget_values_are_refused():
    # No `widget` keys on the inputs -> legacy positional path. The 8th value has no
    # widget to fill; dropping it would submit a node that is not what was saved.
    data = _simple_ui(
        [{"id": 2, "type": "KSampler",
          "widgets_values": [123, "randomize", 20, 4.0, "euler", "simple", 1.0, "extra"],
          "inputs": [], "outputs": []}])
    with pytest.raises(errors.ComfyrackError) as ei:
        ui_to_api(data, OBJECT_INFO)
    assert "2" in str(ei.value) and "extra" in str(ei.value)
    assert "Export (API)" in ei.value.help_text


def test_legacy_load_image_upload_button_value_is_not_a_leftover():
    """LoadImage declares image_upload, so older exports carry a trailing "image" value
    for the UI-only upload button. That is not a leftover and must not be refused."""
    import json
    from pathlib import Path
    oi = json.loads((Path(__file__).parent / "fixtures" / "object_info_0_37.json")
                    .read_text(encoding="utf-8"))
    ui = {"nodes": [{"id": 1, "type": "LoadImage", "widgets_values": ["example.png", "image"],
                     "inputs": [], "outputs": []}], "links": []}
    api = ui_to_api(ui, oi)
    assert api["1"]["inputs"] == {"image": "example.png"}
