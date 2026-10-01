from comfyrack import out


def test_table_pads_columns_and_includes_all_rows():
    rows = [("anima/character-scene", "full-scene render"), ("qwen/edit-scene", "resize")]
    result = out.table(rows, headers=("name", "description"))
    lines = result.splitlines()
    assert "anima/character-scene" in lines[1]
    assert "qwen/edit-scene" in lines[2]
    # Column alignment: description starts at the same offset on every row.
    # Both descriptions are chosen NOT to occur inside their row's first column --
    # otherwise .index() finds the substring there and the assertion tests nothing.
    assert lines[1].index("full-scene render") == lines[2].index("resize")


def test_empty_state_is_definitive_and_names_the_context():
    result = out.empty("workflows", "in family 'nope'")
    assert result == "workflows: 0 found in family 'nope'"


def test_render_json_mode_emits_parseable_json():
    import json
    result = out.render({"count": 2}, as_json=True)
    assert json.loads(result) == {"count": 2}


def test_one_line_keeps_the_first_sentence_and_cuts_long_ones():
    assert out.one_line("Text to image. Needs a LoRA.\nSecond line.") == "Text to image"
    assert out.one_line("x" * 200, width=20) == "x" * 17 + "..."
