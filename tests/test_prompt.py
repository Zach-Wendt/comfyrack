import random
import pytest
import yaml
from comfyrack.config import Config
from comfyrack.lexicon import Lexicon
from comfyrack import prompt, errors


def make_lex(tmp_path, entries):
    (tmp_path / ".comfyrack").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[paths]\nlexicon = "lex"\n', encoding="utf-8")
    (tmp_path / "lex").mkdir(exist_ok=True)
    for term, data in entries.items():
        (tmp_path / "lex" / f"{term}.yaml").write_text(
            yaml.safe_dump({"term": term, **data}, sort_keys=False), encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    return Lexicon(cfg)


ENTRIES = {
    "char_a": {"kind": "character", "renders": {
        "illustrious": {"trigger": "trig_a", "tags": "1girl, (straight blonde hair:1.2)"},
        "flux": {"text": "a young girl with straight blonde hair"}}},
    "char_c": {"kind": "character", "renders": {
        "illustrious": {"trigger": "ly7a", "tags": "1girl, dark hair"}}},
    "harborview-coast": {"kind": "location", "renders": {
        "illustrious": {"tags": "ocean, wooden dock, overcast sky"}}},
}


def test_substitute_replaces_a_braced_term_with_its_render(tmp_path):
    lex = make_lex(tmp_path, ENTRIES)
    out = prompt.substitute("{char_c} standing to the side", lex, "illustrious")
    assert out == "ly7a, 1girl, dark hair standing to the side"


def test_substitute_handles_several_terms_in_one_string(tmp_path):
    lex = make_lex(tmp_path, ENTRIES)
    out = prompt.substitute("{char_a} and {char_c}", lex, "illustrious")
    assert "trig_a" in out and "ly7a" in out


def test_substitute_on_an_unknown_term_raises_with_the_add_command(tmp_path):
    lex = make_lex(tmp_path, ENTRIES)
    with pytest.raises(errors.NotFoundError) as ei:
        prompt.substitute("{harborview}", lex, "illustrious")
    assert "comfyrack lex add" in ei.value.help_text


def test_text_with_no_placeholders_passes_through_unchanged(tmp_path):
    lex = make_lex(tmp_path, ENTRIES)
    assert prompt.substitute("mid-dance, arms raised", lex, "illustrious") == \
        "mid-dance, arms raised"


def test_wildcard_choices_expand_deterministically_under_a_seeded_rng():
    rng = random.Random(1234)
    out = prompt.expand_wildcards("a {red|blue|green} tunic", rng=rng)
    assert out in ("a red tunic", "a blue tunic", "a green tunic")
    assert prompt.expand_wildcards("a {red|blue|green} tunic", rng=random.Random(1234)) == out


def test_wildcard_file_reference_draws_from_the_wildcards_directory(tmp_path):
    wd = tmp_path / "wildcards"
    wd.mkdir()
    (wd / "weather.txt").write_text("overcast\nfoggy\n", encoding="utf-8")
    out = prompt.expand_wildcards("__weather__ light", rng=random.Random(0), wildcards_dir=wd)
    assert out in ("overcast light", "foggy light")
    # Determinism is the entire justification for expanding wildcards client-side
    # (rather than in a node, where the final prompt would be unknown at submit
    # time). Pin it the same way the choice-wildcard test above does: two runs
    # seeded identically must produce the identical draw, not just a member of
    # the same small set.
    assert prompt.expand_wildcards("__weather__ light", rng=random.Random(0),
                                   wildcards_dir=wd) == out


def test_wildcard_file_draws_via_the_caller_supplied_rng_not_the_global_random_module(
        tmp_path, monkeypatch):
    # The determinism test above (same seed -> same draw) is necessary but, with only
    # two options and a fresh process each run, not sufficient: `random.choice` in
    # place of `rng.choice` still has a real chance of coincidentally reproducing the
    # same output. Prove it directly instead: the global random module's own
    # `choice()` must never be called by this code path at all.
    wd = tmp_path / "wildcards"
    wd.mkdir()
    (wd / "weather.txt").write_text("overcast\nfoggy\n", encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("expand_wildcards drew a file-wildcard option via the "
                              "global random module instead of the caller's rng")
    monkeypatch.setattr(random, "choice", boom)
    out = prompt.expand_wildcards("__weather__ light", rng=random.Random(0), wildcards_dir=wd)
    assert out in ("overcast light", "foggy light")


def test_a_missing_wildcard_file_is_a_usage_error_naming_the_directory(tmp_path):
    with pytest.raises(errors.UsageError) as ei:
        prompt.expand_wildcards("__nope__", rng=random.Random(0), wildcards_dir=tmp_path)
    assert str(tmp_path) in ei.value.help_text
    # help_text is "expected {path} under {wildcards_dir}" -- path already contains
    # wildcards_dir as a prefix, so a bare `str(tmp_path) in help_text` check above
    # passes even if "under {wildcards_dir}" is dropped entirely. Pin that the
    # directory is named as its own trailing clause, not just present inside path.
    assert ei.value.help_text.endswith(str(tmp_path))


def test_lexicon_braces_are_not_mistaken_for_wildcard_choices(tmp_path):
    # {char_c} has no pipe, so it is a lexicon term and must survive wildcard expansion.
    assert prompt.expand_wildcards("{char_c} waits", rng=random.Random(0)) == "{char_c} waits"


def test_a_piped_group_is_not_treated_as_a_lexicon_term(tmp_path):
    # Converse of test_lexicon_braces_are_not_mistaken_for_wildcard_choices: a piped
    # group reaching substitute() (i.e. NOT expanded first, as it should have been by
    # expand_wildcards) must be rejected as an unresolved wildcard -- NOT looked up as
    # a lexicon term named "red|blue". If TERM_RE's char class ever grows a pipe,
    # substitute() would instead call lexicon.render("red|blue", ...), which raises
    # NotFoundError (unknown term), not UsageError -- so this discriminates the two
    # code paths, not just the two exception types by accident.
    lex = make_lex(tmp_path, ENTRIES)
    with pytest.raises(errors.UsageError) as ei:
        prompt.substitute("{red|blue} tunic", lex, "illustrious")
    assert "red|blue" in ei.value.message


def test_substitute_raises_on_a_space_inside_braces(tmp_path):
    # "{harborview coast}" (a space) matches neither TERM_RE (no spaces allowed) nor
    # CHOICE_RE's pipe requirement -- it is not a valid lexicon term OR a wildcard
    # choice, and must not reach the model as literal punctuation.
    lex = make_lex(tmp_path, ENTRIES)
    with pytest.raises(errors.UsageError) as ei:
        prompt.substitute("{harborview coast} at dusk", lex, "illustrious")
    assert "harborview coast" in ei.value.message


def test_substitute_raises_on_an_unclosed_brace(tmp_path):
    lex = make_lex(tmp_path, ENTRIES)
    with pytest.raises(errors.UsageError) as ei:
        prompt.substitute("{unclosed and more text", lex, "illustrious")
    assert "unclosed" in ei.value.message


def test_substitute_does_not_trip_on_a_weighted_lexicon_token(tmp_path):
    # (straight blonde hair:1.2) uses parens and a colon, not braces -- the new
    # residual-brace check must not confuse this for an unresolved wildcard, and the
    # weighted token must survive verbatim (no re-spacing, no re-casing).
    lex = make_lex(tmp_path, ENTRIES)
    out = prompt.substitute("{char_a} posing", lex, "illustrious")
    assert "(straight blonde hair:1.2)" in out


def test_booru_assembly_joins_with_commas_and_leads_with_the_subject(tmp_path):
    lex = make_lex(tmp_path, ENTRIES)
    out = prompt.assemble(lex, "illustrious", "booru", subject="char_a",
                          location="harborview-coast", action="mid-dance, arms raised",
                          style="cel shaded")
    assert out.startswith("trig_a, 1girl, (straight blonde hair:1.2)")
    assert "ocean, wooden dock, overcast sky" in out
    assert "mid-dance, arms raised" in out
    assert out.endswith("cel shaded")


def test_natural_assembly_joins_with_sentences(tmp_path):
    lex = make_lex(tmp_path, ENTRIES)
    out = prompt.assemble(lex, "flux", "natural", subject="char_a",
                          action="reaching toward a cresting wave")
    assert out == "a young girl with straight blonde hair. reaching toward a cresting wave"


def test_assemble_does_not_expand_wildcards_and_fails_loudly_if_the_caller_skips_it(tmp_path):
    # assemble() intentionally does not call expand_wildcards itself (callers, e.g.
    # Rack.assemble_prompt, are expected to expand wildcards in `action` first). If a
    # caller forgets, an unexpanded {a|b} choice in action must not reach the model as
    # literal text -- it must raise, via substitute()'s residual-brace check.
    lex = make_lex(tmp_path, ENTRIES)
    with pytest.raises(errors.UsageError) as ei:
        prompt.assemble(lex, "illustrious", "booru", subject="char_a",
                        action="{red|blue} tunic, mid-dance")
    assert "red|blue" in ei.value.message


def test_assembly_substitutes_terms_inside_the_action_slot(tmp_path):
    lex = make_lex(tmp_path, ENTRIES)
    out = prompt.assemble(lex, "illustrious", "booru", subject="char_a",
                          action="{char_c} watching closely")
    assert "ly7a" in out


def test_assembly_with_no_subject_omits_it_rather_than_emitting_an_empty_segment(tmp_path):
    lex = make_lex(tmp_path, ENTRIES)
    out = prompt.assemble(lex, "illustrious", "booru", action="empty room")
    assert out == "empty room"


def test_assemble_raises_on_an_unknown_dialect_instead_of_silently_using_booru(tmp_path):
    # JOIN.get(dialect, ", ") used to silently fall back to the booru comma-joiner for
    # any unrecognized dialect (e.g. a typo'd "naturl"), producing a clean-looking but
    # wrong prompt. An unknown dialect must be loud.
    lex = make_lex(tmp_path, ENTRIES)
    with pytest.raises(errors.UsageError) as ei:
        prompt.assemble(lex, "illustrious", "naturl", action="empty room")
    assert "naturl" in ei.value.message
    assert "booru" in ei.value.help_text and "natural" in ei.value.help_text
