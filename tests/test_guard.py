import pytest
import yaml
from comfyrack.config import Config
from comfyrack.lexicon import Lexicon
from comfyrack.lora_meta import LoraMeta
from comfyrack import guard, errors


def make_lex(tmp_path):
    (tmp_path / ".comfyrack").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".comfyrack" / "config.toml").write_text(
        '[paths]\nlexicon = "lex"\n', encoding="utf-8")
    (tmp_path / "lex").mkdir(exist_ok=True)
    (tmp_path / "lex" / "char_a.yaml").write_text(yaml.safe_dump(
        {"term": "char_a", "kind": "character",
         "renders": {"illustrious": {"trigger": "trig_a", "tags": "1girl"}}}),
        encoding="utf-8")
    cfg = Config.load(start_dir=tmp_path, user_config=tmp_path / "none.toml")
    return Lexicon(cfg)


MIRA_LORA = LoraMeta(
    filename="your_character_lora.safetensors", trigger="trig_a",
    base_model_family="anima",
    tags=["trig_a", "1girl", "young teenager", "(straight blonde hair:1.2)",
          "fair skin", "blue-green eyes"])


def test_a_known_term_produces_no_finding(tmp_path):
    assert guard.check_terms("{char_a} dancing", make_lex(tmp_path)) == []


def test_an_unresolved_term_is_an_error_severity_finding(tmp_path):
    findings = guard.check_terms("{harborview} at night", make_lex(tmp_path))
    assert findings[0].severity == "error"
    assert findings[0].term == "harborview"


def test_a_capitalised_proper_noun_outside_the_lexicon_is_an_error(tmp_path):
    findings = guard.check_terms("standing on the Harborview Coast", make_lex(tmp_path))
    assert any(f.severity == "error" and f.term == "Harborview" for f in findings)


def test_a_sentence_initial_capital_is_not_treated_as_a_proper_noun(tmp_path):
    assert guard.check_terms("Standing on a dock", make_lex(tmp_path)) == []


def test_a_capitalised_word_that_is_a_known_term_is_accepted(tmp_path):
    assert guard.check_terms("a portrait of Char_a dancing", make_lex(tmp_path)) == []


def test_a_leading_proper_noun_in_a_tag_list_is_flagged(tmp_path):
    # A comma-separated tag list is the dominant prompt shape in this pipeline and
    # position 0 carries no grammatical meaning there, so a capitalised unknown at
    # the head of one is exactly the confabulation this guard exists to catch.
    findings = guard.check_terms("Harborview, night, moonlight", make_lex(tmp_path))
    assert any(f.severity == "error" and f.term == "Harborview" for f in findings)


def test_a_leading_known_term_in_a_tag_list_is_accepted(tmp_path):
    assert guard.check_terms("Char_a, night, moonlight", make_lex(tmp_path)) == []


def test_a_weighted_tag_later_in_the_list_does_not_hide_a_leading_proper_noun(tmp_path):
    # (straight blonde hair:1.2) is this pipeline's canonical weighted-tag syntax and
    # it contains a period. Reading that period as sentence punctuation classifies
    # every weighted tag list as prose, which switches position-0 checking back off.
    findings = guard.check_terms(
        "Harborview, (straight blonde hair:1.2), night", make_lex(tmp_path))
    assert any(f.severity == "error" and f.term == "Harborview" for f in findings)


def test_a_short_sentence_head_is_still_read_as_prose(tmp_path):
    # Two words, so the length test alone would call this a tag; the terminal stop
    # is the only thing marking it as a sentence. Pins that scoping the punctuation
    # vote to the head segment did not delete the vote.
    assert guard.check_terms("Harborview waves.", make_lex(tmp_path)) == []


def test_a_leading_capital_on_a_prose_clause_is_still_sentence_case(tmp_path):
    # Same leading position as the tag list above, but the head segment reads as a
    # clause rather than a tag, so the capital is sentence case and not a candidate.
    assert guard.check_terms(
        "Standing on a dock, waiting for the tide", make_lex(tmp_path)) == []


def test_a_description_contradicting_the_trained_hair_colour_is_an_error():
    findings = guard.check_contradiction("trig_a, dark brown hair, mid-dance", MIRA_LORA)
    assert findings[0].severity == "error"
    assert "blonde" in findings[0].detail


def test_a_description_matching_the_trained_vocabulary_produces_no_finding():
    assert guard.check_contradiction(
        "trig_a, (straight blonde hair:1.2), fair skin", MIRA_LORA) == []


def test_a_contradicting_skin_tone_is_detected():
    findings = guard.check_contradiction("trig_a, brown skin", MIRA_LORA)
    assert any("fair skin" in f.detail for f in findings)


def test_an_attribute_the_lora_never_trained_is_not_a_contradiction():
    assert guard.check_contradiction("trig_a, holding a lantern", MIRA_LORA) == []


def test_enforce_raises_on_an_error_finding_listing_the_fix(tmp_path):
    findings = guard.check_terms("{harborview}", make_lex(tmp_path))
    with pytest.raises(errors.UsageError) as ei:
        guard.enforce(findings)
    assert ei.value.exit_code == 2
    assert "comfyrack lex add" in ei.value.help_text


def test_allow_unknown_downgrades_errors_to_warnings(tmp_path):
    findings = guard.check_terms("{harborview}", make_lex(tmp_path))
    warnings = guard.enforce(findings, allow_unknown=True)
    assert any("harborview" in w for w in warnings)


def test_allow_unknown_does_not_downgrade_a_lora_contradiction():
    # --allow-unknown relaxes the lexicon check. A LoRA's trained tags are the most
    # authoritative oracle in the package, so a contradiction against them must stay
    # a hard failure no matter what the flag says.
    findings = guard.check_contradiction("trig_a, dark brown hair", MIRA_LORA)
    with pytest.raises(errors.UsageError) as ei:
        guard.enforce(findings, allow_unknown=True)
    assert "dark brown hair" in str(ei.value)


def test_allow_unknown_downgrades_only_the_unknown_term_beside_a_contradiction(tmp_path):
    findings = (guard.check_terms("{harborview}", make_lex(tmp_path))
                + guard.check_contradiction("trig_a, dark brown hair", MIRA_LORA))
    with pytest.raises(errors.UsageError) as ei:
        guard.enforce(findings, allow_unknown=True)
    assert "dark brown hair" in str(ei.value)
    assert "harborview" not in str(ei.value)


def test_a_contradiction_help_line_does_not_offer_allow_unknown_as_the_fix():
    findings = guard.check_contradiction("trig_a, dark brown hair", MIRA_LORA)
    with pytest.raises(errors.UsageError) as ei:
        guard.enforce(findings)
    assert "--allow-unknown" not in ei.value.help_text


def test_a_free_floating_tag_with_no_authoritative_source_is_a_warning(tmp_path):
    findings = guard.check_terms("{char_a}, weaving flowing_water_magic",
                                 make_lex(tmp_path))
    assert [(f.severity, f.term) for f in findings] == \
        [("warning", "flowing_water_magic")]


def test_a_free_floating_tag_warning_does_not_raise(tmp_path):
    findings = guard.check("{char_a}, weaving flowing_water_magic",
                           make_lex(tmp_path))
    assert guard.enforce(findings) == [
        w for w in guard.enforce(findings) if "flowing_water_magic" in w]
    assert len(guard.enforce(findings)) == 1


def test_a_free_floating_tag_the_lora_was_trained_on_is_not_flagged(tmp_path):
    trained = LoraMeta(filename="your_character_lora.safetensors", trigger="trig_a",
                       base_model_family="anima",
                       tags=["trig_a", "looking_at_viewer"])
    assert guard.check_terms("{char_a}, looking_at_viewer", make_lex(tmp_path),
                             lora_meta=trained) == []


def test_enforce_returns_warnings_without_raising(tmp_path):
    findings = [guard.Finding("warning", "flowing_water_magic", "not a known tag")]
    assert guard.enforce(findings) == \
        ["warning: flowing_water_magic -- not a known tag"]


def test_enforce_with_no_findings_returns_an_empty_list():
    assert guard.enforce([]) == []
