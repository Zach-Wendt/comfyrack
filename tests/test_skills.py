import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"
EXPECTED = ["comfyrack", "comfyrack-prompting", "comfyrack-characters", "comfyrack-batch"]

# A ring 2 skill that still points at a project's own scripts is not portable, which
# is the whole reason these moved out of the sample project.
FORBIDDEN = ["project/scripts", "render_chapter.py", "navigate_room.py",
             "shot.py", "sample_project"]


@pytest.mark.parametrize("name", EXPECTED)
def test_each_expected_skill_exists(name):
    assert (SKILLS / name / "SKILL.md").is_file()


@pytest.mark.parametrize("name", EXPECTED)
def test_each_skill_has_name_and_description_frontmatter(name):
    text = (SKILLS / name / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---")
    head = text.split("---")[1]
    assert "name:" in head and "description:" in head


@pytest.mark.parametrize("name", EXPECTED)
def test_no_skill_references_a_project_script_path(name):
    text = (SKILLS / name / "SKILL.md").read_text(encoding="utf-8")
    for token in FORBIDDEN:
        assert token not in text, f"{name} references {token}"


def test_the_prompting_skill_states_the_guard_and_both_dialects():
    text = (SKILLS / "comfyrack-prompting" / "SKILL.md").read_text(encoding="utf-8")
    assert "booru" in text and "natural" in text
    assert "comfyrack lex" in text
    assert "prompt preview" in text


def test_the_prompting_skill_forbids_writing_positive_text_directly():
    text = (SKILLS / "comfyrack-prompting" / "SKILL.md").read_text(encoding="utf-8")
    assert "positive_text" in text


def test_the_batch_skill_documents_validate_before_render():
    text = (SKILLS / "comfyrack-batch" / "SKILL.md").read_text(encoding="utf-8")
    assert "--validate" in text and "--dry-run" in text


def test_the_batch_skill_documents_every_status_a_batch_can_report():
    """The doc listed 3 of the 4 real statuses. The missing one, `unsupported`,
    is hit by this project's own corpus (the sample shot list s05/s06/s09), so an agent reading
    the skill met a status the skill had never mentioned."""
    text = (SKILLS / "comfyrack-batch" / "SKILL.md").read_text(encoding="utf-8")
    for status in ("`ok`", "`failed`", "`unroutable`", "`unsupported`"):
        assert status in text, f"comfyrack-batch does not document {status}"
    # ...and says what the previously-undocumented one actually means.
    assert "shot_builder" in text


def test_the_batch_skill_documents_the_exit_codes():
    """A fully-failed batch is detectable from the exit code alone now, which is
    only useful to a scripted caller if the doc says so."""
    text = (SKILLS / "comfyrack-batch" / "SKILL.md").read_text(encoding="utf-8")
    assert "Exit codes" in text
    assert "--machine" in text
