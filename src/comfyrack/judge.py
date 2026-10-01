"""VL judging of a rendered sheet against its shot list.

The rubric is generated from the shot list rather than written by hand, so what is
judged is exactly what was asked for. Provenance is what makes this
meaningful: without a record of the resolved parameters, a verdict cannot be tied
back to the run that produced it."""
from pathlib import Path

from .errors import UsageError

# Where a VL workflow takes its instruction text, most specific first. A judge
# graph feeds a vision-language node, whose instruction flag is `custom_prompt`
# on this project's Qwen3-VL workflow; `positive_text` is the canonical spelling
# for a diffusion workflow's prompt and is accepted as a fallback.
PROMPT_FLAGS = ("custom_prompt", "positive_text")

HEADER = """You are checking a contact sheet of rendered illustrations against the shot
list they were rendered from. The sheet tiles the shots in order, left to right, top to
bottom.

For EACH shot, report:
  match     - does the image show the described action and location?
  headcount - how many people are visible, and does that match the expected count?
  defects   - malformed hands or faces, duplicated limbs, extra or missing people,
              text artifacts, style breaks

Answer per shot on one line: `<shot name>: <ok|problem> - <one sentence>`.
Do not describe what is correct at length. Report problems.

SHOT LIST:
"""

PEOPLE_COUNT = {"solo": "exactly 1 person", "two_person": "exactly 2 people",
                "three_person": "exactly 3 people", "group": "3 or more people"}


def build_rubric(shot_list, only: str | None = None) -> str:
    lines = [HEADER]
    for shot in shot_list.shots:
        if only and shot.name != only:
            continue
        expected = PEOPLE_COUNT.get(shot.people, shot.people or "unspecified")
        lines.append(
            f"- {shot.name}: {shot.action_text or '(no action given)'} | "
            f"location: {shot.location_text or '(unspecified)'} | "
            f"expected headcount: {expected}")
    return "\n".join(lines)


def prompt_flag(manifest) -> str:
    """This manifest's own key for the instruction text.

    Read off the manifest instead of assumed, because assuming is what broke
    this: `run()` submitted an override literally named `prompt`, which is a
    canonical ALIAS of `positive_text` -- a flag the builtin `vl-judge` manifest
    does not declare (it declares `custom_prompt`). `resolve_values()` used to
    drop an override it did not recognise without a word, so the generated
    rubric reached nothing and the VL model judged every sheet against whatever
    prompt the workflow JSON happened to bake in. The verdict came back
    confident and about the wrong question."""
    for name in PROMPT_FLAGS:
        key = manifest.resolve_key(name)
        if key in manifest.flags:
            return key
    raise UsageError(
        f"workflow {manifest.name!r} declares no prompt flag, so the rubric has "
        f"nowhere to go",
        help_text=f"comfyrack describe {manifest.name}  # a judge workflow needs "
                  f"one of: {', '.join(PROMPT_FLAGS)}",
    )


def run(rack, sheet_path, shot_list, workflow: str = "vl-judge", machine=None,
        only=None) -> str:
    """Run the VL judge workflow over a sheet. The workflow must declare
    `output: text`, an `image` flag, and one of `PROMPT_FLAGS`."""
    manifest = rack.describe(workflow)
    result = rack.run(workflow,
                      overrides={"image": str(Path(sheet_path)),
                                 prompt_flag(manifest): build_rubric(
                                     shot_list, only=only)},
                      machine=machine)
    return result.text or ""
