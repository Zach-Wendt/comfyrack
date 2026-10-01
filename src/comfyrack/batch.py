"""Shot-list batch execution.

Walk a shot list, route each shot to a machine that holds its models, run it, and
build a compare sheet."""
from pathlib import Path

from . import dispatch
from .errors import UsageError

# The manifest flags that carry the identity LoRA's weight, in the canonical
# spelling registry.CANONICAL_ALIASES defines. Both are set from ONE authored
# `id_strength`: a shot list says how strongly this character's identity should
# render, not how to split that between the UNET and the text encoder -- the
# anima workflows drive the ID LoRA node's strength_model and strength_clip from
# the same number.
ID_STRENGTH_FLAGS = ("id_strength_model", "id_strength_clip")

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _characters_in(shot) -> list:
    """Every character name a shot needs, whether from its own `character:` field
    or from a `via: shot_builder` shot's composite refs.

    `shot.composite` is ALWAYS a dict (never a list -- `_normalize_composite` in
    shots.py guarantees this, raising `UsageError` if the YAML gives it a list),
    shaped {"seed": ..., "refs": [...], "extra_positive": ..., "extra_negative":
    ...}. `refs` is the list of per-character composite entries, each shaped like
    {"character": "char_b", "clause": "..."} in a shot list."""
    names = [shot.character] if shot.character else []
    for entry in shot.composite.get("refs", []):
        name = entry.get("character")
        if name:
            names.append(name)
    return names


def shot_requirements(shot, profile, lexicon=None) -> dict:
    """Model files this shot needs on whatever machine runs it.

    An unknown character contributes nothing rather than raising: a shot naming a
    character with no profile is a shot-list problem for validate() to report, not a
    reason to abort routing for the whole batch.

    A lexicon render's `lora` counts as a requirement too. A trigger token means
    something ONLY while its own LoRA is loaded -- `trig_a` is a trigger for one
    specific LoRA, and against any other model, or with that LoRA absent, it is a
    token the model silently ignores. Routing the shot to a machine that lacks the
    LoRA emits the trigger into a prompt where it grounds nothing, which is exactly
    the plausible-and-wrong render the guard exists to prevent. Lineage keying alone
    does not close this: two families can share a lineage and only one hold the file."""
    known = profile.characters()
    loras = []

    def need(value):
        if value and value not in loras:
            loras.append(value)

    for name in _characters_in(shot):
        char = known.get(name)
        if char is None:
            continue
        need(char.id_lora)
        need(char.style_lora)

    if lexicon is not None:
        for name in _characters_in(shot):
            entry = lexicon.all().get(str(name).lower())
            if entry is None:
                continue
            for block in entry.renders.values():
                if isinstance(block, dict):
                    need(block.get("lora"))

    return {"loras": loras}


def build_jobs(shot_list, profile, only: str | None = None, lexicon=None) -> list:
    jobs = []
    for shot in shot_list.shots:
        if only and shot.name != only:
            continue
        jobs.append(dispatch.Job(
            shot_name=shot.name,
            requirements=shot_requirements(shot, profile, lexicon=lexicon),
            fn=shot))
    return jobs


def _resolve_terms(text: str, shot_list) -> str:
    """Resolve {term} placeholders from the shot list's own `characters:` prose.

    Deliberately NOT the global lexicon. Two resolvers exist for this syntax and the
    shot list wins: its text is authored for THIS sequence and most real files
    depend on it, whereas a lexicon render is keyed by model lineage, which this
    function does not have and should not guess. `comfyrack prompt preview` is where
    lexicon-backed assembly happens.

    An unknown term is left as a literal brace rather than deleted. Both outcomes are
    wrong, but a visible `{char_c}` in the prompt and the provenance record is findable,
    whereas a silently vanished character is not."""
    from .prompt import TERM_RE

    chars = shot_list.characters or {}

    def sub(match):
        entry = chars.get(match.group(1)) or chars.get(match.group(1).lower())
        if isinstance(entry, dict) and entry.get("text"):
            return str(entry["text"])
        return match.group(0)

    return TERM_RE.sub(sub, text or "")


def id_strength_flags(manifest) -> list:
    """This manifest's own keys for the identity LoRA's weight.

    Resolved through `manifest.resolve_key` rather than matched literally, so a
    workflow that keys the concept under one of its declared aliases still gets
    the value. A workflow with no identity LoRA at all (an upscale, a harmonize
    pass) declares neither and gets an empty list."""
    keys = []
    for name in ID_STRENGTH_FLAGS:
        key = manifest.resolve_key(name)
        if key in manifest.flags and key not in keys:
            keys.append(key)
    return keys


def shot_id_strength(shot, shot_list, profile):
    """The identity-LoRA weight this shot renders at, or None when nothing
    declares one.

    The shot's own `id_strength` wins: it is the most specific authored value
    there is, written for this exact frame. Only when the shot is silent does
    this fall back to `profile.resolve_id_strength`, which walks the character's
    own per-sequence / close-shot / default / locked-strength chain and RAISES
    rather than guessing when no tier supplies one. Guessing is the failure this
    whole package exists to prevent -- the render comes back looking finished,
    with the wrong face, and nothing reports a problem.

    An unknown character contributes None rather than raising, matching
    `shot_requirements`: a shot naming a character with no profile is a shot-list
    problem for validate() to report, not a reason to abort the batch."""
    if shot.id_strength is not None:
        return shot.id_strength
    name = shot.character
    if not name or name not in profile.characters():
        return None
    return profile.resolve_id_strength(name, sequence=shot_list.sequence)


def run_batch(rack, shot_list, workflow: str, machines=None, only=None,
              max_retries: int = 1, recipe=None, on_progress=None) -> dict:
    """Render every shot.

    Returns `{"results": [...], "index_errors": {...}}`:

    - `results` is one dict per RENDERABLE shot (dispatch.run_queue's per-shot
      shape: shot/status/machine/detail/..., with one extra status this
      function adds -- see the `via` note below), in the shot list's own
      order. A single shot failing never stops the rest of the batch.
    - `index_errors` maps a machine name to why `dispatch.index()` could not
      probe it (e.g. an offline machine), as `str(exc)`. This is DATA, not a
      `warnings.warn()` side channel: this codebase's established pattern for
      a diagnostic a caller must be able to act on is to return it and let
      the caller decide where it goes (`guard.enforce()` returns warning
      lines for its caller to print; `Rack.assemble_prompt()` returns
      `(text, warnings)`) -- a warning can be filtered by `-W ignore`, hidden
      by the default dedup filter on a second occurrence, or promoted to
      fatal by `-W error`, none of which a caller of a batch render should be
      able to do by accident. A machine already missing from `capabilities`
      because it's in `index_errors` still shows up per-shot too, via
      run_queue's own `unroutable`/rerouted reporting -- `index_errors` is the
      one additional fact routing alone does not carry: that THIS machine
      specifically could not be probed at all.

    A `via: shot_builder` shot (a multi-character composite) never reaches
    dispatch.run_queue and is reported directly with `status: "unsupported"`.
    run_batch does not implement the Qwen-Image-Edit-2511 compositing pipeline
    a `base` + `composite.refs` shot needs -- rendering just the base
    text-conditioned pass would produce a plausible-looking single-character
    image silently missing its second lead, which is exactly the
    plausible-and-wrong failure class this project exists to prevent. Failing
    loud with an explicit, distinct status beats a quiet wrong render.

    `on_progress` takes one preformatted line per progress event, already
    labelled with the shot it belongs to, and is silent by default. The CLI
    passes one that writes to STDERR; stdout carries the machine-readable
    result and must stay parseable."""
    machine_map = ({name: rack.config.machine_url(name) for name in machines}
                   if machines else {"default": rack.config.machine_url(None)})
    capabilities, index_errors = dispatch.index(
        machine_map, lambda url: rack.client_for_url(url))
    # rack.lexicon so a render's trigger and its LoRA cannot be routed apart.
    jobs = build_jobs(shot_list, rack.profile, only=only, lexicon=rack.lexicon)

    renderable_jobs = [j for j in jobs if not j.fn.via]

    # Which flag(s) this workflow carries the identity weight under, resolved
    # ONCE rather than per shot. A workflow that declares none of them cannot
    # apply an authored `id_strength` at all: the ID LoRA node would load at
    # whatever the workflow JSON bakes in (0.0 in the anima graphs,
    # because the strength is expected to arrive per shot). That is a finished-
    # looking render with zero identity and a `9/9 ok` report, so it fails LOUD
    # and up front, before any GPU time, rather than per shot after it.
    manifest = rack.describe(workflow)
    id_flags = id_strength_flags(manifest)
    if not id_flags:
        declaring = [j.shot_name for j in renderable_jobs
                     if j.fn.id_strength is not None]
        if declaring:
            raise UsageError(
                f"workflow {workflow!r} declares no identity-LoRA strength flag, "
                f"but {len(declaring)} shot(s) set `id_strength`: "
                f"{', '.join(declaring)}",
                help_text=f"comfyrack describe {workflow}  # this workflow has no "
                          f"identity LoRA to weight; render these shots with a "
                          f"workflow that does, or drop `id_strength` from them",
            )

    unsupported = [{
        "shot": j.shot_name, "status": "unsupported", "machine": None,
        "detail": (f"`via: {j.fn.via}` composite shot -- run_batch has no "
                  "compositing pipeline wired in; rendering only the base "
                  "would silently drop the composited character(s)"),
    } for j in jobs if j.fn.via]

    def runner(job, machine):
        shot = job.fn
        overrides = {}
        if shot.seed is not None:
            overrides["seed"] = shot.seed
        if shot.positive_prefix or shot.action_text or shot.location_text:
            # {term} placeholders are resolved from the shot list's own
            # `characters:` prose via _resolve_terms() before submit -- see
            # its docstring. That is the ONLY resolution this layer performs:
            # _resolve_terms() deliberately does NOT consult the global
            # lexicon, and no guard.check_terms() call happens anywhere in
            # run_batch (the guard only runs inside Rack.assemble_prompt(),
            # which run_batch never calls). A term neither the shot list nor
            # this function knows about therefore ships to ComfyUI as a
            # literal `{brace}` -- visible in the submitted prompt and the
            # provenance record, but NOT caught or refused by anything at
            # this layer. Do not assume a lexicon fallback or a guard check
            # is happening here; neither is.
            action = _resolve_terms(shot.action_text, shot_list)
            overrides["positive_text"] = ", ".join(
                p for p in (shot.positive_prefix, action, shot.location_text) if p)
        if shot_list.extra_negative or shot.extra_negative:
            overrides["negative_text"] = ", ".join(
                p for p in (shot_list.extra_negative, shot.extra_negative) if p)
        # The identity LoRA's weight. Without this the shot's authored
        # `id_strength: 0.45` reached nothing: `resolve_values` fell through to
        # the value baked into the workflow JSON -- 0.0 -- and every real sequence
        # rendered its lead with the identity LoRA switched off, finished-looking
        # and faceless, reported as `ok`.
        if id_flags:
            strength = shot_id_strength(shot, shot_list, rack.profile)
            if strength is not None:
                for key in id_flags:
                    overrides[key] = strength
        # Route by the URL machine_map already resolved for this machine key,
        # not by re-resolving `machine` as a `[machines]` config NAME.
        # machine_map's own no-`--machines` default path resolves its one
        # entry via `Config.machine_url(None)` (env var, then a *possible*
        # `[machines] default` entry, then the builtin default) -- a
        # different, more permissive branch than `Config.machine_url("default")`
        # (which requires an actual `[machines] default = ...` entry and
        # raises UsageError otherwise). Passing `machine="default"` through to
        # `rack.run()` re-resolves via that second, stricter branch and broke
        # every shot whenever no `[machines]` table happened to be configured
        # -- the common case. `url=` sends the exact URL already resolved and
        # indexed, so it cannot land on a different branch.
        # Label each line with its shot: a batch is N renders in a row, and an
        # unlabelled "step 12/20" halfway down a 9-shot run says nothing about
        # which shot is 60% done.
        report = None
        if on_progress is not None:
            report = lambda line, name=shot.name: on_progress(f"{name}: {line}")  # noqa: E731
        result = rack.run(workflow, recipe=recipe, overrides=overrides,
                          url=machine_map[machine], on_progress=report)
        return {"paths": [str(p) for p in result.paths]}

    results = dispatch.run_queue(renderable_jobs, capabilities, runner,
                                 max_retries=max_retries)
    order = {s.name: i for i, s in enumerate(shot_list.shots)}
    combined = sorted(results + unsupported,
                      key=lambda r: order.get(r["shot"], len(order)))
    return {"results": combined,
           "index_errors": {name: str(exc) for name, exc in index_errors.items()}}


def compare_sheet(paths, out_path, cols: int = 3, thumb: int = 512):
    """Tile renders into one sheet for a single visual pass. Returns None when Pillow
    is absent -- a missing optional dependency must not fail a batch that already
    rendered successfully."""
    if not _PIL_AVAILABLE or not paths:
        return None
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:                                  # noqa: BLE001
            continue
        img.thumbnail((thumb, thumb))
        images.append(img)
    if not images:
        return None

    cell_w = max(i.width for i in images)
    cell_h = max(i.height for i in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), (18, 18, 18))
    for i, img in enumerate(images):
        x = (i % cols) * cell_w + (cell_w - img.width) // 2
        y = (i // cols) * cell_h + (cell_h - img.height) // 2
        sheet.paste(img, (x, y))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
