from pathlib import Path

import pytest
import yaml
from comfyrack import batch, dispatch, errors
from comfyrack.shots import load_shots
from comfyrack.profile import Character
from comfyrack.lexicon import Entry
from comfyrack.registry import Flag, Manifest


class FakeProfile:
    def __init__(self):
        self._chars = {
            "char_a": Character(name="char_a", id_lora="char_a.safetensors",
                                id_strength=0.8, style_lora="style.safetensors",
                                style_strength=1.0, trigger="trig_a",
                                age_bands={"default": 0.6, "close_shot": 0.45}),
            "char_b": Character(name="char_b", id_lora="char_b.safetensors", id_strength=0.8,
                              style_lora="style.safetensors", style_strength=1.0),
        }

    def characters(self):
        return self._chars

    def character(self, name):
        return self._chars[name]

    def resolve_id_strength(self, name, sequence=None, close_shot=False):
        """Mirrors Profile.resolve_id_strength's tiering: per-sequence band beats
        close-shot band beats default band beats the LoRA's own locked strength,
        with None (never falsy 0) as the unset sentinel at every tier."""
        c = self._chars[name]
        bands = c.age_bands or {}
        sequences = bands.get("sequences") or {}
        if sequence is not None and sequences.get(sequence) is not None:
            return float(sequences[sequence])
        if close_shot and bands.get("close_shot") is not None:
            return float(bands["close_shot"])
        if bands.get("default") is not None:
            return float(bands["default"])
        return c.id_strength


SAMPLE = {
    "sequence": 8, "primary_character": "char_a", "comfy_url": None,
    "characters": ["char_a"],
    "shots": [
        {"name": "s01", "character": "char_a", "people": "solo",
         "positive_prefix": "trig_a", "action_text": "dancing",
         "location_text": "a dock", "seed": 1, "id_strength": 0.45},
        {"name": "s02", "character": "char_b", "people": "solo",
         "positive_prefix": "j0sh", "action_text": "waiting",
         "location_text": "a dock", "seed": 2, "id_strength": 0.5},
    ],
}


def write(tmp_path, data=None):
    p = tmp_path / "seq01_shots.yaml"
    p.write_text(yaml.safe_dump(data or SAMPLE, sort_keys=False), encoding="utf-8")
    return p


def test_requirements_include_the_shot_characters_id_lora(tmp_path):
    sl = load_shots(write(tmp_path))
    req = batch.shot_requirements(sl.shots[0], FakeProfile())
    assert "char_a.safetensors" in req["loras"]


def test_requirements_include_the_style_lora(tmp_path):
    sl = load_shots(write(tmp_path))
    assert "style.safetensors" in batch.shot_requirements(sl.shots[0], FakeProfile())["loras"]


def test_requirements_for_a_composite_shot_include_every_character(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"].append({
        "name": "s03", "via": "shot_builder",
        "base": {"character": "char_a", "seed": 3, "action_text": "x",
                 "positive_prefix": "trig_a", "location_text": "y"},
        "composite": {"seed": 4, "refs": [{"character": "char_b", "clause": "..."}]}})
    sl = load_shots(write(tmp_path, data))
    req = batch.shot_requirements(sl.shots[2], FakeProfile())
    assert {"char_a.safetensors", "char_b.safetensors"} <= set(req["loras"])


def test_an_unknown_character_contributes_no_requirement_rather_than_raising(tmp_path):
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"][0]["character"] = "nobody"
    sl = load_shots(write(tmp_path, data))
    req = batch.shot_requirements(sl.shots[0], FakeProfile())
    assert all("nobody" not in m for m in req["loras"])


def test_a_lexicon_renders_lora_becomes_a_routing_requirement(tmp_path):
    """A trigger only grounds while its own LoRA is loaded, so the LoRA a lexicon
    render names must travel with the shot as a routing requirement."""
    class FakeLexicon:
        def all(self):
            return {"char_a": Entry(term="char_a", kind="character", renders={
                "illustrious": {"trigger": "trig_a", "tags": "1girl",
                                "lora": "mira_lex.safetensors"}})}

    sl = load_shots(write(tmp_path))
    req = batch.shot_requirements(sl.shots[0], FakeProfile(), lexicon=FakeLexicon())
    assert "mira_lex.safetensors" in req["loras"]
    # ...and the profile's own LoRAs are still required, not replaced by it.
    assert "char_a.safetensors" in req["loras"]


def test_without_a_lexicon_only_profile_loras_are_required(tmp_path):
    sl = load_shots(write(tmp_path))
    req = batch.shot_requirements(sl.shots[0], FakeProfile())
    assert set(req["loras"]) == {"char_a.safetensors", "style.safetensors"}


def test_build_jobs_produces_one_job_per_shot(tmp_path):
    jobs = batch.build_jobs(load_shots(write(tmp_path)), FakeProfile())
    assert [j.shot_name for j in jobs] == ["s01", "s02"]


def test_only_filter_restricts_the_batch_to_one_shot(tmp_path):
    sl = load_shots(write(tmp_path))
    jobs = batch.build_jobs(sl, FakeProfile(), only="s02")
    assert [j.shot_name for j in jobs] == ["s02"]


def test_only_filter_with_no_match_yields_no_jobs(tmp_path):
    jobs = batch.build_jobs(load_shots(write(tmp_path)), FakeProfile(), only="s99")
    assert jobs == []


def test_batch_routes_each_shot_to_a_machine_that_has_its_lora(tmp_path):
    caps = {
        "local": dispatch.Capability("local", "http://local:8188",
                                     models={"loras": ["char_a.safetensors",
                                                       "char_b.safetensors",
                                                       "style.safetensors"]},
                                     node_classes=set()),
        "remote": dispatch.Capability("remote", "http://remote:8188",
                                      models={"loras": ["char_a.safetensors",
                                                        "style.safetensors"]},
                                      node_classes=set()),
    }
    jobs = batch.build_jobs(load_shots(write(tmp_path)), FakeProfile())
    seen = {}

    def runner(job, machine):
        seen[job.shot_name] = machine
        return {"paths": []}

    results = dispatch.run_queue(jobs, caps, runner)
    assert seen["s02"] == "local"      # only local has char_b
    assert all(r["status"] == "ok" for r in results)


def test_compare_sheet_returns_none_when_pillow_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "_PIL_AVAILABLE", False)
    assert batch.compare_sheet([], tmp_path / "sheet.png") is None


def test_compare_sheet_writes_a_file_when_given_images(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (64, 64), (i * 40, 0, 0)).save(p)
        paths.append(p)
    sheet = batch.compare_sheet(paths, tmp_path / "sheet.png", cols=2)
    assert sheet.is_file()


# -- _resolve_terms ----------------------------------------------------------
#
# The brief's Step 1 fixture has no test covering _resolve_terms at all: none of
# SAMPLE's action_text values contain a {term} placeholder. Without this function,
# a real chapter's "{char_c}" would reach ComfyUI as literal punctuation instead of
# resolving from the shot list's own `characters:` prose -- this was a Critical
# finding earlier this session, so it gets direct coverage here rather than only
# being exercised incidentally through run_batch.

class FakeShotList:
    def __init__(self, characters):
        self.characters = characters


def test_resolve_terms_substitutes_a_known_character_from_shot_list_prose():
    shot_list = FakeShotList({"char_c": {"text": "a girl with silver hair"}})
    out = batch._resolve_terms("mid-dance, {char_c} watching closely", shot_list)
    assert out == "mid-dance, a girl with silver hair watching closely"


def test_resolve_terms_leaves_an_unknown_term_as_a_literal_brace():
    shot_list = FakeShotList({"char_c": {"text": "a girl with silver hair"}})
    out = batch._resolve_terms("mid-dance, {unknown} watching closely", shot_list)
    assert out == "mid-dance, {unknown} watching closely"


def test_resolve_terms_falls_back_case_insensitively():
    shot_list = FakeShotList({"char_c": {"text": "a girl with silver hair"}})
    out = batch._resolve_terms("{Char_c} watches", shot_list)
    assert out == "a girl with silver hair watches"


# -- run_batch end-to-end -----------------------------------------------------
#
# Fix round 1 regression coverage. Nothing in the tests above ever calls
# run_batch(): the routing test drives dispatch.run_queue directly with a
# hand-built capabilities dict, bypassing dispatch.index() entirely -- which is
# exactly how run_batch's `capabilities = dispatch.index(...)` bug (assigning
# the whole (caps, errors) tuple instead of unpacking it) shipped invisibly.
# These tests build a minimal fake Rack and drive run_batch() through a real
# dispatch.index() call so that bug class can't ship silently again.
#
# Fix round 2 regression coverage (all below this point): a Critical
# machine-routing bug in the no-`--machines` default path, index_errors
# returned as data instead of via warnings.warn(), and via: shot_builder
# shots reported "unsupported" instead of silently rendered as a plain
# single-character shot. The original round-1 FakeConfig collapsed
# machine_url(None) and machine_url("default") into one dict lookup, which is
# exactly what let the Critical bug through -- RealisticFakeConfig below
# keeps those two branches genuinely distinct, the same as the real Config.

class FakeComfyClient:
    """Just enough of the ComfyClient interface for dispatch.index(): an
    object_info() a LoraLoader entry can be read out of."""
    def __init__(self, url, loras=()):
        self.url = url
        self._loras = list(loras)

    def object_info(self, class_type=None):
        return {
            "KSampler": {"input": {"required": {}}},
            "LoraLoader": {"input": {"required": {
                "lora_name": [self._loras, {}]}}},
        }


class DeadComfyClient:
    def __init__(self, url):
        self.url = url

    def object_info(self, class_type=None):
        from comfyrack import errors
        raise errors.ComfyrackError(f"cannot reach ComfyUI at {self.url}")


class RealisticFakeConfig:
    """Mirrors comfyrack.config.Config.machine_url's actual two branches:

    - `machine_url(name)` for a real name requires that name to be in
      `machines`, raising UsageError otherwise.
    - `machine_url(None)` (the "no explicit machine" path) never touches that
      lookup -- it checks an env override, then falls back to
      `machines.get("default", ...)`, tolerating there being no `[machines]`
      table at all.

    A fake that collapses both into one dict keyed by `name or "default"`
    (round 1's FakeConfig) cannot distinguish these, which is exactly what
    let the Critical routing bug -- run_batch's no-`--machines` path passing
    the invented key "default" through to `machine_url("default")`, the
    STRICT branch -- ship invisibly."""
    def __init__(self, machines=None, default_url="http://127.0.0.1:8188",
                env_url=None):
        self.machines = machines or {}
        self.default_url = default_url
        self.env_url = env_url

    def machine_url(self, name):
        if name:
            if name not in self.machines:
                from comfyrack.errors import UsageError
                raise UsageError(f"unknown machine {name!r}")
            return self.machines[name]
        if self.env_url:
            return self.env_url
        return self.machines.get("default", self.default_url)


class FakeLexicon:
    def all(self):
        return {}


class FakeRunResult:
    def __init__(self, paths):
        self.paths = paths


def fake_manifest(*, id_flags=True):
    """A manifest shaped like the REAL registry/anima/character-scene one for the
    flags run_batch actually submits. It carries id_strength_model/
    id_strength_clip because the real workflow does -- a stub declaring only
    `seed` would have let run_batch drop a shot's id_strength on the floor
    unnoticed, which is precisely how the Critical shipped."""
    flags = {
        "seed": Flag(param="seed", node="9", type="int", required=True),
        "positive_text": Flag(param="text", node="6", type="str", required=True),
        "negative_text": Flag(param="text", node="7", type="str", required=True),
    }
    if id_flags:
        flags["id_strength_model"] = Flag(param="strength_model", node="5",
                                          type="float", required=True)
        flags["id_strength_clip"] = Flag(param="strength_clip", node="5",
                                         type="float", required=True)
    return Manifest(name="character-scene", family="anima", layer="builtin",
                    description="", workflow_path=Path("character-scene.json"),
                    output_node_title="Save Image", flags=flags)


class FakeRack:
    """Just enough of Rack's surface for run_batch(): .config.machine_url,
    .client_for_url, .profile, .lexicon, .describe, .run. `.run()` mirrors the
    real `Rack.run()`'s `url=` vs `machine=` branching exactly (`url=` bypasses
    `Config.machine_url()` entirely; `machine=` goes through it) -- so a
    batch.py regression back to passing `machine=` down the no-`--machines`
    default path fails here the same way it fails against the real Rack."""
    def __init__(self, config, client_factory, manifest=None):
        self.config = config
        self.profile = FakeProfile()
        self.lexicon = FakeLexicon()
        self.manifest = manifest if manifest is not None else fake_manifest()
        self._client_factory = client_factory
        self._clients = {}
        self.run_calls = []

    def describe(self, name):
        return self.manifest

    def client_for_url(self, url):
        if url not in self._clients:
            self._clients[url] = self._client_factory(url)
        return self._clients[url]

    def client(self, machine=None):
        return self.client_for_url(self.config.machine_url(machine))

    def run(self, workflow, recipe=None, overrides=None, machine=None, out=None,
            skip_preflight=False, url=None, on_progress=None):
        client = self.client_for_url(url) if url is not None else self.client(machine)
        self.run_calls.append({"workflow": workflow, "overrides": overrides,
                               "machine": machine, "url": url, "client_url": client.url,
                               "on_progress": on_progress})
        if on_progress is not None:
            on_progress("step 10/20 (50%)")
        seed = (overrides or {}).get("seed", 0)
        return FakeRunResult([f"/renders/{seed}.png"])


ALL_LORAS = ["char_a.safetensors", "char_b.safetensors", "style.safetensors"]


def test_run_batch_with_no_machines_flag_still_resolves_a_working_client(tmp_path):
    """Critical regression: with NO `[machines]` table configured at all (the
    common case -- relying on COMFY_URL or the builtin default) and no
    `--machines` flag, run_batch's default path used to invent the key
    "default" and pass it to rack.run(machine="default"), which re-resolved
    through Config.machine_url("default") -- the STRICT, named-machine
    branch, requiring an actual `[machines] default = ...` entry. Every shot
    failed with UsageError: unknown machine 'default'. RealisticFakeConfig
    here has `machines={}`, so this fails loudly against the unfixed code and
    must pass against the fix, which routes by the already-resolved URL
    instead of re-resolving a synthetic name."""
    sl = load_shots(write(tmp_path))
    config = RealisticFakeConfig(machines={}, default_url="http://local:8188")
    rack = FakeRack(config, client_factory=lambda url: FakeComfyClient(url, loras=ALL_LORAS))

    out = batch.run_batch(rack, sl, "character-scene")

    assert [r["shot"] for r in out["results"]] == ["s01", "s02"]
    assert all(r["status"] == "ok" for r in out["results"]), out["results"]
    assert rack.run_calls
    # Every actual submission targeted the one real URL, not a name that
    # needed re-resolving.
    assert all(c["client_url"] == "http://local:8188" for c in rack.run_calls)


def test_run_batch_with_explicit_machines_still_routes_to_the_right_one(tmp_path):
    """The explicit `--machines` path already used real, configured machine
    names, so it worked before this fix -- verify it keeps working: `local`
    holds char_b's LoRA, `remote` does not, so s02 (char_b) must land on local."""
    sl = load_shots(write(tmp_path))
    config = RealisticFakeConfig(machines={"local": "http://local:8188",
                                           "remote": "http://remote:8188"})

    def factory(url):
        if "local" in url:
            return FakeComfyClient(url, loras=ALL_LORAS)
        return FakeComfyClient(url, loras=["char_a.safetensors", "style.safetensors"])

    rack = FakeRack(config, client_factory=factory)
    out = batch.run_batch(rack, sl, "character-scene", machines=["local", "remote"])

    assert all(r["status"] == "ok" for r in out["results"]), out["results"]
    by_shot = {r["shot"]: r["machine"] for r in out["results"]}
    assert by_shot["s02"] == "local"      # only local has char_b.safetensors


def test_run_batch_returns_index_errors_as_data_not_a_warning(tmp_path, recwarn):
    """Important: index_errors must be something the CALLER receives and
    decides what to do with -- matching guard.enforce()'s and
    Rack.assemble_prompt()'s established "return diagnostics as data" pattern
    -- not a warnings.warn() side channel a caller can't intercept, can
    accidentally silence (-W ignore), can accidentally make fatal (-W error),
    or that the default dedup filter could hide on a second machine's
    failure. `local` is dead; `remote` holds everything, so the batch itself
    still completes."""
    sl = load_shots(write(tmp_path))
    config = RealisticFakeConfig(machines={"local": "http://local:8188",
                                           "remote": "http://remote:8188"})

    def factory(url):
        return DeadComfyClient(url) if "local" in url else FakeComfyClient(url, loras=ALL_LORAS)

    rack = FakeRack(config, client_factory=factory)
    out = batch.run_batch(rack, sl, "character-scene", machines=["local", "remote"])

    assert "local" in out["index_errors"]
    assert "cannot reach" in out["index_errors"]["local"]
    assert all(r["status"] == "ok" for r in out["results"]), out["results"]
    # No warnings.warn()-based side channel duplicating this.
    assert not any("could not be indexed" in str(w.message) for w in recwarn)


def test_run_batch_reports_a_via_shot_as_unsupported_rather_than_silently_rendering_the_base(tmp_path):
    """Important: a via: shot_builder composite shot must not silently render
    as a plain single-character shot missing its second lead -- that is the
    exact plausible-and-wrong failure class this project exists to prevent.
    It must come back with a distinct, explicit status, and rack.run() must
    never actually be invoked for it (proven by seed: the composite shot's own
    render-stage seed, 3, must never appear in any submitted overrides)."""
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    data["shots"].append({
        "name": "s03", "via": "shot_builder",
        "base": {"character": "char_a", "seed": 3, "action_text": "x",
                 "positive_prefix": "trig_a", "location_text": "y"},
        "composite": {"seed": 4, "refs": [{"character": "char_b", "clause": "..."}]}})
    sl = load_shots(write(tmp_path, data))
    config = RealisticFakeConfig(machines={}, default_url="http://local:8188")
    rack = FakeRack(config, client_factory=lambda url: FakeComfyClient(url, loras=ALL_LORAS))

    out = batch.run_batch(rack, sl, "character-scene")

    by_shot = {r["shot"]: r for r in out["results"]}
    assert [r["shot"] for r in out["results"]] == ["s01", "s02", "s03"]
    assert by_shot["s03"]["status"] == "unsupported"
    assert by_shot["s01"]["status"] == "ok"
    assert by_shot["s02"]["status"] == "ok"
    assert not any(c["overrides"].get("seed") == 3 for c in rack.run_calls)


# -- id_strength plumbing (Critical) ------------------------------------------
#
# Final-review Critical. Shot lists author a per-shot `id_strength` (0.45 on s01 of the
# sample shot list), it reached
# `Shot.id_strength` intact, and then run_batch's runner() closure -- which only
# ever set seed, positive_text and negative_text -- dropped it. resolve_values
# then fell through to the value baked into the workflow JSON, 0.0, so the
# identity LoRA rendered switched off: a finished-looking, faceless sequence
# reported as `9/9 ok`. These tests assert on the OVERRIDES ACTUALLY SUBMITTED, because the
# submitted dict is where the break was -- resolve_id_strength itself was fine
# and fully unit-tested, it just had no production caller.

def _overrides_by_shot(rack, shot_list):
    seed_to_shot = {s.seed: s.name for s in shot_list.shots}
    return {seed_to_shot[c["overrides"]["seed"]]: c["overrides"]
            for c in rack.run_calls}


def test_run_batch_submits_a_shots_id_strength_under_the_manifests_id_flags(tmp_path):
    sl = load_shots(write(tmp_path))
    config = RealisticFakeConfig(machines={}, default_url="http://local:8188")
    rack = FakeRack(config, client_factory=lambda url: FakeComfyClient(url, loras=ALL_LORAS))

    out = batch.run_batch(rack, sl, "character-scene")

    assert all(r["status"] == "ok" for r in out["results"]), out["results"]
    by_shot = _overrides_by_shot(rack, sl)
    # s01 authors id_strength: 0.45, s02 authors 0.5 -- each shot's own value,
    # not one value applied to the whole batch.
    assert by_shot["s01"]["id_strength_model"] == 0.45
    assert by_shot["s01"]["id_strength_clip"] == 0.45
    assert by_shot["s02"]["id_strength_model"] == 0.5
    assert by_shot["s02"]["id_strength_clip"] == 0.5


def test_run_batch_uses_the_manifests_own_key_when_the_flag_is_spelled_as_an_alias(tmp_path):
    """The flag names are read off the manifest through resolve_key, not
    hardcoded into the submitted dict, so a workflow keying the concept under
    one of its declared aliases still receives the value."""
    sl = load_shots(write(tmp_path))
    manifest = fake_manifest(id_flags=False)
    manifest.flags["id_model_weight"] = Flag(param="strength_model", node="5",
                                             type="float",
                                             aliases=["id_strength_model"])
    manifest.__post_init__()          # rebuild the alias map after the edit
    config = RealisticFakeConfig(machines={}, default_url="http://local:8188")
    rack = FakeRack(config,
                    client_factory=lambda url: FakeComfyClient(url, loras=ALL_LORAS),
                    manifest=manifest)

    batch.run_batch(rack, sl, "character-scene")

    by_shot = _overrides_by_shot(rack, sl)
    assert by_shot["s01"]["id_model_weight"] == 0.45
    assert "id_strength_model" not in by_shot["s01"]


def test_run_batch_falls_back_to_the_characters_profile_when_a_shot_omits_id_strength(tmp_path):
    """A shot with no `id_strength` of its own must not fall through to the
    baked 0.0 either. profile.resolve_id_strength walks the character's
    per-sequence / close-shot / default / locked chain -- the sample shot list has no sequence
    band for char_a, so her `default` band, 0.6, governs."""
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    del data["shots"][0]["id_strength"]
    sl = load_shots(write(tmp_path, data))
    config = RealisticFakeConfig(machines={}, default_url="http://local:8188")
    rack = FakeRack(config, client_factory=lambda url: FakeComfyClient(url, loras=ALL_LORAS))

    batch.run_batch(rack, sl, "character-scene")

    by_shot = _overrides_by_shot(rack, sl)
    assert by_shot["s01"]["id_strength_model"] == 0.6
    # The shot that DOES author one still wins over its character's band.
    assert by_shot["s02"]["id_strength_model"] == 0.5


def test_a_shot_naming_no_known_character_submits_no_id_strength_at_all(tmp_path):
    """An unknown character contributes nothing rather than raising, matching
    shot_requirements -- and `no id_strength override` is different from
    `id_strength 0.0`, so nothing is invented."""
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    del data["shots"][0]["id_strength"]
    data["shots"][0]["character"] = "nobody"
    sl = load_shots(write(tmp_path, data))
    config = RealisticFakeConfig(machines={}, default_url="http://local:8188")
    rack = FakeRack(config, client_factory=lambda url: FakeComfyClient(url, loras=ALL_LORAS))

    batch.run_batch(rack, sl, "character-scene")

    assert "id_strength_model" not in _overrides_by_shot(rack, sl)["s01"]


def test_a_workflow_with_no_id_strength_flag_refuses_shots_that_declare_one(tmp_path):
    """The same Critical in its other shape: a workflow with no identity-LoRA
    knob cannot apply an authored id_strength, so submitting the batch anyway
    would render every shot at whatever the graph bakes in. It fails loud and
    UP FRONT -- before any GPU time -- naming the shots that would have been
    silently downgraded."""
    sl = load_shots(write(tmp_path))
    config = RealisticFakeConfig(machines={}, default_url="http://local:8188")
    rack = FakeRack(config,
                    client_factory=lambda url: FakeComfyClient(url, loras=ALL_LORAS),
                    manifest=fake_manifest(id_flags=False))

    with pytest.raises(errors.UsageError) as ei:
        batch.run_batch(rack, sl, "character-scene")

    assert "s01" in str(ei.value) and "s02" in str(ei.value)
    assert ei.value.exit_code == 2
    assert not rack.run_calls          # nothing was submitted


def test_a_workflow_with_no_id_strength_flag_is_fine_when_no_shot_declares_one(tmp_path):
    """Ring 1 must keep working: an upscale or harmonize workflow has no
    identity LoRA, and a shot list that never mentions id_strength runs on it
    unchanged."""
    data = yaml.safe_load(yaml.safe_dump(SAMPLE))
    for shot in data["shots"]:
        del shot["id_strength"]
        shot["character"] = ""
    sl = load_shots(write(tmp_path, data))
    config = RealisticFakeConfig(machines={}, default_url="http://local:8188")
    rack = FakeRack(config,
                    client_factory=lambda url: FakeComfyClient(url, loras=ALL_LORAS),
                    manifest=fake_manifest(id_flags=False))

    out = batch.run_batch(rack, sl, "character-scene")

    assert all(r["status"] == "ok" for r in out["results"]), out["results"]


# -- on_progress threading ----------------------------------------------------

def test_run_batch_labels_each_progress_line_with_its_shot(tmp_path):
    """A batch is N renders in a row, so an unlabelled `step 12/20` halfway
    down a 9-shot run says nothing about which shot is 60% done."""
    sl = load_shots(write(tmp_path))
    config = RealisticFakeConfig(machines={}, default_url="http://local:8188")
    rack = FakeRack(config, client_factory=lambda url: FakeComfyClient(url, loras=ALL_LORAS))
    lines = []

    batch.run_batch(rack, sl, "character-scene", on_progress=lines.append)

    assert lines == ["s01: step 10/20 (50%)", "s02: step 10/20 (50%)"]


def test_run_batch_passes_no_callback_down_when_it_was_given_none(tmp_path):
    """Default stays silent so library callers never open a websocket nobody
    reads -- runner._start_progress skips the socket entirely on a None."""
    sl = load_shots(write(tmp_path))
    config = RealisticFakeConfig(machines={}, default_url="http://local:8188")
    rack = FakeRack(config, client_factory=lambda url: FakeComfyClient(url, loras=ALL_LORAS))

    batch.run_batch(rack, sl, "character-scene")

    assert all(c["on_progress"] is None for c in rack.run_calls)
