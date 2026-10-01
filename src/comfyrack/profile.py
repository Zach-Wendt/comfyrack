"""Project profile data: characters, locations, recipes.

comfyrack owns the schema; the project owns the data. The schema deliberately matches
the recipe.yaml sidecars this pipeline already writes, so adopting comfyrack means
pointing [paths] at existing directories rather than moving files.

Ring 2 is inert when no paths are configured -- a project using only ring 1 sees
empty collections, not errors."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import ComfyrackError, NotFoundError, UsageError

# recipes.load_recipe owns these two conventions; see Profile._recipes_dir.
RECIPE_EXTS = (".yaml", ".yml")


@dataclass
class Character:
    name: str
    id_lora: str = ""
    # `None`, not 0.0, is "unset" for both strengths -- the same sentinel discipline
    # shots._val_field uses for seed and id_strength. 0.0 is a real, renderable
    # weight (an identity LoRA switched off), so a default of 0.0 would make an
    # omitted `strength:` indistinguishable from a deliberate one.
    id_strength: float | None = None
    style_lora: str = ""
    style_strength: float | None = None
    trigger: str = ""
    base_model_family: str = ""
    sampler: dict = field(default_factory=dict)
    age_bands: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    path: Path | None = None


@dataclass
class Subject(Character):
    """Generic identity-locked actor -- supersedes Character as the general case."""
    pass


@dataclass
class Location:
    key: str
    plate_path: Path | None = None
    description: str = ""


@dataclass
class Scene(Location):
    """Named, reusable location + mood/style/action context."""
    raw: dict = field(default_factory=dict)
    path: Path | None = None


def _load_mapping(path: Path, what: str, help_text: str) -> dict:
    """yaml.safe_load, guarded the way lexicon.all() and shots.load_shots guard it.

    A file that parses to a non-mapping -- a list, a bare string, a number -- makes
    the .get() calls below raise AttributeError, and a syntax error raises
    yaml.parser.ParserError. Neither is a ComfyrackError, so neither carries an exit
    code or a help line, and because characters()/locations() walk a whole directory
    the traceback would not even name the file that is broken."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ComfyrackError(f"{path} is not a readable {what}: {exc}",
                             help_text=help_text) from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ComfyrackError(
            f"{path} is not a readable {what}: top-level YAML is a "
            f"{type(raw).__name__}, not a mapping",
            help_text=help_text)
    return raw


def _mapping_field(raw: dict, key: str, path: Path) -> dict:
    """A nested block that is present but not a mapping fails the same way a
    non-mapping document does, so it gets the same guard."""
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ComfyrackError(
            f"{path}: `{key}:` is a {type(value).__name__}, not a mapping",
            help_text=f"`{key}:` must be a mapping of settings, not a bare value")
    return value


def _number(value, field_name: str, path) -> float | None:
    """Coerce a strength/band value, preserving `None` as UNSET.

    `float(x or 0.0)` -- what this replaced -- collapsed three different inputs
    (absent, empty, and an explicit 0) into the same silent 0.0, and let a
    non-numeric value out as a bare ValueError."""
    if value is None:
        return None
    if isinstance(value, bool):   # YAML `strength: yes` is a bool, not a weight
        raise ComfyrackError(
            f"{path}: `{field_name}` is {value!r}, which is not a number",
            help_text=f"`{field_name}` must be a number, e.g. {field_name}: 0.8")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ComfyrackError(
            f"{path}: `{field_name}` is {value!r}, which is not a number",
            help_text=f"`{field_name}` must be a number, e.g. {field_name}: 0.8",
        ) from exc


class Profile:
    def __init__(self, config):
        self.config = config

    # -- characters --------------------------------------------------------

    def characters(self) -> dict:
        root = self.config.path("characters")
        if root is None and self.config.project_root:
            root = self.config.project_root / ".comfyrack" / "characters"
        if not root or not root.is_dir():
            return {}
        out = {}
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            recipe = d / "recipe.yaml"
            if not recipe.is_file():
                continue
            raw = _load_mapping(
                recipe, "character recipe",
                help_text="a character recipe.yaml must be a mapping with "
                          "character/id_lora/style_lora keys")
            id_lora = _mapping_field(raw, "id_lora", recipe)
            style = _mapping_field(raw, "style_lora", recipe)
            out[d.name] = Character(
                name=raw.get("character", d.name),
                id_lora=id_lora.get("name", ""),
                id_strength=_number(id_lora.get("strength"),
                                    "id_lora.strength", recipe),
                trigger=id_lora.get("trigger", ""),
                style_lora=style.get("name", ""),
                style_strength=_number(style.get("strength"),
                                       "style_lora.strength", recipe),
                base_model_family=raw.get("base_model_family", ""),
                sampler=_mapping_field(raw, "sampler", recipe),
                age_bands=_mapping_field(raw, "age_bands", recipe),
                raw=raw,
                path=recipe,
            )
        return out

    def character(self, name: str) -> Character:
        chars = self.characters()
        if name in chars:
            return chars[name]
        subjs = self.subjects()
        if name in subjs:
            s = subjs[name]
            return Character(
                name=s.name, id_lora=s.id_lora, id_strength=s.id_strength,
                style_lora=s.style_lora, style_strength=s.style_strength,
                trigger=s.trigger, base_model_family=s.base_model_family,
                sampler=s.sampler, age_bands=s.age_bands, raw=s.raw, path=s.path
            )
        known = ", ".join(sorted(set(chars) | set(subjs))) or "(none)"
        raise NotFoundError(f"no character named {name!r}",
                            help_text=f"known characters: {known}")

    # -- subjects -----------------------------------------------------------

    def subjects(self) -> dict:
        root = self.config.path("subjects")
        if root is None and self.config.project_root:
            root = self.config.project_root / ".comfyrack" / "subjects"
        if not root or not root.is_dir():
            return {}
        out = {}
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            recipe = d / "recipe.yaml"
            if not recipe.is_file():
                recipe = d / "subject.yaml"
            if not recipe.is_file():
                continue
            raw = _load_mapping(
                recipe, "subject recipe",
                help_text="a subject recipe.yaml must be a mapping with "
                          "subject/character/id_lora/style_lora keys")
            id_lora = _mapping_field(raw, "id_lora", recipe)
            style = _mapping_field(raw, "style_lora", recipe)
            out[d.name] = Subject(
                name=raw.get("subject", raw.get("character", d.name)),
                id_lora=id_lora.get("name", ""),
                id_strength=_number(id_lora.get("strength"),
                                    "id_lora.strength", recipe),
                trigger=id_lora.get("trigger", ""),
                style_lora=style.get("name", ""),
                style_strength=_number(style.get("strength"),
                                       "style_lora.strength", recipe),
                base_model_family=raw.get("base_model_family", ""),
                sampler=_mapping_field(raw, "sampler", recipe),
                age_bands=_mapping_field(raw, "age_bands", recipe),
                raw=raw,
                path=recipe,
            )
        return out

    def subject(self, name: str) -> Subject:
        subjs = self.subjects()
        if name in subjs:
            return subjs[name]
        chars = self.characters()
        if name in chars:
            c = chars[name]
            return Subject(
                name=c.name, id_lora=c.id_lora, id_strength=c.id_strength,
                style_lora=c.style_lora, style_strength=c.style_strength,
                trigger=c.trigger, base_model_family=c.base_model_family,
                sampler=c.sampler, age_bands=c.age_bands, raw=c.raw, path=c.path
            )
        known = ", ".join(sorted(set(subjs) | set(chars))) or "(none)"
        raise NotFoundError(f"no subject named {name!r}",
                            help_text=f"known subjects: {known}")

    def resolve_id_strength(self, name: str, sequence=None,
                            close_shot: bool = False) -> float:
        """Per-sequence band beats close-shot band beats default beats the LoRA's own
        locked strength.

        `None` is the "unset" sentinel at every tier, never a falsy 0: an explicit
        `0` in any band is a real weight and survives the chain instead of falling
        through to the next tier.

        When no tier supplies one, this raises rather than picking a number. An
        identity LoRA applied at a guessed strength is the project's signature
        failure -- the render comes back looking finished, with the wrong face, and
        nothing anywhere reports a problem."""
        c = self.subject(name)
        bands = c.age_bands or {}
        sequences = bands.get("sequences")
        if not isinstance(sequences, dict):
            if sequences is not None:
                raise ComfyrackError(
                    f"{c.path}: `age_bands.sequences:` is a "
                    f"{type(sequences).__name__}, not a mapping",
                    help_text="`sequences:` must map a sequence to a strength, "
                              "e.g. sequences: {12: 0.7}")
            sequences = {}

        if sequence is not None and sequences.get(sequence) is not None:
            return _number(sequences[sequence],
                           f"age_bands.sequences.{sequence}", c.path)
        if close_shot and bands.get("close_shot") is not None:
            return _number(bands["close_shot"], "age_bands.close_shot", c.path)
        if bands.get("default") is not None:
            return _number(bands["default"], "age_bands.default", c.path)
        if c.id_strength is not None:
            return c.id_strength
        raise UsageError(
            f"character {name!r} has no id_strength: neither `id_lora.strength` "
            f"nor an `age_bands` entry is set",
            help_text=f"set `id_lora:` `strength:` (or an `age_bands` value) in "
                      f"{c.path} -- an identity LoRA with an unknown weight is not "
                      f"safely renderable",
        )

    # -- locations ---------------------------------------------------------

    def locations(self) -> dict:
        root = self.config.path("locations")
        if root is None and self.config.project_root:
            root = self.config.project_root / ".comfyrack" / "locations"
        if not root or not root.is_dir():
            return {}
        out = {}
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            plate = d / "reference_plate.png"
            meta = d / "location.yaml"
            desc = ""
            if meta.is_file():
                desc = _load_mapping(
                    meta, "location file",
                    help_text="a location.yaml must be a mapping with a "
                              "`description:` key").get("description", "") or ""
            out[d.name] = Location(key=d.name,
                                   plate_path=plate if plate.is_file() else None,
                                   description=desc)
        return out

    def location(self, key: str) -> Location:
        locs = self.locations()
        if key in locs:
            return locs[key]
        scs = self.scenes()
        if key in scs:
            s = scs[key]
            return Location(key=s.key, plate_path=s.plate_path, description=s.description)
        known = ", ".join(sorted(set(locs) | set(scs))) or "(none)"
        raise NotFoundError(f"no location named {key!r}",
                            help_text=f"known locations: {known}")

    # -- scenes ------------------------------------------------------------

    def scenes(self) -> dict:
        root = self.config.path("scenes")
        if root is None and self.config.project_root:
            root = self.config.project_root / ".comfyrack" / "scenes"
        if not root or not root.is_dir():
            return {}
        out = {}
        for item in sorted(root.iterdir()):
            if item.is_dir():
                plate = item / "reference_plate.png"
                meta = item / "scene.yaml"
                if not meta.is_file():
                    meta = item / "location.yaml"
                desc = ""
                raw = {}
                if meta.is_file():
                    raw = _load_mapping(
                        meta, "scene file",
                        help_text="a scene.yaml must be a mapping with a "
                                  "`description:` key")
                    desc = raw.get("description", "") or ""
                out[item.name] = Scene(
                    key=item.name,
                    plate_path=plate if plate.is_file() else None,
                    description=desc,
                    raw=raw,
                    path=meta if meta.is_file() else None,
                )
            elif item.is_file() and item.suffix in RECIPE_EXTS:
                raw = _load_mapping(
                    item, "scene file",
                    help_text="a scene YAML file must be a mapping with a "
                              "`description:` key")
                desc = raw.get("description", "") or ""
                out[item.stem] = Scene(
                    key=item.stem,
                    plate_path=None,
                    description=desc,
                    raw=raw,
                    path=item,
                )
        return out

    def scene(self, key: str) -> Scene:
        scs = self.scenes()
        if key in scs:
            return scs[key]
        locs = self.locations()
        if key in locs:
            l = locs[key]
            return Scene(key=l.key, plate_path=l.plate_path, description=l.description)
        known = ", ".join(sorted(set(scs) | set(locs))) or "(none)"
        raise NotFoundError(f"no scene named {key!r}",
                            help_text=f"known scenes: {known}")

    # -- recipes -----------------------------------------------------------

    def _recipes_dir(self) -> Path:
        """Where recipes.load_recipe looks. Mirrored here, not reinvented.

        `recipe list` and `recipe get <name>` must agree about what exists. Globbing
        only `*.yaml` under only the configured path made list() under-report twice
        over: a `.yml` recipe was invisible but loadable, and a project with no
        `[paths] recipes` reported nothing while get() still resolved out of the
        fallback directory."""
        configured = self.config.path("recipes")
        if configured is not None:
            return configured
        root = self.config.project_root
        return ((root / ".comfyrack" / "recipes") if root
                else Path.cwd() / ".comfyrack" / "recipes")

    def recipes(self) -> list:
        root = self._recipes_dir()
        if not root.is_dir():
            return []
        # A set: load_recipe resolves foo.yaml and foo.yml to one recipe (.yaml
        # wins), so listing "foo" twice would misdescribe what get() can fetch.
        return sorted({p.stem for p in root.iterdir()
                       if p.is_file() and p.suffix in RECIPE_EXTS})

    def recipe(self, name: str) -> dict:
        from .recipes import load_recipe
        return load_recipe(self.config, name)
