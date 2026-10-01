"""Shot-list schema.

Named `shots` rather than after one project's installment vocabulary (chapter,
episode, song), because the shot list is the generic unit. The sequence number
comes from the `sequence:` key."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import UsageError


@dataclass
class Shot:
    name: str = ""
    character: str = ""
    people: str = ""
    positive_prefix: str = ""
    action_text: str = ""
    location_text: str = ""
    seed: int | None = None
    id_strength: float | None = None
    extra_negative: str = ""
    via: str | None = None
    base: dict = field(default_factory=dict)
    composite: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


    @property
    def subject(self) -> str:
        return self.character

    @subject.setter
    def subject(self, value: str) -> None:
        self.character = value

    @property
    def scene_text(self) -> str:
        return self.location_text

    @scene_text.setter
    def scene_text(self, value: str) -> None:
        self.location_text = value


@dataclass
class ShotList:
    sequence: object = None
    primary_character: str = ""
    machine: str | None = None
    extra_negative: str = ""
    characters: dict = field(default_factory=dict)
    shots: list = field(default_factory=list)
    path: Path | None = None

    @property
    def primary_subject(self) -> str:
        return self.primary_character

    @primary_subject.setter
    def primary_subject(self, value: str) -> None:
        self.primary_character = value

    @property
    def subjects(self) -> dict:
        return self.characters

    @subjects.setter
    def subjects(self, value: dict) -> None:
        self.characters = value


def _normalize_characters(value) -> dict:
    """`characters:` is a mapping of name -> {text: "<prose>"} in most
    shot lists, absent (parses as None) in the rest, and never a list in practice -- but a plain list of names costs nothing to tolerate, so accept it
    as a per-name dict of empty descriptions rather than rejecting it."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {name: {} for name in value}
    raise UsageError(
        f"`characters:` is a {type(value).__name__}, not a mapping",
        help_text="`characters:` should be a mapping of name to {text: ...}",
    )


def _normalize_composite(value) -> dict:
    """`composite:` is a mapping -- seed, refs, extra_positive, extra_negative --
    on `via: shot_builder` shots, absent on plain shots.
    Always returns a dict with `refs` (list) and `seed` present so callers never
    have to special-case a missing composite."""
    if value is None:
        value = {}
    elif not isinstance(value, dict):
        raise UsageError(
            f"`composite:` is a {type(value).__name__}, not a mapping",
            help_text="`composite:` should be a mapping with `refs:` and `seed:`",
        )
    normalized = dict(value)
    normalized["refs"] = normalized.get("refs") or []
    normalized.setdefault("seed", None)
    return normalized


def _str_field(field_name: str, src: dict, raw: dict, default: str = "") -> str:
    """Read a string prompt field through `src` (the shot itself, or `base` for a
    `via` shot) and fall back to the top-level shot dict if it is absent there --
    a hand-authored `via` shot may put any of these fields top-level instead of
    under `base`, and every prompt field should tolerate that identically."""
    return src.get(field_name, default) or raw.get(field_name, default)


def _val_field(field_name: str, src: dict, raw: dict):
    """Same read-through as `_str_field`, but for fields where `None` (not falsy
    zero) is the real "absent" sentinel -- seed and id_strength."""
    val = src.get(field_name)
    return val if val is not None else raw.get(field_name)


def _shot_from(raw: dict) -> Shot:
    base = raw.get("base") or {}
    # A shot_builder shot keeps its prompt fields on `base`; read through so callers
    # do not have to special-case the two shapes.
    src = base if raw.get("via") else raw
    composite = _normalize_composite(raw.get("composite"))
    # `Shot.seed` is one seed a caller can read without knowing the three places
    # it can live. It stays the shot's own seed (top-level, or `base` for a
    # `via` shot) whenever there is one -- that is the seed of the shot's own
    # render stage, and it must never be shadowed by the composite step's seed.
    # It falls back to `composite.seed` only when the shot truly has none of its
    # own, which happens for exactly one shape:
    # `base.kind: location_plate` shots have no native-render stage at all, so
    # `composite.seed` is the only seed that governs their render, not a second,
    # different one.
    seed = _val_field("seed", src, raw)
    if seed is None:
        seed = composite.get("seed")
    return Shot(
        name=raw.get("name", ""),
        character=_str_field("subject", src, raw) or _str_field("character", src, raw),
        people=_str_field("people", src, raw),
        positive_prefix=_str_field("positive_prefix", src, raw),
        action_text=_str_field("action_text", src, raw),
        location_text=_str_field("scene_text", src, raw) or _str_field("location_text", src, raw) or _str_field("scene", src, raw),
        seed=seed,
        id_strength=_val_field("id_strength", src, raw),
        extra_negative=_str_field("extra_negative", src, raw),
        via=raw.get("via"),
        base=base,
        composite=composite,
        raw=raw,
    )


def load_shots(path) -> ShotList:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise UsageError(
            f"{path.name} is not a shot list: document root is a "
            f"{type(raw).__name__}, not a mapping",
            help_text="a shot list file must be a YAML mapping with a `shots:` key",
        )
    if "shots" not in raw:
        raise UsageError(
            f"{path.name} has no `shots:` key",
            help_text="a shot list needs a top-level `shots:` array",
        )
    shots_raw = raw["shots"]
    if not isinstance(shots_raw, list):
        raise UsageError(
            f"{path.name}'s `shots:` key is a {type(shots_raw).__name__}, not a list",
            help_text="each entry under `shots:` must be a mapping for one shot",
        )
    return ShotList(
        sequence=raw.get("sequence"),
        primary_character=raw.get("primary_subject") if "primary_subject" in raw else raw.get("primary_character", ""),
        machine=raw.get("comfy_url"),
        extra_negative=raw.get("extra_negative", "") or "",
        characters=_normalize_characters(raw.get("subjects") if "subjects" in raw else raw.get("characters")),
        shots=[_shot_from(s) for s in shots_raw],
        path=path,
    )


def validate(shot_list: ShotList) -> list:
    problems, seen = [], set()
    for i, s in enumerate(shot_list.shots):
        label = s.name or f"shot #{i}"
        if not s.name:
            problems.append(f"{label}: missing `name`")
        elif s.name in seen:
            problems.append(f"{s.name}: duplicate shot name")
        else:
            seen.add(s.name)
        if s.seed is None:
            problems.append(f"{label}: missing `seed` (renders would not be reproducible)")
        # A `location_plate` base composites straight onto a pre-rendered plate --
        # there is no text-conditioned render stage, so it has no `action_text`
        # by design. Every other base kind, including a `via` shot's default
        # `native` base, still needs one.
        if s.base.get("kind") != "location_plate" and not s.action_text:
            problems.append(f"{label}: missing `action_text`")
        if s.via and not s.composite.get("refs"):
            problems.append(f"{label}: `via: {s.via}` shot has no `composite.refs`")
    return problems
