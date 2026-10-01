"""Three-layer workflow registry.

Layers, lowest precedence first: builtin (shipped) -> user (~/.comfyrack/registry)
-> project (<root>/.comfyrack/registry). Workflows are shared; what a project owns
is recipes, so the project layer is usually small.

Scope affects DISPLAY only. Every layer always loads, load() resolves any name, and
find() searches everything -- scope only decides which families `list` expands,
because an unfiltered list of 100 manifests costs 4-6k tokens."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import ComfyrackError, NotFoundError

# Two layouts have to work. An installed wheel gets the corpus force-included at
# comfyrack/registry (see pyproject.toml); a src checkout keeps it at the repo root,
# outside the package. Prefer the package-adjacent copy and fall back, so neither
# layout silently reports "0 workflows found in any registry layer" -- _iter_paths
# skips a non-existent root without a word.
_PKG_REGISTRY = Path(__file__).resolve().parent / "registry"
_REPO_REGISTRY = Path(__file__).resolve().parent.parent.parent / "registry"
BUILTIN_DIR = _PKG_REGISTRY if _PKG_REGISTRY.is_dir() else _REPO_REGISTRY
USER_DIR = Path.home() / ".comfyrack" / "registry"
LAYER_ORDER = ("project", "user", "builtin")

# One concept, one name. Measured against the existing registry (2026-07-28), the
# positive prompt alone was already spelled three ways across six workflows
# (positive_text / prompt / positive_prompt), which means a recipe cannot cross
# families and a batch runner driving one spelling fails on the others.
#
# A manifest SHOULD key its flags with the canonical name. When it does not, the
# alias map lets callers use either spelling, so replacing a workflow does not
# invalidate the recipes written against it.
CANONICAL_ALIASES = {
    "positive_text": ["prompt", "positive_prompt", "positive"],
    "negative_text": ["negative", "negative_prompt"],
    "seed": ["noise_seed"],
    "steps": ["num_steps"],
    "cfg": ["cfg_scale", "guidance"],
    "denoise": ["denoise_strength"],
    "width": [],
    "height": [],
    "image": ["input_image", "source_image"],
    "filename_prefix": [],
    "sampler_name": ["sampler"],
    "scheduler": [],
    "unet_name": [],
    "clip_name": [],
    "vae_name": [],
    "id_lora_name": [],
    "id_strength_model": [],
    "id_strength_clip": [],
    "style_lora_name": [],
    "style_strength_model": [],
    "style_strength_clip": [],
}

_ALIAS_TO_CANONICAL = {alias: canon
                       for canon, aliases in CANONICAL_ALIASES.items()
                       for alias in aliases}


def canonical_of(name: str) -> str:
    """The canonical spelling of a flag name, or the name itself if it is already
    canonical or is not a known concept."""
    if name in CANONICAL_ALIASES:
        return name
    return _ALIAS_TO_CANONICAL.get(name, name)


@dataclass
class Flag:
    param: str
    node: str | None = None
    node_title: str | None = None
    type: str = "str"          # str | int | float | image
    required: bool = False
    default: object = None
    help: str = ""
    aliases: list = field(default_factory=list)
    # Optional form metadata. Static fallback only: `describe` overwrites these from
    # a live ComfyUI's /object_info when one answers.
    choices: list | None = None
    min: float | None = None
    max: float | None = None
    min_length: int | None = None
    max_length: int | None = None


FORM_KEYS = ("choices", "min", "max", "min_length", "max_length")


def lint_manifest(manifest) -> list:
    """Report flags keyed by an alias rather than the canonical name."""
    problems = []
    for key in sorted(manifest.flags):
        canon = canonical_of(key)
        if canon != key:
            problems.append(
                f"{manifest.family}/{manifest.name}: flag {key!r} should be keyed "
                f"{canon!r} (the canonical name for this concept)")
    return problems


@dataclass
class Family:
    name: str
    base_model_family: str = ""
    prompt_dialect: str = ""


@dataclass
class Manifest:
    name: str
    family: str
    layer: str
    description: str
    workflow_path: Path
    output_node_title: str
    output_type: str = "image"     # image | text | video | audio
    flags: dict = field(default_factory=dict)
    mode: list = field(default_factory=list)   # t2i | t2v | i2i | i2v | v2v | r2v | r2i
    # Extra results `run` returns, keyed by node title: [{"title", "kind"}], kind is
    # image | text. Empty means only output_node_title is returned.
    outputs: list = field(default_factory=list)
    # Free-form: custom_node_packs [{name, git}] and models [{directory, name}].
    requires: dict = field(default_factory=dict)

    def __post_init__(self):
        self.alias_map = self._build_alias_map()

    def _build_alias_map(self) -> dict:
        """Every name a caller might use -> the actual flag key.

        A name claimed by two flags is registered by neither: silently picking one
        would route a value to the wrong node."""
        claims: dict = {}
        for key, flag in self.flags.items():
            names = {canonical_of(key)}
            names.update(CANONICAL_ALIASES.get(canonical_of(key), []))
            names.update(flag.aliases or [])
            names.discard(key)
            for name in names:
                claims.setdefault(name, []).append(key)
        return {name: keys[0] for name, keys in claims.items()
                if len(keys) == 1 and name not in self.flags}

    def resolve_key(self, name: str) -> str:
        """Map any accepted spelling to this manifest's actual flag key. An unknown
        name is returned unchanged so the caller reports it with its own error."""
        if name in self.flags:
            return name
        return self.alias_map.get(name, name)


class Registry:
    def __init__(self, config, builtin_dir=None, user_dir=None):
        self.config = config
        project_dir = (config.project_root / ".comfyrack" / "registry"
                       if config.project_root else None)
        self._roots = {
            "builtin": Path(builtin_dir) if builtin_dir else BUILTIN_DIR,
            "user": Path(user_dir) if user_dir else USER_DIR,
            "project": project_dir,
        }

    # -- scanning ----------------------------------------------------------

    def _iter_paths(self):
        """Yields (layer, family, manifest_path) across all layers."""
        for layer in LAYER_ORDER:
            root = self._roots.get(layer)
            if not root or not root.is_dir():
                continue
            for family_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                for path in sorted(family_dir.glob("*.manifest.yaml")):
                    yield layer, family_dir.name, path

    def _all(self) -> dict:
        """name -> (layer, family, path), first (highest-precedence) layer wins."""
        out = {}
        for layer, family, path in self._iter_paths():
            name = path.name[: -len(".manifest.yaml")]
            out.setdefault(name, (layer, family, path))
        return out

    def shadowed(self) -> list:
        """(name, winning_layer, shadowed_layer) for every name in >1 layer."""
        seen, out = {}, []
        for layer, _family, path in self._iter_paths():
            name = path.name[: -len(".manifest.yaml")]
            if name in seen:
                out.append((name, seen[name], layer))
            else:
                seen[name] = layer
        return out

    # -- loading -----------------------------------------------------------

    def _parse(self, name, layer, family, path) -> Manifest:
        """A malformed manifest names itself.

        discover() parses EVERY manifest on every call, and cmd_home, cmd_list,
        cmd_find and Rack.list all route through it -- so an uncaught YAMLError or
        KeyError here took down the entire listing with a traceback that never said
        which of the 100+ files was bad. `onboard` writes more of them."""
        try:
            return self._parse_unguarded(name, layer, family, path)
        except (KeyError, yaml.YAMLError, TypeError, AttributeError) as exc:
            detail = (f"missing required key {exc.args[0]!r}"
                      if isinstance(exc, KeyError) else str(exc).strip())
            raise ComfyrackError(
                f"malformed manifest {path}: {detail}",
                help_text="a manifest needs `workflow:` and `output_node_title:`, and "
                          "each flag needs `param:`; fix that file or move it out of "
                          "the registry",
            ) from exc

    def _parse_unguarded(self, name, layer, family, path) -> Manifest:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        flags = {}
        for fname, fdef in (raw.get("flags") or {}).items():
            flags[fname] = Flag(
                param=fdef["param"],
                node=str(fdef["node"]) if fdef.get("node") is not None else None,
                node_title=fdef.get("node_title"),
                type=fdef.get("type", "str"),
                required=fdef.get("required", False),
                default=fdef.get("default"),
                help=fdef.get("help", ""),
                # `or []` matters: a bare `aliases:` with nothing after it parses as
                # None, and Task 8b's alias map calls set.update() on this.
                aliases=list(fdef.get("aliases") or []),
                **{k: fdef[k] for k in FORM_KEYS if fdef.get(k) is not None},
            )
        return Manifest(
            name=name, family=family, layer=layer,
            description=raw.get("description", ""),
            workflow_path=path.parent / raw["workflow"],
            output_node_title=raw["output_node_title"],
            output_type=raw.get("output", "image"),
            flags=flags,
            mode=list(raw.get("mode", [])),
            outputs=[{"title": o, "kind": "image"} if isinstance(o, str)
                     else {"title": o["title"], "kind": o.get("kind", "image")}
                     for o in (raw.get("outputs") or [])],
            requires=dict(raw.get("requires") or {}),
        )

    def discover(self, family: str | None = None) -> list:
        out = []
        for name, (layer, fam, path) in sorted(self._all().items()):
            if family is not None and fam != family:
                continue
            out.append(self._parse(name, layer, fam, path))
        return out

    def load(self, name: str) -> Manifest:
        entry = self._all().get(name)
        if entry is None and "/" in name:
            # `list` prints family/name; accept it wherever a bare name works.
            fam, _, bare = name.partition("/")
            hit = self._all().get(bare)
            if hit is not None and hit[1] == fam:
                entry, name = hit, bare
        if entry is None:
            known = sorted(self._all())
            close = [n for n in known if name in n or n in name][:5]
            hint = (f"did you mean: {', '.join(close)}?" if close
                    else "run `comfyrack list` to see registered workflows")
            raise NotFoundError(f"no workflow named {name!r}", help_text=hint)
        layer, fam, path = entry
        return self._parse(name, layer, fam, path)

    def find(self, query: str) -> list:
        q = query.lower()
        return [m for m in self.discover()
                if q in m.name.lower() or q in m.description.lower()
                or q in m.family.lower()]

    # -- families ----------------------------------------------------------

    def family(self, name: str) -> Family | None:
        for layer in LAYER_ORDER:
            root = self._roots.get(layer)
            if not root:
                continue
            path = root / name / "family.yaml"
            if path.is_file():
                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                return Family(name=name,
                              base_model_family=raw.get("base_model_family", ""),
                              prompt_dialect=raw.get("prompt_dialect", ""))
        return None

    def family_counts(self) -> list:
        """(family, count, note) with out-of-scope families collapsed to a hint."""
        scope = self.config.list_scope()
        counts = {}
        for m in self.discover():
            counts[m.family] = counts.get(m.family, 0) + 1
        rows = []
        for fam in sorted(counts):
            in_scope = (not scope) or fam in scope
            note = "active" if in_scope else f"(comfyrack list --family {fam})"
            rows.append((fam, counts[fam], note))
        return rows
