"""Recipes and flag value resolution.

A manifest describes WHAT KNOBS EXIST (generic, follows the graph). A recipe
describes WHAT VALUES YOU USE (project-owned). Keeping them separate is what lets
a shared workflow be tuned per project without forking it.

Value resolution, highest precedence first:
    --set override  ->  recipe block  ->  manifest default  ->  value baked into
    the workflow JSON."""
from pathlib import Path

import yaml

from .errors import NotFoundError, UsageError

_COERCE = {"int": int, "float": float, "str": str, "image": str}


def parse_set(pairs, manifest) -> dict:
    """Parse `--set k=v` pairs into typed values using the manifest's flag types."""
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise UsageError(
                f"malformed --set {pair!r}: expected key=value",
                help_text=f"valid keys: {', '.join(sorted(manifest.flags))}",
            )
        key, raw = pair.split("=", 1)
        key = manifest.resolve_key(key)
        if key not in manifest.flags:
            raise UsageError(
                f"unknown key {key!r} for workflow {manifest.name!r}",
                help_text=f"valid keys: {', '.join(sorted(manifest.flags))}",
            )
        ftype = manifest.flags[key].type
        try:
            out[key] = _COERCE.get(ftype, str)(raw)
        except (TypeError, ValueError):
            raise UsageError(
                f"--set {key}={raw!r} is not a valid {ftype}",
                help_text=f"comfyrack describe {manifest.name}",
            )
    return out


def resolve_values(manifest, graph, recipe=None, overrides=None) -> dict:
    """Final value per flag. Missing required flags are reported together, so one
    fix pass resolves them all rather than one failed call per flag.

    An override naming a flag this manifest does not declare is an ERROR, the same
    way `parse_set()` treats one. The loop below only ever iterates the manifest's
    own flags, so an unrecognised -- or simply misspelled -- override key used to
    fall off the end of it and vanish. That is the failure mode that let a batch
    render every shot with its identity LoRA at the workflow's baked-in 0.0 while
    reporting `9/9 ok`: the caller believed it had set a value, the resolver never
    saw the key, and nothing anywhere said otherwise.

    Recipe keys are deliberately NOT held to the same rule. A recipe is a
    project-owned block of values that is routinely shared across sibling
    workflows in a family (an `anima_v1` recipe feeding character-scene AND
    harmonize), so a key one of them does not declare is normal. An override, by
    contrast, is a per-call instruction aimed at THIS workflow -- if it does not
    land, the caller has to hear about it."""
    recipe = {manifest.resolve_key(k): v for k, v in (recipe or {}).items()}
    overrides = {manifest.resolve_key(k): v for k, v in (overrides or {}).items()}

    unknown = sorted(k for k in overrides if k not in manifest.flags)
    if unknown:
        raise UsageError(
            f"unknown override key(s) for workflow {manifest.name!r}: "
            f"{', '.join(unknown)}",
            help_text=f"valid keys: {', '.join(sorted(manifest.flags))}",
        )

    values, missing = {}, []

    for name, flag in manifest.flags.items():
        if name in overrides:
            values[name] = overrides[name]
            continue
        if name in recipe:
            values[name] = recipe[name]
            continue
        if flag.default is not None:
            values[name] = flag.default
            continue
        try:
            node_id = graph.resolve(node_id=flag.node, title=flag.node_title)
            baked = graph.get_param(node_id, flag.param)
        except Exception:
            baked = None
        if baked is not None:
            values[name] = baked
        elif flag.required:
            missing.append(name)

    if missing:
        raise UsageError(
            f"missing required flags for {manifest.name!r}: {', '.join(sorted(missing))}",
            help_text=f"comfyrack describe {manifest.name}",
        )
    return values


def load_recipe(config, name: str) -> dict:
    recipes_dir = config.path("recipes")
    if recipes_dir is None:
        recipes_dir = ((config.project_root / ".comfyrack" / "recipes")
                       if config.project_root else Path.cwd() / ".comfyrack" / "recipes")
    for ext in (".yaml", ".yml"):
        path = recipes_dir / f"{name}{ext}"
        if path.is_file():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raise NotFoundError(
        f"no recipe named {name!r}",
        help_text=f"searched {recipes_dir}; run `comfyrack recipe list`",
    )
