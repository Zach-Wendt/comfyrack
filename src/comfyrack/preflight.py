"""Validate a resolved graph against a target machine before queueing.

Without this, a missing custom node or an absent LoRA surfaces mid-batch as
`value_not_in_list` or `required_input_missing` after GPU time has been spent, and
a lineage mismatch never surfaces at all -- it just renders worse.

SCOPE, stated plainly so a reader does not mistake a boundary for a bug:

- Model presence is caught only where the loader exposes its filename as a COMBO
  widget, because /object_info's enumerated choices are what we compare against.
  Stock loaders (CheckpointLoaderSimple, LoraLoader, VAELoader, UpscaleModelLoader)
  all do. A custom node taking a model name as a plain STRING is invisible here.
- `lora_meta` has no source in ring 1, so `lineage_mismatch` is DORMANT until ring 2
  wires `lora_meta.scan()` into the runner. It is written and tested now so that
  wiring is a one-argument change rather than a new feature."""
from dataclasses import dataclass

from .graph import combo_choices
from .discover import ref_index, ref_lookup
from .errors import ComfyrackError


@dataclass
class Problem:
    kind: str        # missing_node_class | invalid_combo_value | lineage_mismatch
    node_id: str
    detail: str


def _combo_options(object_info_entry: dict) -> dict:
    out = {}
    spec = object_info_entry.get("input", {})
    for section in ("required", "optional"):
        for name, val in spec.get(section, {}).items():
            choices = combo_choices(val)
            if choices is not None:
                out[name] = choices
    return out


def check(graph, object_info: dict, lora_meta: dict | None = None,
          checkpoint_lineage: str | None = None, requires: dict | None = None) -> list:
    problems = []
    # A missing class usually means a missing pack. The manifest's `requires` names
    # the pack(s) and where to get them.
    # `requires.nodes` maps each class to its pack; when it names one, the hint names
    # only that pack.
    def label(p):
        return f"{p.get('name', '?')} ({p['git']})" if p.get("git") else str(p.get("name", "?"))
    pack_list = (requires or {}).get("custom_node_packs") or []
    by_name = {p.get("name"): p for p in pack_list}
    node_map = (requires or {}).get("nodes") or {}
    packs = ", ".join(label(p) for p in pack_list)
    all_hint = f"; this workflow needs: {packs}" if packs else ""

    def hint_for(class_type):
        owner = node_map.get(class_type)
        if owner in by_name:
            return f"; it comes from {label(by_name[owner])}"
        return all_hint
    # Indexed ONCE, and by reference rather than by raw string: a subfoldered
    # LoRA is `Anima/AnimaEditV1.safetensors` in one source and
    # `Anima\AnimaEditV1.safetensors` (or bare) in another, and a raw `in` test
    # made the lineage check silently skip exactly those 13-of-206 files instead
    # of firing on them.
    lora_index = ref_index(lora_meta) if lora_meta else {}
    for node_id, node in graph.data.items():
        class_type = node.get("class_type", "")
        if class_type not in object_info:
            problems.append(Problem("missing_node_class", node_id,
                                    f"node class {class_type!r} is not installed"
                                    f"{hint_for(class_type)}"))
            continue

        options = _combo_options(object_info[class_type])
        for param, value in (node.get("inputs") or {}).items():
            # A link is exactly [node_id, slot] pointing at a node that exists. Treating
            # ANY list as a link would silently skip multi-select combo widgets, whose
            # value is a JSON array -- a false negative in the very class of input this
            # module exists to catch.
            if (isinstance(value, list) and len(value) == 2
                    and str(value[0]) in graph.data):
                continue
            # A multiselect COMBO's value is a list of choices; check each one.
            picked = value if isinstance(value, list) else [value]
            if param in options and any(v not in options[param] for v in picked):
                shown = ", ".join(str(o) for o in options[param][:6])
                problems.append(Problem(
                    "invalid_combo_value", node_id,
                    f"{class_type}.{param}={value!r} is not available; options include: {shown}"))

            # Keyed on the VALUE being a known LoRA file, not on the param being called
            # `lora_name`. Multi-LoRA stacker nodes name their slots lora_01, lora_name_1
            # and so on; keying on the param name leaves every one of them unchecked.
            entry = (ref_lookup(lora_index, value)
                     if lora_index and checkpoint_lineage and isinstance(value, str)
                     else None)
            if entry is not None:
                lineage = entry.get("base_model_family")
                if lineage and lineage != checkpoint_lineage:
                    problems.append(Problem(
                        "lineage_mismatch", node_id,
                        f"LoRA {value!r} was trained on {lineage!r} but the checkpoint "
                        f"lineage is {checkpoint_lineage!r}; it will load and degrade "
                        f"silently rather than error"))
    return problems


def raise_if_problems(problems: list, machine_url: str) -> None:
    if not problems:
        return
    lines = "\n".join(f"  [{p.kind}] node {p.node_id}: {p.detail}" for p in problems)
    raise ComfyrackError(
        f"preflight failed against {machine_url} ({len(problems)} problem(s)):\n{lines}",
        help_text="comfyrack models --machine <name>  # see what that machine actually has",
    )
