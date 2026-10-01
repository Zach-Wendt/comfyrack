"""API-format workflow graph operations.

Node addressing resolves by id first and falls back to _meta.title. Titles are
freely editable in the ComfyUI GUI, so a title-only join key breaks silently when
someone renames a node; storing and checking both makes the break loud."""
import copy
import hashlib
import json

from .errors import NotFoundError, UsageError


def combo_choices(spec) -> list | None:
    """Return the list of choices for a combo input spec, or ``None`` when
    *spec* is not a combo.

    Three shapes emitted by ``/object_info`` are recognised:

    - old list:: [[choice1, choice2, ...], {...opts}]
    - V3 COMBO:: ["COMBO", {"options": [choice1, ...], ...}]
    - dynamic combo:: ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "a", ...}, ...]}]
      whose choices are the option ``key`` values.

    Primitives (INT, FLOAT, STRING, BOOLEAN) and link types (MODEL, CLIP,
    LATENT...) return ``None``.
    """
    if not isinstance(spec, list) or not spec:
        return None
    t = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    # Old list shape: [[a, b, ...], {...}]
    if isinstance(t, list):
        return t
    if t in ("COMBO", "COMFY_DYNAMICCOMBO_V3"):
        options = opts.get("options")
        if not isinstance(options, list):
            return None
        if t == "COMBO":
            return options
        # Dynamic combo: options are dicts, choices are their "key" values.
        return [o["key"] for o in options
                if isinstance(o, dict) and "key" in o]
    return None


def widget_names(object_info_entry: dict) -> list[str]:
    """Widget input names for a node class, in declaration order.

    A widget is an input whose type is a primitive (INT/FLOAT/STRING/BOOLEAN) or a
    combo (a list of choices). Link inputs (MODEL, CLIP, LATENT...) are not widgets.
    Ported from Superflow's comfy_facade.get_widget_keys; needed for UI->API
    conversion and for inferring manifest flags at onboard time."""
    spec = object_info_entry.get("input", {})
    out = []
    for section in ("required", "optional"):
        for name, val in spec.get(section, {}).items():
            if not isinstance(val, list) or not val:
                continue
            t = val[0]
            if t in ("INT", "FLOAT", "STRING", "BOOLEAN") or combo_choices(val) is not None:
                out.append(name)
    return out


class Graph:
    def __init__(self, data: dict):
        self.data = data

    def copy(self) -> "Graph":
        return Graph(copy.deepcopy(self.data))

    def title_of(self, node_id: str) -> str:
        node = self.data.get(node_id, {})
        return node.get("_meta", {}).get("title") or node_id

    def resolve(self, node_id: str | None = None, title: str | None = None) -> str:
        if node_id is not None and node_id in self.data:
            return node_id
        if title is not None:
            matches = [nid for nid in self.data if self.title_of(nid) == title]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise NotFoundError(
                    f"node title {title!r} is ambiguous: matches ids {', '.join(sorted(matches))}",
                    help_text="give the manifest flag an explicit `node:` id",
                )
        want = title if title is not None else node_id
        raise NotFoundError(
            f"no node matching id={node_id!r} title={title!r}",
            help_text=f"nodes present: {', '.join(sorted(self.title_of(n) for n in self.data))}"
            if want else None,
        )

    def set_param(self, node_id: str, param: str, value) -> None:
        inputs = self.data[node_id].setdefault("inputs", {})
        if param not in inputs:
            raise UsageError(
                f"node {node_id} ({self.title_of(node_id)}) has no input {param!r}",
                help_text=f"valid inputs: {', '.join(sorted(inputs))}",
            )
        inputs[param] = value

    def get_param(self, node_id: str, param: str):
        return self.data[node_id].get("inputs", {}).get(param)

    def nodes(self) -> list:
        return [(nid, self.title_of(nid), n.get("class_type", ""))
                for nid, n in self.data.items()]

    def class_types(self) -> set:
        return {n.get("class_type", "") for n in self.data.values()}

    def ensure_titles(self) -> "Graph":
        for nid, node in self.data.items():
            node.setdefault("_meta", {}).setdefault("title", nid)
        return self

    def content_hash(self) -> str:
        payload = json.dumps(self.data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
