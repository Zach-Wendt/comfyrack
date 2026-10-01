"""UI-format (GUI export) -> API-format conversion.

Superflow stores 69 workflows in UI format while this project's registry is
API format. Conversion needs /object_info because widgets_values is a positional
array whose names live only in the node class definition.

Fails loud on anything unrecognized: a half-converted graph submits successfully
and renders wrong, which is worse than an error."""
from .errors import ComfyrackError
from .graph import widget_names

# UI-only tokens ComfyUI writes into widgets_values immediately after a widget that
# declares a control_after_generate companion. They are not inputs on the API side.
#
# Membership in this set is NOT sufficient to skip a value: a legitimate combo or
# string widget can hold "fixed" or "randomize", and skipping it would shift every
# later value on that node by one -- the silent mis-assignment this module exists to
# prevent. The skip is gated on the PRECEDING widget actually declaring the companion.
CONTROL_TOKENS = {"randomize", "increment", "decrement", "fixed"}


def _has_control_companion(spec: dict, param: str) -> bool:
    """True when this widget declares a control_after_generate companion.

    ComfyUI attaches it to seed-like INT widgets and declares it in /object_info, so
    read it there rather than inferring from the value that happens to follow."""
    for section in ("required", "optional"):
        val = spec.get(section, {}).get(param)
        if isinstance(val, list) and len(val) > 1 and isinstance(val[1], dict):
            if val[1].get("control_after_generate"):
                return True
    return False


def _has_upload_companion(spec: dict, param: str) -> bool:
    """True when this widget declares an upload button (image_upload, video_upload...)."""
    for section in ("required", "optional"):
        val = spec.get(section, {}).get(param)
        if isinstance(val, list) and len(val) > 1 and isinstance(val[1], dict):
            if any(k.endswith("_upload") and v for k, v in val[1].items()):
                return True
    return False


def _ui_widget_names(node: dict) -> list[str]:
    """Widget names the node itself declares, in order (inputs that carry a `widget` key).

    The current frontend writes one such entry per widget, linked or not, and lists a
    dynamic combo's sub-widgets by their dotted name ("sampling_mode.top_k"). Older
    exports carry no `widget` key on plain widgets."""
    out = []
    for slot in node.get("inputs") or []:
        w = slot.get("widget")
        if isinstance(w, dict):
            out.append(w.get("name") or slot.get("name"))
    return out


def _known_input(spec: dict, name: str) -> bool:
    """True when the target ComfyUI declares this input. A dotted dynamic-combo name is
    known when its parent is."""
    base = name.split(".", 1)[0]
    return any(base in spec.get(sec, {}) for sec in ("required", "optional"))


def _dynamic_sub_inputs(spec: dict, param: str, chosen) -> list[str]:
    """Names of the sub-widgets a dynamic combo adds for the chosen option, else []."""
    for section in ("required", "optional"):
        val = spec.get(section, {}).get(param)
        if isinstance(val, list) and len(val) > 1 and isinstance(val[1], dict):
            for opt in val[1].get("options") or []:
                if isinstance(opt, dict) and opt.get("key") == chosen:
                    inp = opt.get("inputs") or {}
                    return [f"{param}.{n}" for sec in ("required", "optional")
                            for n in inp.get(sec, {})]
    return []


def _map_ui_widgets(nid, class_type, node, values, spec, linked):
    """Current-frontend mapping. Every declared widget keeps its slot in widgets_values,
    linked or not. Returns {input_name: value} for widgets that are not linked and that
    the target ComfyUI knows (a widget the target lacks is dropped, not sent)."""
    queue = _ui_widget_names(node)
    listed = set(queue)
    total = len(queue)
    inputs = {}
    vi = 0
    while queue:
        name = queue.pop(0)
        if vi >= len(values):
            raise ComfyrackError(
                f"node {nid} ({class_type}): {len(values)} widget value(s) for "
                f"{total} widget input(s); unfilled: {', '.join([name] + queue)}",
                help_text="re-export this workflow from ComfyUI -- the saved values "
                          "no longer match the node's current inputs",
            )
        value = values[vi]
        vi += 1
        if name not in linked and _known_input(spec, name):
            inputs[name] = value
        # A dynamic combo adds widget slots that depend on the chosen value. If the
        # export did not list them as inputs, take them from the node definition.
        extra = [n for n in _dynamic_sub_inputs(spec, name, value) if n not in listed]
        if extra:
            queue[:0] = extra
            listed.update(extra)
            total += len(extra)
        if (vi < len(values) and values[vi] in CONTROL_TOKENS
                and _has_control_companion(spec, name)):
            vi += 1
    return inputs


def is_ui_format(data: dict) -> bool:
    return isinstance(data, dict) and isinstance(data.get("nodes"), list)


def ui_to_api(data: dict, object_info: dict) -> dict:
    if not is_ui_format(data):
        raise ComfyrackError(
            "not a UI-format workflow: expected a top-level 'nodes' array",
            help_text="if this is already API format, load it directly",
        )

    # Schema-1 exports (version 1) carry links as objects, not the 6-element arrays
    # this converter reads; every object link would be silently dropped, producing a
    # graph with missing connections that still submits.
    if data.get("version") == 1:
        raise ComfyrackError(
            "UI format version 1 (object links) cannot be converted",
            help_text="re-export this workflow from ComfyUI with Workflow > Export (API)",
        )
    for link in data.get("links", []):
        if isinstance(link, dict):
            raise ComfyrackError(
                f"link {link.get('id', '?')} is an object link (schema 1); this "
                "converter only understands array links",
                help_text="re-export this workflow from ComfyUI with Workflow > Export (API)",
            )

    # Subgraphs and proxy widgets have no API-format equivalent this converter can
    # produce; emitting the nodes as-is would submit a different graph.
    if (data.get("definitions") or {}).get("subgraphs"):
        raise ComfyrackError(
            "workflow uses subgraphs, which cannot be converted",
            help_text="re-export this workflow from ComfyUI with Workflow > Export (API)",
        )
    for node in data["nodes"]:
        mode = node.get("mode")
        if mode in (2, 4):
            state = "muted" if mode == 2 else "bypassed"
            raise ComfyrackError(
                f"node {node.get('id')} ({node.get('type')}) is {state}; converting "
                "it as if active would produce a wrong graph",
                help_text="re-export this workflow from ComfyUI with Workflow > Export (API)",
            )
        if "proxyWidgets" in (node.get("properties") or {}):
            raise ComfyrackError(
                f"node {node.get('id')} ({node.get('type')}) has proxyWidgets, which "
                "cannot be converted",
                help_text="re-export this workflow from ComfyUI with Workflow > Export (API)",
            )

    # link id -> (origin_node_id, origin_slot)
    link_src = {}
    for link in data.get("links", []):
        if isinstance(link, list) and len(link) >= 5:
            link_id, origin_node, origin_slot = link[0], link[1], link[2]
            link_src[link_id] = (str(origin_node), origin_slot)

    api = {}
    for node in data["nodes"]:
        class_type = node.get("type")
        if class_type not in object_info:
            raise ComfyrackError(
                f"unknown node class {class_type!r} on node id {node.get('id')}",
                help_text="install the custom node on the target machine, "
                          "or run `comfyrack nodes` to see what is available",
            )
        nid = str(node["id"])
        inputs = {}

        # Positional widget values -> named inputs.
        values = list(node.get("widgets_values") or [])
        spec = object_info[class_type].get("input", {})
        current_frontend = bool(_ui_widget_names(node))
        if current_frontend:
            # Current frontend: the node names its own widgets, in order.
            linked_now = {s.get("name") for s in (node.get("inputs") or [])
                          if s.get("link") is not None}
            inputs.update(_map_ui_widgets(nid, class_type, node, values, spec,
                                          linked_now))
            names = []
        else:
            names = widget_names(object_info[class_type])

        # ComfyUI's "Convert Widget to Input" removes a widget from widgets_values and
        # adds a linked input slot in its place -- a very common pattern (routing a seed
        # from a Primitive node, say). Those names are NOT positional and must be
        # excluded before counting, or a perfectly valid export looks short by exactly
        # the number of converted widgets.
        linked = {s.get("name") for s in (node.get("inputs") or [])
                  if s.get("link") is not None}
        positional = [n for n in names if n not in linked]

        vi = 0
        for idx, name in enumerate(positional):
            if vi >= len(values):
                # Running out of values means the export is stale or the node class
                # changed. Emitting a node with unset inputs would submit and render
                # something plausible and wrong, so refuse.
                raise ComfyrackError(
                    f"node {nid} ({class_type}): {len(values)} widget value(s) for "
                    f"{len(positional)} positional widget input(s); unfilled: "
                    f"{', '.join(positional[idx:])}",
                    help_text="re-export this workflow from ComfyUI -- the saved values "
                              "no longer match the node's current inputs",
                )
            inputs[name] = values[vi]
            vi += 1
            # Skip the control_after_generate companion ONLY when this widget declares
            # one. Testing the value alone would eat a legitimate "fixed"/"randomize".
            if (vi < len(values) and values[vi] in CONTROL_TOKENS
                    and _has_control_companion(spec, name)):
                vi += 1

        # Legacy path only: values left over after every positional widget is filled
        # mean the export and the node definition disagree. Dropping them would submit
        # a node that looks fine but is not what the workflow author saved. Exception:
        # an input declaring `image_upload` (or video/audio) gets a UI-only upload
        # button whose value ("image") trails the real ones, e.g. LoadImage.
        uploads = sum(1 for n in positional if _has_upload_companion(spec, n))
        if not current_frontend and len(values) - vi > uploads:
            raise ComfyrackError(
                f"node {nid} ({class_type}): {len(values)} widget value(s) for "
                f"{len(positional)} positional widget input(s); unconsumed: "
                f"{', '.join(repr(v) for v in values[vi:])}",
                help_text="re-export this workflow from ComfyUI with Workflow > Export (API)",
            )

        # Linked inputs -> [origin_node_id, origin_slot].
        for slot in node.get("inputs") or []:
            link_id = slot.get("link")
            if link_id is not None and link_id in link_src:
                inputs[slot["name"]] = list(link_src[link_id])

        api[nid] = {
            "class_type": class_type,
            "inputs": inputs,
            "_meta": {"title": node.get("title") or class_type},
        }
    return api


def load_workflow(path, object_info=None):
    """Load a workflow JSON as a Graph, converting from UI format if needed.

    `object_info` may be a dict or a zero-arg callable returning one. The callable
    form exists so an API-format workflow -- which needs no node defs at all -- never
    triggers the multi-MB /object_info fetch."""
    import json
    from .graph import Graph

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if is_ui_format(data):
        if object_info is None:
            raise ComfyrackError(
                f"{path} is UI format and needs /object_info to convert",
                help_text="pass --machine so comfyrack can read the node definitions",
            )
        if callable(object_info):
            object_info = object_info()
        data = ui_to_api(data, object_info)
    return Graph(data).ensure_titles()
