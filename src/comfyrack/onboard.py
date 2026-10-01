"""Register a workflow JSON into a registry layer.

Flags are INFERRED from /object_info rather than hand-authored: every widget input
on every node becomes a candidate flag defaulting to the value already baked into
the graph, which is why `required` is rare. A human then trims the result."""
import json
import re
import shutil
from pathlib import Path

import yaml

from .config import DEFAULT_URL
from .convert import is_ui_format
from .errors import ComfyrackError
from .graph import Graph, widget_names

_TYPE_MAP = {"INT": "int", "FLOAT": "float", "STRING": "str", "BOOLEAN": "str"}
OUTPUT_CLASSES = ("SaveImage", "SaveAnimatedWEBP", "VHS_VideoCombine",
                  "SaveAudio", "PreviewImage", "SaveText", "Save Text File")

CONFIG_TEMPLATE = """[machines]
default = "{url}"

[list]
scope = []

[paths]
results = ".comfyrack/results"
recipes = ".comfyrack/recipes"
lexicon = ".comfyrack/lexicon"
subjects = ".comfyrack/subjects"
scenes = ".comfyrack/scenes"
"""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def infer_flags(graph: Graph, object_info: dict) -> dict:
    """Widget inputs across the whole graph -> candidate flags.

    A param name used by more than one node is prefixed with that node's title, so
    a two-sampler workflow yields base_seed and refiner_seed rather than a collision."""
    seen = {}
    for node_id, _title, class_type in graph.nodes():
        entry = object_info.get(class_type)
        if not entry:
            continue
        for param in widget_names(entry):
            if param not in (graph.data[node_id].get("inputs") or {}):
                continue
            seen.setdefault(param, []).append((node_id, class_type))

    flags = {}
    for param, holders in seen.items():
        for node_id, class_type in holders:
            title = graph.title_of(node_id)
            key = param if len(holders) == 1 else f"{_slug(title)}_{param}"
            spec = object_info[class_type].get("input", {})
            declared = None
            for section in ("required", "optional"):
                if param in spec.get(section, {}):
                    declared = spec[section][param][0]
            ftype = "str" if isinstance(declared, list) else _TYPE_MAP.get(declared, "str")
            flags[key] = {"node": node_id, "node_title": title, "param": param,
                          "type": ftype, "required": False}
    return flags


def _find_output_title(graph: Graph, object_info: dict) -> str:
    """The node whose output the run produces.

    A class counts as an output when it is a known saver/preview class OR its
    /object_info entry declares output_node: true -- custom nodes (ImageSaver and
    friends) are outputs without being in the hardcoded list."""
    def _is_output(class_type: str) -> bool:
        if class_type in OUTPUT_CLASSES:
            return True
        entry = object_info.get(class_type)
        return bool(entry) and entry.get("output_node") is True

    candidates = [(nid, title) for nid, title, ct in graph.nodes() if _is_output(ct)]
    if len(candidates) == 1:
        return candidates[0][1]
    if not candidates:
        raise ComfyrackError(
            "could not identify an output node in this workflow",
            help_text="set output_node_title by hand in the generated manifest",
        )
    saves = [t for nid, t in candidates]
    raise ComfyrackError(
        f"multiple output nodes found: {', '.join(saves)}",
        help_text="set output_node_title by hand to pick one",
    )


def _load_source(src_path: Path):
    """Read and parse the source workflow JSON, turning the two normal ways this
    fails -- a missing file, or content that isn't JSON -- into a ComfyrackError
    with a help line, instead of letting a bare traceback escape to stderr. This
    is the NORMAL failure mode for onboarding arbitrary external workflow files."""
    try:
        raw_text = src_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ComfyrackError(
            f"cannot read workflow file {src_path}: {e}",
            help_text="pass the path to an existing ComfyUI workflow JSON "
                      "exported as API format",
        ) from e
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ComfyrackError(
            f"{src_path} is not valid JSON: {e}",
            help_text="pass the path to a ComfyUI workflow JSON exported as API "
                      "format -- not some other file",
        ) from e


def onboard(src_path, dest_dir, name: str, object_info: dict, force: bool = False,
            ui_path=None, comfy_dir=None, lookups=None):
    """`ui_path` is the same workflow's UI-format export. It is optional and is read only
    for the sources it carries (model URLs, pack IDs) in `requires`."""
    from . import deps

    src_path, dest_dir = Path(src_path), Path(dest_dir)
    data = _load_source(src_path)
    ui = _load_source(Path(ui_path)) if ui_path else None
    # API format only. UI format changes with every frontend release, so converting
    # it outside the browser drifts; refusing beats registering a wrong graph.
    if is_ui_format(data):
        raise ComfyrackError(
            f"{src_path} is UI format; onboard takes API format only",
            help_text="open the workflow in ComfyUI and use Workflow > Export (API)",
        )
    graph = Graph(data).ensure_titles()

    dest_dir.mkdir(parents=True, exist_ok=True)
    json_path = dest_dir / f"{name}.json"
    manifest_path = dest_dir / f"{name}.manifest.yaml"
    if not force:
        existing = next((p for p in (json_path, manifest_path) if p.exists()), None)
        if existing is not None:
            raise ComfyrackError(
                f"{existing} already exists",
                help_text="pass --force to overwrite",
            )

    shutil.copy(src_path, json_path)

    manifest = {
        "description": f"TODO: describe {name}",
        "workflow": f"{name}.json",
        "output_node_title": _find_output_title(graph, object_info),
        "output": "image",
        "flags": infer_flags(graph, object_info),
    }
    requires, _unknowns = deps.record(graph, object_info, ui=ui, comfy_dir=comfy_dir,
                                      lookups=lookups)
    manifest["requires"] = requires
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False),
                             encoding="utf-8")
    return json_path, manifest_path


def init_project(root, url: str = DEFAULT_URL) -> Path:
    """Write .comfyrack/config.toml. Idempotent: an existing config is left alone."""
    path = Path(root) / ".comfyrack" / "config.toml"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE.format(url=url), encoding="utf-8")
    return path
