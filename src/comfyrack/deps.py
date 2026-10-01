"""Where each node class and model file of a workflow comes from.

`record()` builds a manifest's `requires` block (used by `onboard --ui` and `deps backfill`).
`plan()` compares that block with a machine and lists what is missing; `apply()` hands the
missing packs and models to ComfyUI-Manager over HTTP (`comfyrack setup`).

Sources, strongest first. Nodes: the live /object_info, checked against the pack folder on
disk when a ComfyUI path is given; then the comfy registry, the UI export's
`cnr_id`/`aux_id`, and Manager's extension-node-map. Models: the UI export's
`properties.models`, the existing record, Manager's model-list, then one Hugging Face lookup.

`python_module` in /object_info is not trusted alone. A pack that re-exports ComfyUI's
global NODE_CLASS_MAPPINGS makes ComfyUI tag every class loaded before it with that pack's
name (seen with comfyui-workflow-encrypt), and a frontend UI export copies the same wrong
name into `cnr_id`. So a folder counts only if its source names the class, and the
registry's class lookup wins over a UI `cnr_id`."""
import re
import subprocess
import time
import tomllib
import uuid
from pathlib import Path

import requests

from .discover import normalize_ref
from .preflight import _combo_options

CORE = "core"
UNKNOWN = "unknown"
MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")
CORE_MODULES = ("nodes", "comfy_extras", "comfy_api_nodes")
REGISTRY_NODE = "https://api.comfy.org/comfy-nodes/{}/node"
MANAGER_RAW = "https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main/"
HF_API = "https://huggingface.co/api/models"
# Loader input name -> models/ folder, for a model no other source places.
PARAM_DIRS = {"ckpt_name": "checkpoints", "unet_name": "diffusion_models",
              "clip_name": "text_encoders", "clip_name1": "text_encoders",
              "clip_name2": "text_encoders", "vae_name": "vae", "lora_name": "loras",
              "control_net_name": "controlnet", "model_name": "upscale_models"}


def _get_json(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


class Lookups:
    """Network lookups, each fetched at most once per run. Tests pass a fake `get_json`."""

    def __init__(self, get_json=None):
        self.get_json = get_json or (lambda url, params=None: _get_json(url, params))
        self._cache = {}

    def _once(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def registry_pack(self, class_type):
        """{registry_id, git} for the registry pack that defines class_type, or None."""
        d = self._once(("reg", class_type),
                       lambda: self.get_json(REGISTRY_NODE.format(class_type)))
        if not isinstance(d, dict) or not d.get("id"):
            return None
        return {"registry_id": d["id"], "git": _clean_git(d.get("repository"))}

    def node_map_repos(self, class_type):
        m = self._once("enm", lambda: self.get_json(MANAGER_RAW + "extension-node-map.json")) or {}
        return [_clean_git(url) for url, v in m.items()
                if isinstance(v, list) and v and class_type in v[0]]

    def model_list(self):
        d = self._once("ml", lambda: self.get_json(MANAGER_RAW + "model-list.json")) or {}
        return d.get("models") or []

    def model_list_entry(self, filename):
        return next((m for m in self.model_list() if m.get("filename") == filename), None)

    def hf_file(self, filename, directory=None):
        """(repo, path) of a Hugging Face repo holding exactly `filename`. Comfy-Org's
        repackaged repos first, then a search by the filename stem."""
        comfy = self._once("hf-comfy", lambda: self.get_json(
            HF_API, {"author": "Comfy-Org", "full": "true", "limit": 1000})) or []
        stem = Path(filename).stem
        found = _hf_match(comfy, filename, directory)
        if found:
            return found
        # A search on the full stem can match only a re-upload named after the file
        # (seen: a 0-download mirror of lightx2v's Lightning LoRA). HF searches repo
        # IDs, so shorter stem prefixes reach the original. Stop at the first query
        # that finds the file in a repo someone downloads; the most-downloaded wins.
        tokens = re.split(r"[-_.]", stem)
        queries = [stem] + ["-".join(tokens[:n]) for n in range(len(tokens) - 1, 1, -1)]
        pool, best = {}, None
        for q in queries[:6]:
            for m in self._once(("hf", q), lambda q=q: self.get_json(
                    HF_API, {"search": q, "full": "true", "sort": "downloads",
                             "limit": 20})) or []:
                if isinstance(m, dict) and m.get("id"):
                    pool.setdefault(m["id"], m)
            ranked = sorted(pool.values(), key=lambda m: -(m.get("downloads") or 0))
            best = _hf_match(ranked, filename, directory)
            if best and (pool[best[0]].get("downloads") or 0) > 0:
                break
        return best


def _hf_match(repos, filename, directory):
    hits = [(m["id"], s["rfilename"]) for m in repos if isinstance(m, dict)
            for s in m.get("siblings") or [] if s.get("rfilename", "").split("/")[-1] == filename]
    if directory:
        # Several Comfy-Org repos carry the same file; prefer the one filed under its folder.
        hits.sort(key=lambda h: f"{directory}/" not in h[1])
    return hits[0] if hits else None


def _clean_git(url):
    if not url:
        return None
    url = url.strip().rstrip("/")
    return url[:-4] if url.endswith(".git") else url


# -- UI exports --------------------------------------------------------------

def ui_nodes(ui: dict) -> list:
    """Every node in a UI-format export, subgraph definitions included."""
    nodes = list(ui.get("nodes") or [])
    for sub in (ui.get("definitions") or {}).get("subgraphs") or []:
        nodes.extend(sub.get("nodes") or [])
    return nodes


def ui_sources(ui: dict | None):
    """({class: properties}, [{name, url, directory}]) from a UI export."""
    classes, models = {}, []
    for node in ui_nodes(ui or {}):
        props = node.get("properties") or {}
        if node.get("type"):
            classes.setdefault(node["type"], props)
        for m in props.get("models") or []:
            if m.get("name") and m not in models:
                models.append(m)
    return classes, models


# -- nodes -------------------------------------------------------------------

def _folder_defines(folder: Path, class_type: str) -> bool:
    pat = re.compile(r"""["']""" + re.escape(class_type) + r"""["']""")
    for py in folder.rglob("*.py"):
        if any(part in (".git", "__pycache__", "node_modules", ".venv") for part in py.parts):
            continue
        try:
            if pat.search(py.read_text(encoding="utf-8", errors="ignore")):
                return True
        except OSError:
            continue
    return False


def folder_pack(folder: Path) -> dict:
    """Pack record read from an installed folder. A git clone gives its remote; a registry
    install has no .git (git would answer with ComfyUI's own remote) but has pyproject.toml."""
    pack = {"name": folder.name}
    if (folder / ".git").exists():
        try:
            url = subprocess.run(["git", "-C", str(folder), "remote", "get-url", "origin"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            url = ""
        if url:
            pack["git"] = _clean_git(url)
    pyproject = folder / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError):
            data = {}
        project = data.get("project") or {}
        if "comfy" in (data.get("tool") or {}) and project.get("name"):
            pack["registry_id"] = project["name"]
        repo = (project.get("urls") or {}).get("Repository")
        if repo and "git" not in pack:
            pack["git"] = _clean_git(repo)
    return pack


def _find_folder(custom_nodes: Path, hint: str | None, class_type: str):
    if hint and (custom_nodes / hint).is_dir() and _folder_defines(custom_nodes / hint, class_type):
        return custom_nodes / hint
    for folder in sorted(p for p in custom_nodes.iterdir() if p.is_dir()):
        if folder.name != hint and not folder.name.startswith(".") and folder.name != "__pycache__" \
                and _folder_defines(folder, class_type):
            return folder
    return None


def resolve_node(class_type, object_info, ui_classes, custom_nodes, lookups):
    """(CORE | pack dict | None) for one class."""
    entry = object_info.get(class_type)
    hint = None
    if entry is not None:
        module = entry.get("python_module") or ""
        if module.split(".")[0] in CORE_MODULES:
            return CORE
        if module.startswith("custom_nodes."):
            hint = module.split(".", 1)[1]
    ui = ui_classes.get(class_type) or {}
    if entry is None and ui.get("cnr_id") == "comfy-core":
        return CORE
    if custom_nodes is not None and custom_nodes.is_dir():
        folder = _find_folder(custom_nodes, hint, class_type)
        if folder is not None:
            return folder_pack(folder)
    reg = lookups.registry_pack(class_type)
    if reg:
        return {"name": _repo_name(reg["git"]) or reg["registry_id"], **_drop_none(reg)}
    if ui.get("aux_id"):
        git = f"https://github.com/{ui['aux_id']}"
        return {"name": _repo_name(git), "git": git}
    if ui.get("cnr_id") and ui["cnr_id"] != "comfy-core":
        return {"name": ui["cnr_id"], "registry_id": ui["cnr_id"]}
    repos = lookups.node_map_repos(class_type)
    if hint:
        repos = [r for r in repos if _repo_name(r).lower() == hint.lower()] or repos
    if len(repos) == 1:
        return {"name": _repo_name(repos[0]), "git": repos[0]}
    return None


def _repo_name(git):
    return git.rstrip("/").split("/")[-1] if git else None


def _drop_none(d):
    return {k: v for k, v in d.items() if v is not None}


# -- models ------------------------------------------------------------------

def is_model_ref(value) -> bool:
    return isinstance(value, str) and value.lower().endswith(MODEL_EXTS)


def graph_models(graph) -> list:
    """[(class_type, param, value)] for every model filename a graph loads."""
    out = []
    for node in graph.data.values():
        for param, value in (node.get("inputs") or {}).items():
            if is_model_ref(value):
                out.append((node.get("class_type", ""), param, value))
    return out


def resolve_model(value, param, ui_models, existing, lookups):
    """{directory, name, url} or {directory, name, source: unknown}."""
    name = normalize_ref(value)
    base = name.split("/")[-1]
    ui = next((m for m in ui_models if normalize_ref(m["name"]).split("/")[-1] == base), None)
    old = next((m for m in existing if normalize_ref(m.get("name")).split("/")[-1] == base), None)
    directory = ((ui or {}).get("directory") or (old or {}).get("directory")
                 or PARAM_DIRS.get(param))
    url = (ui or {}).get("url") or (old or {}).get("url")
    if not url:
        entry = lookups.model_list_entry(base)
        if entry:
            url = entry.get("url")
            directory = directory or (entry.get("save_path") or "").split("/")[0] or None
    if not url:
        hit = lookups.hf_file(base, directory)
        if hit:
            url = f"https://huggingface.co/{hit[0]}/resolve/main/{hit[1]}"
            parent = hit[1].split("/")[-2] if "/" in hit[1] else None
            directory = directory or parent
    rec = {"directory": directory or UNKNOWN, "name": name}
    if url:
        rec["url"] = url
    else:
        rec["source"] = UNKNOWN
    return rec


# -- the record --------------------------------------------------------------

def record(graph, object_info, ui=None, comfy_dir=None, existing=None, lookups=None):
    """(requires, unknowns). `requires` keeps any keys of `existing` this does not own.
    `unknowns` lists every class and model no source could place, for the user."""
    lookups = lookups or Lookups()
    existing = dict(existing or {})
    ui_classes, ui_models = ui_sources(ui)
    custom_nodes = Path(comfy_dir) / "custom_nodes" if comfy_dir else None
    old_packs = {p.get("name"): p for p in existing.get("custom_node_packs") or []}

    nodes, packs, unknowns = {}, {}, []
    for class_type in sorted({n.get("class_type", "") for n in graph.data.values()}):
        found = resolve_node(class_type, object_info, ui_classes, custom_nodes, lookups)
        if found is None:
            nodes[class_type] = UNKNOWN
            unknowns.append(f"node {class_type}")
        elif found == CORE:
            nodes[class_type] = CORE
        else:
            nodes[class_type] = found["name"]
            pack = packs.setdefault(found["name"], {"name": found["name"]})
            for key in ("git", "registry_id"):
                value = found.get(key) or (old_packs.get(found["name"]) or {}).get(key)
                if value:
                    pack.setdefault(key, value)

    models = []
    for _class, param, value in graph_models(graph):
        rec = resolve_model(value, param, ui_models, existing.get("models") or [], lookups)
        if rec["name"] in {m["name"] for m in models}:
            continue
        models.append(rec)
        if rec.get("source") == UNKNOWN:
            unknowns.append(f"model {rec['name']}")
    # A model the graph names in no filename input (a pack that loads it by itself) is
    # kept from the existing record: only a person could have written it there.
    seen = {m["name"].split("/")[-1] for m in models}
    models += [m for m in existing.get("models") or []
               if normalize_ref(m.get("name")).split("/")[-1] not in seen]

    out = {"nodes": nodes, "custom_node_packs": list(packs.values()), "models": models}
    for key, value in existing.items():
        out.setdefault(key, value)
    if not out["custom_node_packs"]:
        del out["custom_node_packs"]
    if not out["models"]:
        del out["models"]
    return out, unknowns


# -- the plan ----------------------------------------------------------------

def _head(url):
    return requests.head(url, allow_redirects=True, timeout=30)


def head_size(url, head=None) -> int | None:
    """Bytes behind a download URL from one HTTP HEAD, or None."""
    head = head or _head
    try:
        r = head(url)
        size = r.headers.get("x-linked-size") or r.headers.get("content-length")
        return int(size) if size else None
    except (requests.RequestException, ValueError, AttributeError):
        return None


def human_size(n) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def plan(manifest, graph, client, head=None) -> dict:
    """What the machine behind `client` lacks for this workflow.

    Packs: graph classes absent from /object_info, mapped through `requires.nodes`.
    Models: preflight's combo check on loaders that exist; for a loader whose pack is
    missing there is no combo to check, so its file is looked up in /models/<folder>."""
    object_info = client.object_info()
    requires = manifest.requires or {}
    node_map = requires.get("nodes") or {}
    pack_by_name = {p.get("name"): p for p in requires.get("custom_node_packs") or []}
    recorded = {normalize_ref(m.get("name")).split("/")[-1]: m for m in requires.get("models") or []}

    packs, unplaced = {}, []
    missing_classes = sorted({n.get("class_type", "") for n in graph.data.values()
                              if n.get("class_type", "") not in object_info})
    for class_type in missing_classes:
        owner = node_map.get(class_type)
        if owner in (None, UNKNOWN, CORE):
            unplaced.append(class_type)
            continue
        pack = packs.setdefault(owner, {**pack_by_name.get(owner, {"name": owner}), "classes": []})
        pack["classes"].append(class_type)

    missing, listings, lora_values = set(), {}, set()
    for class_type, param, value in graph_models(graph):
        rec = recorded.get(normalize_ref(value).split("/")[-1]) or {}
        if param == "lora_name" or rec.get("directory") == "loras":
            lora_values.add(value)
        if class_type in object_info:
            choices = _combo_options(object_info[class_type]).get(param)
            if choices is not None and value not in choices:
                missing.add(value)
            continue
        folder = rec.get("directory") or ("loras" if value in lora_values else None)
        if folder not in listings:
            listings[folder] = _folder_listing(client, folder)
        if normalize_ref(value) not in listings[folder]:
            missing.add(value)

    # LoRAs are bring-your-own: named, never looked up, never installed.
    loras = sorted(v for v in missing if v in lora_values)
    models = []
    for value in sorted(missing - lora_values):
        base = normalize_ref(value).split("/")[-1]
        rec = dict(recorded.get(base) or {"directory": UNKNOWN, "name": normalize_ref(value),
                                          "source": UNKNOWN})
        if rec.get("url"):
            rec["size"] = head_size(rec["url"], head)
        models.append(rec)
    return {"workflow": manifest.name, "machine": client.url,
            "packs": list(packs.values()), "models": models, "loras": loras,
            "unplaced_classes": unplaced}


def _folder_listing(client, folder) -> set:
    if not folder or folder == UNKNOWN:
        return set()
    r = client._request("GET", f"/api/models/{folder}", timeout=30, return_error_response=True)
    if r.status_code != 200:
        return set()
    try:
        return {normalize_ref(n) for n in r.json()}
    except ValueError:
        return set()


# -- apply through ComfyUI-Manager --------------------------------------------

NO_MANAGER = ("ComfyUI-Manager does not answer on {url}. --enable-manager only works when the "
              "comfyui-manager package is installed in ComfyUI's Python; the ComfyUI log says "
              "'the comfyui-manager package must be installed first' when it is not.")
NO_MANAGER_HELP = ("install it, then restart ComfyUI with --enable-manager: "
                   "<ComfyUI's python> -m pip install -r <ComfyUI>/manager_requirements.txt "
                   "(portable build: python_embeded/python.exe -m pip install -r "
                   "ComfyUI/manager_requirements.txt); then run this again. "
                   "To see the plan without Manager: comfyrack setup <workflow>")


def manager_api(client) -> str | None:
    """'v2' for the bundled Manager 4.x, 'legacy' for the custom-node Manager, else None."""
    for api, path in (("v2", "/api/v2/manager/version"), ("legacy", "/api/manager/version")):
        r = client._request("GET", path, timeout=30, return_error_response=True)
        if r.status_code == 200 and not r.text.lstrip().startswith("<"):
            return api
    return None


def _pack_request(api, pack, ui_id, client_id):
    reg = pack.get("registry_id")
    if api == "v2":
        params = ({"id": reg, "version": "latest", "selected_version": "latest"} if reg else
                  {"id": pack["name"], "version": "unknown", "selected_version": "unknown"})
        params.update(mode="remote", channel="default")
        return "/api/v2/manager/queue/task", {"ui_id": ui_id, "client_id": client_id,
                                              "kind": "install", "params": params}
    body = ({"id": reg, "version": "latest", "selected_version": "latest"} if reg else
            {"id": pack["name"], "version": "unknown", "selected_version": "unknown",
             "files": [pack.get("git")]})
    body.update(channel="default", mode="remote", ui_id=ui_id)
    return "/api/manager/queue/install", body


def _model_request(api, model, ui_id, client_id, lookups):
    filename = model["name"].split("/")[-1]
    listed = lookups.model_list_entry(filename) or {}
    body = {"name": filename, "filename": filename, "url": model["url"],
            "type": listed.get("type", model["directory"]),
            "base": listed.get("base", ""),
            "save_path": listed.get("save_path") if listed.get("url") == model["url"]
            else model["directory"], "ui_id": ui_id}
    if api == "v2":
        body["client_id"] = client_id
        return "/api/v2/manager/queue/install_model", body
    return "/api/manager/queue/install_model", body


def apply(client, the_plan, lookups=None, poll=2.0, timeout_s=7200, on_progress=None) -> dict:
    """Queue every planned pack and model on Manager, start the queue, wait for it.

    Returns {api, queued: [...], refused: [...], results: {ui_id: text}, restart: bool}.
    Packs Manager refuses come back in `refused` with their git URL; nothing else is tried."""
    from .errors import ComfyrackError

    api = manager_api(client)
    if api is None:
        raise ComfyrackError(NO_MANAGER.format(url=client.url),
                             help_text=NO_MANAGER_HELP)
    lookups = lookups or Lookups()
    client_id = f"comfyrack-{uuid.uuid4().hex[:8]}"
    queued, refused = [], []
    started = time.time()
    last = None
    items = [("pack", p) for p in the_plan["packs"]]
    items += [("model", m) for m in the_plan["models"]]
    for kind, item in items:
        label = item["name"]
        if kind == "model" and not item.get("url"):
            refused.append({"kind": kind, "name": label, "reason": "no download URL on record"})
            continue
        ui_id = f"{kind}:{label}"
        path, body = (_pack_request(api, item, ui_id, client_id) if kind == "pack"
                      else _model_request(api, item, ui_id, client_id, lookups))
        r = client._request("POST", path, json=body, timeout=60, return_error_response=True)
        if r.status_code == 200:
            queued.append({"kind": kind, "name": label, "ui_id": ui_id})
        else:
            refused.append({"kind": kind, "name": label, "status": r.status_code,
                            "reason": (r.text or "").strip()[:300],
                            **({"git": item.get("git")} if kind == "pack" else
                               {"url": item.get("url")})})
    results = {}
    if queued:
        prefix = "/api/v2/manager/queue" if api == "v2" else "/api/manager/queue"
        client._request("POST", f"{prefix}/start", timeout=60, return_error_response=True)
        deadline = time.time() + timeout_s
        while True:
            params = {"client_id": client_id} if api == "v2" else None
            status = client._request("GET", f"{prefix}/status", params=params, timeout=30).json()
            done = min(status.get("done_count", 0), len(queued))
            if on_progress and done != last:
                last = done
                on_progress(f"manager: {done} of {len(queued)} installed "
                            f"({int(time.time() - started)}s)")
            if not status.get("is_processing") and not status.get("pending_count") \
                    and status.get("done_count", 0) >= status.get("total_count", 0):
                break
            if time.time() >= deadline:
                raise ComfyrackError(f"Manager queue still busy after {timeout_s}s",
                                     help_text="check the ComfyUI log, then rerun setup")
            time.sleep(poll)
        if api == "v2":
            r = client._request("GET", f"{prefix}/history", params={"client_id": client_id},
                                timeout=30, return_error_response=True)
            hist = r.json() if r.status_code == 200 else {}
            for ui_id, item in (hist or {}).items():
                st = (item or {}).get("status") or {}
                results[ui_id] = st.get("status_str") or (item or {}).get("result")
    restart = any(q["kind"] == "pack" for q in queued)
    return {"api": api, "queued": queued, "refused": refused, "results": results,
            "restart": restart}


# -- writing the record --------------------------------------------------------

def dump_requires(requires: dict) -> str:
    import yaml
    return yaml.safe_dump({"requires": requires}, sort_keys=False, width=100,
                          allow_unicode=True)


def replace_requires(text: str, requires: dict) -> str:
    """Manifest text with its top-level `requires:` block replaced (or added before
    `flags:`). The rest of the file keeps its formatting, so a backfill diff shows only
    the record."""
    lines = text.splitlines(keepends=True)
    block = dump_requires(requires)
    start = next((i for i, l in enumerate(lines) if l.startswith("requires:")), None)
    if start is None:
        at = next((i for i, l in enumerate(lines) if l.startswith("flags:")), len(lines))
        if at == len(lines) and lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return "".join(lines[:at]) + block + "".join(lines[at:])
    end = start + 1
    while end < len(lines) and (lines[end][:1] in (" ", "-", "\n", "\r") or not lines[end].strip()):
        end += 1
    return "".join(lines[:start]) + block + "".join(lines[end:])
