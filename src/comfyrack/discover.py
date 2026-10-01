"""ComfyUI introspection. These functions replace the comfyui MCP servers:
node types, node schemas, model lists, embeddings, queue depth, and GPU stats all
become subcommands, so no tool schemas are loaded into an agent's context."""
from .errors import ComfyrackError, NotFoundError
from .graph import combo_choices, widget_names

# Loader node class -> the model category its combo widget enumerates.
MODEL_LOADERS = {
    "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
    "LoraLoader": ("lora_name", "loras"),
    "LoraLoaderModelOnly": ("lora_name", "loras"),
    "VAELoader": ("vae_name", "vae"),
    "UNETLoader": ("unet_name", "unet"),
    "CLIPLoader": ("clip_name", "clip"),
    "ControlNetLoader": ("control_net_name", "controlnet"),
    "UpscaleModelLoader": ("model_name", "upscale_models"),
}


def normalize_ref(name) -> str:
    """A model reference in one comparison-safe spelling.

    ComfyUI names a model by its path RELATIVE to the models directory, using the
    HOST's separator -- a Windows machine reports `Anima\\AnimaEditV1.safetensors`
    where a Linux machine reports `Anima/AnimaEditV1.safetensors` for the same file.
    13 of this project's 206 real LoRAs live in a subfolder, and routing compares
    refs that came from different machines, so the separator has to stop
    mattering before anything is compared."""
    return str(name or "").replace("\\", "/").strip("/")


def ref_index(mapping) -> dict:
    """Index a mapping (or any iterable of names) by every spelling a caller may
    look it up by: the normalised reference, and the bare filename.

    The bare alias exists because the two sides of a real comparison do not
    always agree about the subfolder. A character recipe hand-writes
    `id_lora: AnimaEditV1.safetensors`; the machine enumerates
    `Anima/AnimaEditV1.safetensors`. Treating those as different files makes
    `dispatch._missing()` declare a LoRA that is sitting right there permanently
    unroutable.

    A bare name claimed by two different subfolders is registered by NEITHER --
    same rule `Manifest._build_alias_map` follows, and for the same reason:
    silently picking one would resolve to the wrong file."""
    items = (mapping.items() if hasattr(mapping, "items")
             else ((n, n) for n in mapping))
    pairs = [(normalize_ref(k), v) for k, v in items]
    counts = {}
    for norm, _v in pairs:
        bare = norm.rsplit("/", 1)[-1]
        counts[bare] = counts.get(bare, 0) + 1
    index = {}
    for norm, value in pairs:
        index[norm] = value
    for norm, value in pairs:
        bare = norm.rsplit("/", 1)[-1]
        if counts[bare] == 1:
            index.setdefault(bare, value)
    return index


def ref_lookup(index: dict, name):
    """Resolve `name` against a `ref_index()`, tolerating a subfolder on either
    side. Returns None when nothing matches."""
    norm = normalize_ref(name)
    if norm in index:
        return index[norm]
    return index.get(norm.rsplit("/", 1)[-1])


def node_types(client, pattern: str | None = None) -> list:
    names = sorted(client.object_info().keys())
    if pattern:
        p = pattern.lower()
        names = [n for n in names if p in n.lower()]
    return names


def node_info(client, class_type: str) -> dict:
    oi = client.object_info(class_type)
    if class_type not in oi:
        # `comfyrack node NoSuchNode` used to give a bare KeyError traceback where
        # the sibling `comfyrack nodes NoSuch` gives a clean empty state.
        raise NotFoundError(
            f"no node type {class_type!r} on this machine",
            help_text=f"comfyrack nodes {class_type[:12]}  # search installed node types",
        )
    entry = oi[class_type]
    spec = entry.get("input", {})
    inputs = {}
    for section in ("required", "optional"):
        for name, val in spec.get(section, {}).items():
            inputs[name] = val[0] if isinstance(val, list) and val else val
    return {
        "class_type": class_type,
        "inputs": inputs,
        "widgets": widget_names(entry),
        "outputs": entry.get("output", []),
    }


def models(client, type_: str | None = None) -> dict:
    oi = client.object_info()
    out = {}
    for class_type, (param, category) in MODEL_LOADERS.items():
        entry = oi.get(class_type)
        if not entry:
            continue
        spec = entry.get("input", {}).get("required", {})
        val = spec.get(param)
        choices = combo_choices(val)
        if choices is not None:
            out.setdefault(category, [])
            for name in choices:
                if name not in out[category]:
                    out[category].append(name)
    if type_:
        return {type_: out.get(type_, [])}
    return out


def embeddings(client) -> list:
    """Textual-inversion names installed on the target machine, sorted.

    Read from GET /models/embeddings (see ComfyClient.embeddings for why NOT the
    bare /embeddings path, which a real build serves as an HTML page). Plan 1
    Task 18 confirmed against a real ComfyUI that `_embeddings` is ABSENT from
    /object_info, re-confirmed here against 0.28.0, so the previous
    /object_info-only lookup returned `[]` for a machine that had embeddings
    installed -- a function that looks authoritative while knowing nothing.

    Names come back as the server lists them: extensions included, subfolders
    prefixed. The /object_info fallback below lists them stripped, so the two
    sources do NOT agree on shape. That difference is left visible rather than
    normalised away, because no machine available here has an embedding installed
    to verify a normalisation against.

    /object_info stays as a fallback for builds that expose `_embeddings` and lack
    the route. It is consulted only when the route is missing (a 404), answers with
    something that is not JSON (an HTML page), or yields a non-list. An empty list
    from a working route is authoritative -- that machine has no embeddings -- and
    re-asking /object_info would spend a multi-MB fetch to learn the same thing."""
    fetch = getattr(client, "embeddings", None)
    if fetch is not None:
        try:
            names = fetch()
        except ComfyrackError:
            names = None
        if isinstance(names, list):
            return sorted(str(n) for n in names)

    raw = client.object_info().get("_embeddings")
    return sorted(str(n) for n in raw) if isinstance(raw, list) else []


def queue_summary(client) -> dict:
    q = client.queue_status()
    return {"running": len(q.get("queue_running", [])),
            "pending": len(q.get("queue_pending", []))}


def stats_summary(client) -> dict:
    s = client.system_stats()
    devices = s.get("devices") or [{}]
    d = devices[0]
    gb = 1024 ** 3
    return {
        "device": d.get("name", "unknown"),
        "vram_total_gb": round(d.get("vram_total", 0) / gb, 1),
        "vram_free_gb": round(d.get("vram_free", 0) / gb, 1),
    }


FORM_TIMEOUT_S = 5


def form_meta(manifest, graph, client) -> dict | None:
    """flag name -> {choices/min/max/min_length/max_length} read from the target's
    /object_info for each flag's node class and param, or None when nothing answers.

    Scoped per-class requests, not the multi-MB full listing: a form needs a handful
    of classes. A short timeout keeps `describe` quick against a machine that is off."""
    meta = {}
    try:
        for name, flag in manifest.flags.items():
            if flag.node is None and flag.node_title is None:
                continue
            node_id = graph.resolve(node_id=flag.node, title=flag.node_title)
            class_type = graph.data[node_id].get("class_type", "")
            info = client.object_info(class_type, timeout=FORM_TIMEOUT_S).get(class_type)
            if not info:
                continue
            spec = None
            for section in ("required", "optional"):
                spec = (info.get("input") or {}).get(section, {}).get(flag.param, spec)
            if not isinstance(spec, list) or not spec:
                continue
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            entry = {}
            # All three combo shapes: old list, ["COMBO", {"options": [...]}],
            # and ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": ...}, ...]}]
            # (whose choices are the option keys).
            choices = combo_choices(spec)
            if isinstance(choices, list):
                entry["choices"] = choices
            for key in ("min", "max", "min_length", "max_length"):
                if opts.get(key) is not None:
                    entry[key] = opts[key]
            if entry:
                meta[name] = entry
    except (ComfyrackError, OSError, ValueError):
        return None
    return meta
