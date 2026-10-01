"""Load -> resolve -> patch -> preflight -> submit.

Orchestrators needing structural graph surgery use load_graph() + patch() +
submit() directly; everything else uses run()."""
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import errors, lora_meta, preflight, provenance
from .convert import load_workflow
from .graph import Graph
from .lexicon import DEFAULT_LINEAGE_MAP
from .recipes import resolve_values

# How long submit() will wait for the progress thread to flush its last line once
# the job is already finished. The job's result is in hand by then, so a socket
# that never saw its terminal frame must not hold it up.
PROGRESS_JOIN_TIMEOUT_S = 2.0


@dataclass
class RunResult:
    paths: list = field(default_factory=list)
    text: str | None = None
    # title -> {"paths": [...]} or {"text": "..."}; empty unless the manifest has `outputs:`
    outputs: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)


def load_graph(manifest, client) -> Graph:
    """Fresh Graph per call (a few KB of JSON) so one caller's mutations never leak
    into another's. UI-format workflows are converted using the target's node defs.

    The node defs are passed as a CALLABLE, not a fetched dict: only a UI-format
    workflow needs them, and all 100 builtin workflows are already API format, so
    fetching eagerly spent a multi-MB /object_info round trip on every registered
    workflow to convert nothing."""
    object_info = (lambda: client.object_info()) if client is not None else None
    return load_workflow(manifest.workflow_path, object_info=object_info)


def _is_local_file(value) -> bool:
    """True only for a value that names a file on THIS machine.

    A `type: image` value is one of two different things and the difference is not
    in the manifest: either a path the caller wants uploaded, or a server-side ref
    that already lives in ComfyUI's input dir. resolve_values() falls back to the
    value baked into the workflow JSON, and a LoadImage node always has a filename
    baked in -- so treating every image value as a path to open() made
    `comfyrack run <wf>` crash with FileNotFoundError on 67 of the 100 builtin
    manifests, whose own help strings say the value IS a server-side ref."""
    if not isinstance(value, (str, Path)) or not str(value):
        return False
    try:
        return Path(value).is_file()
    except OSError:
        # A server-side ref can be any string, including one the local OS rejects
        # as a path (embedded NUL, over MAX_PATH, a reserved Windows name).
        return False


def patch(graph: Graph, manifest, values: dict, client=None) -> Graph:
    for name, flag in manifest.flags.items():
        if name not in values:
            continue
        value = values[name]
        if flag.type == "image" and _is_local_file(value):
            if client is None:
                raise errors.UsageError(
                    f"image flag '{name}' names a local file ({value!r}) and needs a "
                    f"client to upload it",
                    help_text="pass client= to patch() so the image can be uploaded to "
                              "the target machine, or use run() which supplies one.",
                )
            value = client.upload_image(value)
        node_id = graph.resolve(node_id=flag.node, title=flag.node_title)
        graph.set_param(node_id, flag.param, value)
    return graph


def _lineage_of(raw) -> str | None:
    """The lineage a LoRA's ss_base_model_version names, or None for a string this
    package has no mapping for.

    A LoRA header records the CHECKPOINT it was trained against ("anima"); a
    family.yaml records that checkpoint's LINEAGE ("illustrious"). Comparing the
    two raw would report a mismatch for every anima LoRA run on the anima family
    -- a false hard failure on this project's primary family, on the very first
    real run of a check that had never seen real data. DEFAULT_LINEAGE_MAP is the
    same mapping `lex sync` uses, so both sides of the comparison speak lineage.

    Unknown is deliberately NOT "different". A trainer writes this field freely
    ("flux1-dev", "sdxl_base_v1-0", whatever Civitai's baseModel said), and
    calling an unrecognised string a mismatch would abort a valid run over
    vocabulary rather than over a real problem -- an outcome worse than the
    degraded render the check exists to prevent. So an unrecognised value returns
    None and that LoRA drops out of the check entirely. Matching the leading
    alphabetic token as a second pass ("flux1-dev" -> flux, "sdxl_base_v1-0" ->
    sdxl) keeps that exit rare instead of routine."""
    key = str(raw or "").lower()
    if key in DEFAULT_LINEAGE_MAP:
        return DEFAULT_LINEAGE_MAP[key]
    head = re.split(r"[^a-z]", key, maxsplit=1)[0]
    return DEFAULT_LINEAGE_MAP.get(head)


def lora_lineages(loras_dir) -> dict:
    """model reference -> {"base_model_family": lineage}, the mapping
    preflight.check wants.

    Keys are `lora_meta.scan()`'s: the path relative to `loras_dir`, separator-
    normalised, which is the same shape a workflow's `lora_name` widget holds.
    preflight.check resolves them through `discover.ref_lookup`, so a subfoldered
    LoRA is matched whichever side carries the subfolder -- it used to drop out
    of the check entirely, silently, while dispatch loudly called the same file
    unroutable.

    The translation happens HERE rather than in preflight.check, which is
    deliberately free of both config and disk: scan() returns LoraMeta objects
    over a configured directory, check() reads a plain mapping.

    An unset or absent directory is an empty mapping, which leaves the check
    dormant -- the behaviour a project that never configured `[paths] loras` has
    today, not a new failure mode."""
    if not loras_dir or not Path(loras_dir).is_dir():
        return {}
    out = {}
    for ref, meta in lora_meta.scan(loras_dir).items():
        lineage = _lineage_of(meta.base_model_family)
        if lineage:
            out[ref] = {"base_model_family": lineage}
    return out


def preflight_graph(registry, client, manifest, graph, loras_dir=None) -> list:
    """Single site for lineage resolution: both halves of the comparison are
    resolved here, the checkpoint's from the family and the LoRA's from disk."""
    fam = registry.family(manifest.family)
    return preflight.check(graph, client.object_info(),
                           lora_meta=lora_lineages(loras_dir),
                           checkpoint_lineage=fam.base_model_family if fam else None,
                           requires=manifest.requires)


def format_progress(event: dict) -> str | None:
    """One human-readable line for a websocket event, or None for one with nothing
    to say. Formatting lives here, not in the callback, so every caller that wants
    progress gets the same wording for free and only decides WHERE it goes."""
    kind = event.get("type")
    data = event.get("data") or {}
    if kind == "progress":
        value, maximum = data.get("value"), data.get("max")
        if isinstance(value, int) and isinstance(maximum, int) and maximum > 0:
            return f"step {value}/{maximum} ({100 * value // maximum}%)"
        return None
    if kind == "executing":
        node = data.get("node")
        return "finished" if node is None else f"executing node {node}"
    if kind == "execution_cached":
        nodes = data.get("nodes") or []
        return f"cached {len(nodes)} node(s)" if nodes else None
    return None


class _ProgressStopped(Exception):
    """Raised inside the progress callback to unwind out of ComfyClient.progress()'s
    receive loop. progress() does not wrap on_event, so this propagates through its
    `finally: ws.close()` -- which is the point: it both stops the pump and closes
    the socket at the first frame after the stop signal, instead of leaving it
    parked in recv() until the socket's own timeout."""


class _Reporter:
    """Handle for the progress thread: the join alone is not enough to end it."""

    def __init__(self, thread, stop):
        self.thread = thread
        self.stop = stop

    def finish(self, timeout: float) -> None:
        """Wait briefly for a clean exit, then REVOKE the callback either way.

        Setting the flag after the join is what keeps the last real line while
        making the thread inert: whatever it does afterwards, it can no longer
        reach on_progress."""
        self.thread.join(timeout)
        self.stop.set()


def _start_progress(client, prompt_id: str, on_progress):
    """Stream progress on a background thread WHILE the caller polls the job record.

    progress() and wait() both block until the job finishes, so running them in
    sequence would double a five-minute render's wall clock. Nor can progress()
    replace wait(): it returns None, discards binary frames on principle, and
    returns early and silently when a socket drops -- the result, and the
    execution error if there is one, exist only in the job/history record (see
    http.py's module docstring, and the two silent-wrong-result bugs it records).
    So the websocket runs ALONGSIDE the poll, and the poll stays the single
    source of truth.

    The thread is a daemon and is never joined for more than
    PROGRESS_JOIN_TIMEOUT_S: once wait() returns the job is over, and a socket
    still waiting on a frame that will never come must not delay the result.
    Walking away from it is not enough, though. submit() starts this BEFORE it
    queues, so a job that finishes before the websocket connects leaves the
    thread parked in recv() with a live socket for its full timeout -- and in a
    batch, shot N's "step 10/20" would then print during shot N+1. So the join is
    paired with a stop flag (_Reporter.finish): once set, the pump drops the
    callback and unwinds instead of delivering into someone else's run.

    Exceptions are swallowed for the same reason binary frames are discarded --
    this channel is cosmetic. A missing websocket library, a proxy that refuses
    the upgrade, or a callback that raises must not destroy work the GPU already
    did. Returns None (opening nothing) when no callback was supplied, so library
    and test callers never pay for a socket nobody reads.

    The callback is invoked from this thread; a caller doing more than writing a
    line needs its own locking."""
    stream = getattr(client, "progress", None)
    if on_progress is None or stream is None:
        return None
    stop = threading.Event()

    def pump():
        def on_event(event):
            if stop.is_set():
                raise _ProgressStopped()
            text = format_progress(event)
            if text is not None:
                on_progress(text)
        try:
            stream(prompt_id, on_event)
        except Exception:
            return

    thread = threading.Thread(target=pump, daemon=True,
                              name=f"comfyrack-progress-{prompt_id}")
    thread.start()
    return _Reporter(thread, stop)


def _save_images(client, items, out_path, base, record) -> list:
    paths = []
    for i, item in enumerate(items):
        data = client.view(item["filename"], item.get("subfolder", ""),
                           item.get("type", "output"))
        if out_path:
            if i == 0:
                dest = Path(out_path)
            else:
                # Derive from out_path: foo.png -> foo_2.png, foo_3.png, etc.
                out_p = Path(out_path)
                dest = out_p.parent / f"{out_p.stem}_{i + 1}{out_p.suffix}"
        else:
            dest = base / Path(item["filename"]).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        if record:
            provenance.write_sidecar(dest, record)
        paths.append(dest)
    return paths


def _extra_outputs(client, graph, manifest, entry, base, record, primary_paths) -> dict:
    """Every node the manifest's `outputs:` lists, keyed by title. The primary node
    reuses the files already written; --out applies to it alone."""
    result = {}
    for spec in manifest.outputs:
        title, kind = spec["title"], spec["kind"]
        if title == manifest.output_node_title and kind != "text":
            result[title] = {"paths": [str(p) for p in primary_paths]}
            continue
        items = client.outputs(entry, graph.resolve(title=title))
        if kind == "text":
            result[title] = {"text": "\n".join(str(i) for i in items).strip()}
        else:
            result[title] = {"paths": [str(p) for p in
                                       _save_images(client, items, None, base, record)]}
    return result


def submit(client, graph: Graph, manifest, out_path=None, results_dir=None,
           record: dict | None = None, on_progress=None) -> RunResult:
    """`on_progress` takes one preformatted line per progress event. A CLI passes
    one that writes to STDERR: stdout carries the machine-readable result and has
    to stay parseable, which is why progress cannot go there.

    The prompt_id is generated HERE, before anything is queued, and the progress
    pump starts before queue_prompt: a job that finishes before the websocket
    connects would otherwise be invisible to the pump. queue_prompt() is handed
    the same id, so the pump, the poll, and the server all agree from the start."""
    prompt_id = str(uuid.uuid4())
    reporter = _start_progress(client, prompt_id, on_progress)
    try:
        client.queue_prompt(graph.data, prompt_id=prompt_id)
        entry = client.wait(prompt_id)
    finally:
        if reporter is not None:
            reporter.finish(PROGRESS_JOIN_TIMEOUT_S)
    node_id = graph.resolve(title=manifest.output_node_title)
    items = client.outputs(entry, node_id)

    if out_path and Path(out_path).is_dir():
        # --out names a folder: write into it under the default file names.
        base, out_path = Path(out_path), None
    else:
        base = Path(out_path).parent if out_path else Path(
            results_dir or Path.cwd()) / manifest.family / manifest.name

    if manifest.output_type == "text":
        # No artifact on disk, so no sidecar -- see provenance.py's module docstring
        # for why that is the contract rather than an omission. The record still
        # rides out on RunResult.record.
        text = str(items[0]).strip() if items else ""
        extra = _extra_outputs(client, graph, manifest, entry, base, record, [])
        return RunResult(paths=[], text=text, outputs=extra, record=record or {})

    base.mkdir(parents=True, exist_ok=True)
    paths = _save_images(client, items, out_path, base, record)
    extra = _extra_outputs(client, graph, manifest, entry, base, record, paths)
    return RunResult(paths=paths, text=None, outputs=extra, record=record or {})


def run(registry, client, name: str, recipe=None, overrides=None, out_path=None,
        results_dir=None, skip_preflight: bool = False,
        recipe_name: str | None = None, loras_dir=None,
        on_progress=None) -> RunResult:
    """`loras_dir` feeds preflight's lineage check and is scanned ONCE per run --
    scan() opens every safetensors header under it, and a directory of 205 LoRAs
    rescanned per node is thousands of file reads for one graph."""
    manifest = registry.load(name)
    graph = load_graph(manifest, client)
    values = resolve_values(manifest, graph, recipe=recipe, overrides=overrides)
    graph = patch(graph, manifest, values, client=client)

    if not skip_preflight:
        problems = preflight_graph(registry, client, manifest, graph,
                                   loras_dir=loras_dir)
        preflight.raise_if_problems(problems, getattr(client, "url", "the target machine"))

    record = provenance.build_record(manifest, values,
                                     getattr(client, "url", ""), graph,
                                     recipe_name=recipe_name)
    return submit(client, graph, manifest, out_path=out_path,
                  results_dir=results_dir, record=record, on_progress=on_progress)
